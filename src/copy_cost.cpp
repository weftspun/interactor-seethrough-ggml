#include "copy_cost.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {

// A 64-byte line holds 16 f32s. Kept as bytes so mixed-precision graphs get
// the right element count per line without a second constant.
constexpr int64_t kLineBytes = 64;
constexpr int64_t kPageBytes = 4096;

// Length, in elements, of the contiguous prefix of `t`: how many elements can
// be read sequentially before the layout forces a jump. A permuted view that
// keeps dim 0 innermost still has a contiguous run of ne[0] -- which is why
// this alone did not explain the cross_frame_block regression.
int64_t contig_run(const ggml_tensor *t) {
	const size_t ts = ggml_type_size(t->type);
	int64_t run = 1;
	size_t expect = ts;
	for (int d = 0; d < GGML_MAX_DIMS; d++) {
		if ((size_t)t->nb[d] != expect) {
			break;
		}
		run *= t->ne[d];
		expect *= (size_t)t->ne[d];
	}
	return run;
}

// Byte stride between consecutive contiguous runs: the distance the read head
// jumps when the run ends. 0 means fully contiguous (no jump). This is the
// quantity that changed in the regression -- the runs stayed the same length,
// but moved from adjacent to pages apart.
int64_t gather_stride(const ggml_tensor *t) {
	const size_t ts = ggml_type_size(t->type);
	size_t expect = ts;
	for (int d = 0; d < GGML_MAX_DIMS; d++) {
		if ((size_t)t->nb[d] != expect) {
			return (int64_t)t->nb[d];
		}
		expect *= (size_t)t->ne[d];
	}
	return 0;
}

bool is_materialising(enum ggml_op op) {
	return op == GGML_OP_CONT || op == GGML_OP_CPY || op == GGML_OP_DUP;
}

} // namespace

void copy_cost_report(ggml_cgraph *gf, const char *label, int top_n) {
	std::vector<CopyCost> costs;
	costs.reserve(64);

	const int n_nodes = ggml_graph_n_nodes(gf);
	for (int i = 0; i < n_nodes; i++) {
		const ggml_tensor *node = ggml_graph_node(gf, i);
		if (!is_materialising(node->op)) {
			continue;
		}
		const ggml_tensor *src = node->src[0];
		if (!src) {
			continue;
		}

		const int64_t n = ggml_nelements(src);
		const int64_t run = contig_run(src);
		const int64_t stride = gather_stride(src);
		const int64_t line_elems =
				std::max<int64_t>(1, kLineBytes / (int64_t)ggml_type_size(src->type));
		// Cache lines touched: a run shorter than a line still pays for a
		// whole line, so the divisor is min(run, line_elems). This is
		// `cost L n r = n / min r L` from the Lean contract, in elements.
		const int64_t lines = n / std::max<int64_t>(1, std::min(run, line_elems));

		costs.push_back(CopyCost{
				node->name[0] ? node->name : nullptr,
				n,
				run,
				stride,
				lines,
				stride >= kPageBytes,
		});
	}

	if (costs.empty()) {
		fprintf(stderr, "[perf] %s copies: none\n", label);
		return;
	}

	int64_t total_elems = 0, total_lines = 0, scattered = 0, scattered_elems = 0;
	for (const CopyCost &c : costs) {
		total_elems += c.elements;
		total_lines += c.lines;
		if (c.page_scattered) {
			scattered++;
			scattered_elems += c.elements;
		}
	}

	// Totals first: these are the numbers a decision should be made on. A
	// count on its own is what produced the regression this file documents.
	fprintf(stderr,
			"[perf] %s copies: n=%zu elems=%lld lines=%lld page_scattered=%lld (%.1f%% of elems)\n",
			label, costs.size(), (long long)total_elems, (long long)total_lines,
			(long long)scattered,
			total_elems ? 100.0 * (double)scattered_elems / (double)total_elems : 0.0);

	// Rank by lines touched, then by whether the gather is page-scattered --
	// the two costs are not commensurable, so both are shown rather than
	// folded into one invented weight.
	std::sort(costs.begin(), costs.end(), [](const CopyCost &a, const CopyCost &b) {
		if (a.lines != b.lines) {
			return a.lines > b.lines;
		}
		return a.gather_stride > b.gather_stride;
	});

	const int shown = std::min<int>(top_n, (int)costs.size());
	for (int i = 0; i < shown; i++) {
		const CopyCost &c = costs[i];
		fprintf(stderr,
				"[perf]   %2d. elems=%-10lld run=%-6lld stride=%-9lld lines=%-9lld%s %s\n",
				i + 1, (long long)c.elements, (long long)c.contig_run,
				(long long)c.gather_stride, (long long)c.lines,
				c.page_scattered ? " PAGE-SCATTERED" : "",
				c.name ? c.name : "");
	}
	if (shown < (int)costs.size()) {
		fprintf(stderr, "[perf]   ... %zu more copies not shown (totals above cover all)\n",
				costs.size() - (size_t)shown);
	}
}
