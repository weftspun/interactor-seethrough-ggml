// Copy-cost analysis over a built ggml graph.
//
// Motivated by a measured negative result: eliminating two materialising
// copies per cross_frame_block in favour of one (bit-identical output, sha256
// f14427a0..82299) ran 6% SLOWER -- 373.3s against 351.2s at res 1280.
//
// Copy COUNT is not a cost model. The contract in
// 2-contract/tensor-copy-cost-model proves the general form of that: there
// exist configurations with strictly fewer copies and strictly greater cost.
//
// But that contract models cost by the innermost contiguous run, and the
// regression above does not fit it: both the old and new permutations leave
// dim 0 innermost, so the contiguous run is identical. What changed was the
// stride at which those runs are GATHERED -- from hd*n_head to hd*n_head*S,
// a factor of S (~6400 at the 80x80 latent level). Same bytes, same run
// length, consecutive runs now pages apart instead of adjacent.
//
// So this reports both quantities per copy, because guessing which one
// dominates is exactly the mistake that produced the regression.
#pragma once

#include "ggml.h"

#include <cstdint>

struct CopyCost {
	const char *name; // node name (may be null)
	int64_t elements; // total elements moved
	int64_t contig_run; // innermost contiguous run, in elements
	int64_t gather_stride; // bytes between consecutive runs (0 = contiguous)
	int64_t lines; // cache lines touched (elements / min(run, line))
	bool page_scattered; // gather_stride >= a 4KiB page
};

// Walk `gf` and report the materialising copies (CONT/CPY/DUP) ranked by
// estimated cost. `top_n` limits the per-copy detail lines; totals always
// cover every copy in the graph.
//
// Writes to stderr in the existing `[perf]` style so it lands in the same logs
// as the op counts it sits beside.
// Takes a non-const graph because ggml's public accessors
// (ggml_graph_n_nodes / ggml_graph_node) do; the graph is not modified.
void copy_cost_report(ggml_cgraph *gf, const char *label, int top_n = 12);
