// Minimal Metal NaN probe: load the UNet, run one forward pass with
// random sample + zero text embeddings, check if output is NaN.
//   test_metal_nan <layerdiff-unet.gguf>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ops.h"
#include "unet_frame.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s layerdiff-unet.gguf\n", argv[0]);
		return 1;
	}
	setvbuf(stdout, nullptr, _IONBF, 0);

	// pick first GPU device (Metal on macOS)
	ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
	if (!d) {
		d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
	}
	if (!d) {
		fprintf(stderr, "no GPU\n");
		return 1;
	}
	fprintf(stderr, "device: %s\n", ggml_backend_dev_name(d));

	Model m;
	if (!m.load_backend(argv[1], ggml_backend_dev_buffer_type(d))) {
		fprintf(stderr, "load failed\n");
		return 1;
	}
	fprintf(stderr, "weights: %zu tensors\n", m.weights.size());

	// disable all Vulkan workarounds
	m.flash_attn = false;
	m.tiled_naive_attn = false;
	m.direct_conv = false;
	m.conv_row_chunk = false;

	const int F = 13, ZR = 96, RES = 768;
	size_t max_nodes = 294912;
	size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
	ggml_init_params ip = { meta, nullptr, true };
	m.ctx_g = ggml_init(ip);
	ggml_context *ctx = m.ctx_g;

	ggml_tensor *sample = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ZR, ZR, 8, F);
	ggml_tensor *ehs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2048, 77, F);
	ggml_tensor *text = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1280, F);
	ggml_tensor *tids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 6, F);
	ggml_tensor *ts = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, F);
	for (auto *t : { sample, ehs, text, tids, ts }) {
		ggml_set_input(t);
	}

	auto emb = time_embed_mlp(m, ggml_timestep_embedding(ctx, ts, 320, 10000), "time_embedding");
	auto aug = ggml_add(ctx, text, group_embedding(m, text, "group_embeds.0"));
	emb = ggml_add(ctx, emb, sdxl_add_embed(m, aug, tids));
	auto ehs2 = ggml_add(ctx, ehs, group_embedding(m, ehs, "group_embeds2.0"));
	auto out = unet_frame_forward(m, sample, emb, ehs2);
	ggml_set_output(out);

	ggml_backend_t backend = ggml_backend_dev_init(d, nullptr);
	ggml_cgraph *gf = ggml_new_graph_custom(ctx, max_nodes, false);
	ggml_build_forward_expand(gf, out);
	ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
	if (!ggml_gallocr_alloc_graph(alloc, gf)) {
		fprintf(stderr, "alloc failed\n");
		return 1;
	}

	// set inputs: random sample, zero embeddings
	std::vector<float> samp(ZR * ZR * 8 * F);
	for (auto &v : samp) {
		v = (float)rand() / RAND_MAX * 2 - 1;
	}
	ggml_backend_tensor_set(sample, samp.data(), 0, samp.size() * 4);
	std::vector<float> zeros_ehs((size_t)2048 * 77 * F, 0);
	ggml_backend_tensor_set(ehs, zeros_ehs.data(), 0, zeros_ehs.size() * 4);
	std::vector<float> zeros_text((size_t)1280 * F, 0);
	ggml_backend_tensor_set(text, zeros_text.data(), 0, zeros_text.size() * 4);
	std::vector<float> tid_v(6 * F);
	for (int f = 0; f < F; f++) {
		float ids[6] = { (float)RES, (float)RES, 0, 0, (float)RES, (float)RES };
		for (int i = 0; i < 6; i++) {
			tid_v[f * 6 + i] = ids[i];
		}
	}
	ggml_backend_tensor_set(tids, tid_v.data(), 0, tid_v.size() * 4);
	std::vector<float> ts_v(F, 999.0f);
	ggml_backend_tensor_set(ts, ts_v.data(), 0, F * 4);

	fprintf(stderr, "computing...\n");
	if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
		fprintf(stderr, "compute failed\n");
		return 1;
	}

	std::vector<float> result(ggml_nelements(out));
	ggml_backend_tensor_get(out, result.data(), 0, result.size() * 4);

	int nan_count = 0, inf_count = 0;
	double max_abs = 0;
	for (size_t i = 0; i < result.size(); i++) {
		if (std::isnan(result[i])) {
			nan_count++;
		} else if (std::isinf(result[i])) {
			inf_count++;
		} else {
			max_abs = std::max(max_abs, std::abs((double)result[i]));
		}
	}
	printf("output: %zu elems, nan=%d inf=%d max_abs=%.4f\n", result.size(), nan_count, inf_count, max_abs);
	printf("%s\n", nan_count == 0 ? "PASS" : "FAIL (NaN detected)");

	ggml_gallocr_free(alloc);
	ggml_free(ctx);
	return nan_count > 0 ? 1 : 0;
}
