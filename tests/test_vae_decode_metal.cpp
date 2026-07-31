// Test trans_vae_decode on Metal: load both VAEs, decode a random latent,
// check if the output has non-zero alpha.
//   test_vae_decode_metal <layerdiff-vae.gguf> <trans-vae.gguf>
#include "ops.h"
#include "vae.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s layerdiff-vae.gguf trans-vae.gguf\n", argv[0]); return 1; }
    setvbuf(stdout, nullptr, _IONBF, 0);

    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!d) d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    if (!d) { fprintf(stderr, "no GPU\n"); return 1; }
    fprintf(stderr, "device: %s\n", ggml_backend_dev_name(d));

    Model m;
    if (!m.load_backend(argv[1], ggml_backend_dev_buffer_type(d))) { fprintf(stderr, "load vae failed\n"); return 1; }
    if (!m.load_backend(argv[2], ggml_backend_dev_buffer_type(d))) { fprintf(stderr, "load trans-vae failed\n"); return 1; }
    fprintf(stderr, "weights: %zu tensors\n", m.weights.size());

    // disable Vulkan workarounds
    m.flash_attn = false;
    m.tiled_naive_attn = false;
    m.direct_conv = false;
    m.conv_row_chunk = false;

    const int ZR = 96, RES = 768;  // latent 96x96 -> pixel 768x768
    size_t max_nodes = 393216;
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    m.ctx_g = ggml_init(ip);

    ggml_tensor * z = ggml_new_tensor_4d(m.ctx_g, GGML_TYPE_F32, ZR, ZR, 4, 1);
    ggml_set_input(z);

    ggml_tensor * out = trans_vae_decode(m, z);
    ggml_set_output(out);

    ggml_backend_t backend = ggml_backend_dev_init(d, nullptr);
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

    // random latent
    std::vector<float> lat(ZR * ZR * 4);
    for (auto & v : lat) v = (float)rand() / RAND_MAX * 2 - 1;
    ggml_backend_tensor_set(z, lat.data(), 0, lat.size() * 4);

    fprintf(stderr, "computing trans_vae_decode...\n");
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * 4);

    int nan_count = 0;
    double max_abs = 0, alpha_sum = 0;
    int alpha_nonzero = 0;
    // output is (W, H, 4, 1) = (768, 768, 4, 1), channel 0 = alpha
    for (int64_t y = 0; y < RES; y++) {
        for (int64_t x = 0; x < RES; x++) {
            float alpha = result[(y * RES + x) * 4 + 0];
            if (std::isnan(alpha)) nan_count++;
            else {
                max_abs = std::max(max_abs, std::abs((double)alpha));
                alpha_sum += alpha;
                if (alpha > 0.01f) alpha_nonzero++;
            }
        }
    }
    printf("output: %zu elems, nan=%d max_abs=%.4f alpha_mean=%.6f alpha_nonzero=%d/%d\n",
           result.size(), nan_count, max_abs, alpha_sum / (RES * RES), alpha_nonzero, RES * RES);
    printf("%s\n", nan_count == 0 && alpha_nonzero > 0 ? "PASS" : "FAIL");

    ggml_gallocr_free(alloc);
    ggml_free(m.ctx_g);
    return (nan_count > 0 || alpha_nonzero == 0) ? 1 : 0;
}
