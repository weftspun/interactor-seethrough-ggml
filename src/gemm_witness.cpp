#include "seethrough_capi.h"
#include "ops.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// gemm witness: pure f32×f32 matmul at scale = 2^batch.
// Reference: CPU backend (f32 accumulation throughout).
// Candidate: GPU backend (Metal simdgroup, truncates to half internally).
// Returns interval violation (>1.0 = witness for half-precision overflow).
// ---------------------------------------------------------------------------

extern "C" ST_API double st_witness_check_gemm(uint32_t m, uint32_t n, uint32_t k,
                                                uint32_t scale_bits, uint64_t seed) {
    if (m < 1 || n < 1 || k < 1) return -1.0;

    const double scale = scale_bits < 31 ? (double)(1u << scale_bits) : 1.0;
    const int64_t M = m, N = n, K = k;

    // Build bare ggml graph: C = A @ B  where A[K,N], B[K,M], C[N,M]
    // ggml_mul_mat convention: both input tensors share ne0 = K (inner dim)
    //   weight = [K, N], act = [K, M], result = [N, M]
    auto build_and_run = [&](bool use_cpu) -> std::vector<float> {
        const size_t max_nodes = 4096;
        size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
        ggml_init_params ip = { meta, nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);  // [K, N]
        ggml_tensor * act    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);  // [K, M]
        ggml_set_input(weight);
        ggml_set_input(act);
        // mul_mat_f32 has our Metal-aware half-precision workarounds
        ggml_tensor * r = mul_mat_f32(ctx, weight, act);  // C: [N, M]
        ggml_set_output(r);

        ggml_backend_t backend = use_cpu
            ? ggml_backend_cpu_init()
            : []() -> ggml_backend_t {
                ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                if (!d) d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
                return d ? ggml_backend_dev_init(d, nullptr) : ggml_backend_cpu_init();
              }();

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        std::vector<float> result;
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nrm(0.0f, (float)scale);
        std::vector<float> wv((size_t)K * N), av((size_t)K * M);
        for (float & v : wv) v = nrm(rng);
        for (float & v : av) v = nrm(rng);

        if (ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_backend_tensor_set(weight, wv.data(), 0, wv.size() * 4);
            ggml_backend_tensor_set(act, av.data(), 0, av.size() * 4);
            if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
                result.resize(ggml_nelements(r));
                ggml_backend_tensor_get(r, result.data(), 0, result.size() * 4);
            }
        }
        ggml_gallocr_free(alloc);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return result;
    };

    auto ref = build_and_run(true);   // CPU
    auto cand = build_and_run(false); // GPU (Metal)

    if (ref.empty() || cand.empty()) return -1.0;

    double amp = 0;
    for (float v : ref) amp = std::max(amp, (double)std::fabs(v));
    const double rtol = std::sqrt((double)K) / 1024.0;
    const double atol = 1e-3;
    const double tol = atol + rtol * amp;
    double worst = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        worst = std::max(worst, std::fabs((double)cand[i] - (double)ref[i]) / tol);
    }
    return worst;
}