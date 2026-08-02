// Attention backend speed benchmark.
// Times the full attention computation (Q/K/V projections, scaled attention,
// output projection) on CPU (Accelerate-BLAS) vs GPU (Metal flash). Each
// backend builds its own graph and buffers so the timed region is pure
// steady-state execution. Prints per-iteration microseconds + GPU/CPU ratio.
//
// Build: cmake --build build --target attn_bench
// Run:   build/attn_bench
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>

static ggml_backend_t gpu_backend() {
    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!d) d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    return d ? ggml_backend_dev_init(d, nullptr) : nullptr;
}

// Build the attention graph and time `iters` steady-state iterations, returning
// per-iteration microseconds. `use_gpu` selects CPU/GPU backend; `use_flash`
// selects the flash graph or the naive matmul graph. Both are independent so we
// can compare flash-on-CPU vs flash-on-GPU fairly.
static double time_attn(int heads, int tq, int tk, int batch, bool use_gpu,
                        bool use_flash, int iters, int warmup, uint64_t seed) {
    const int C = 64 * heads;   // full channel width
    const int hd = 64;          // per-head dim
    const int nq = tq * batch;  // query rows

    const size_t max_nodes = 8192;
    size_t meta = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);

    // weights: Wq,Wk,Wv [C,C]; Wo [C,C]; bo [C]
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    ggml_tensor * bo = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, nq);
    ggml_set_input(x);

    ggml_backend_t backend = use_gpu ? gpu_backend() : ggml_backend_cpu_init();
    if (!backend) return -1;
    ggml_backend_buffer_t wbuf =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    // random weights/input
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> n(0.0f, 0.5f);
    for (ggml_tensor * t : {wq, wk, wv, wo}) {
        std::vector<float> v(ggml_nelements(t));
        for (float & z : v) z = n(rng);
        ggml_backend_tensor_set(t, v.data(), 0, v.size()*4);
    }
    std::vector<float> bv(C); for (float & z : bv) z = n(rng);
    ggml_backend_tensor_set(bo, bv.data(), 0, C*4);
    std::vector<float> xv((size_t)C*nq); for (float & z : xv) z = n(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*4);

    // projections
    ggml_tensor * q = ggml_mul_mat(ctx, wq, x);   // [nq, C]
    ggml_tensor * k = ggml_mul_mat(ctx, wk, x);
    ggml_tensor * v = ggml_mul_mat(ctx, wv, x);

    auto reshape4 = [&](ggml_tensor * t){
        return ggml_reshape_4d(ctx, t, hd, heads, tq, batch);
    };

    ggml_tensor * out = nullptr;
    if (use_flash) {
        ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, reshape4(q), 0,2,1,3));
        ggml_tensor * k4 = ggml_cont(ctx, ggml_permute(ctx, reshape4(k), 0,2,1,3));
        ggml_tensor * v4 = ggml_cont(ctx, ggml_permute(ctx, reshape4(v), 0,2,1,3));
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, q4, k4, v4, nullptr,
                                               1.0f/sqrtf(hd), 0.0f, 0.0f);
        ggml_tensor * kqv = ggml_reshape_3d(ctx, fa, C, tq, batch);
        out = ggml_add(ctx, ggml_mul_mat(ctx, wo, kqv), bo);
    } else {
        // naive path: q = permute-scaled, softmax, v mul
        ggml_tensor * qs = ggml_scale(ctx, q, 1.0f/sqrtf(hd));
        ggml_tensor * qr = ggml_cont(ctx, ggml_permute(ctx, reshape4(qs), 0,2,1,3));
        ggml_tensor * kr = ggml_cont(ctx, ggml_permute(ctx, reshape4(k), 0,2,1,3));
        ggml_tensor * vr = ggml_cont(ctx, ggml_permute(ctx, reshape4(v), 1,2,0,3));
        ggml_tensor * kq = ggml_mul_mat(ctx, kr, qr);
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor * kqv = ggml_mul_mat(ctx, vr, kq);
        kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0,2,1,3));
        ggml_tensor * kqv3 = ggml_reshape_3d(ctx, kqv, C, tq, batch);
        out = ggml_add(ctx, ggml_mul_mat(ctx, wo, kqv3), bo);
    }
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    double per = -1;
    if (ggml_gallocr_alloc_graph(alloc, gf)) {
        for (int i = 0; i < warmup; i++) ggml_backend_graph_compute(backend, gf);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        per = (double) us / (double) iters;
    }
    ggml_gallocr_free(alloc);
    // Free the weight buffer explicitly: on Metal this removes its residency
    // set from the device collection. If we skip this, the static device
    // destructor hits "GGML_ASSERT([rsets->data count] == 0)" at process exit
    // (sig 6 / SIGABRT). Order matters: free buffers before the backend.
    if (wbuf) ggml_backend_buffer_free(wbuf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return per;
}

int main() {
    struct S { int H, Tq, Tk, B; };
    S shapes[] = { {8, 512,512,1}, {8,1280,1280,1}, {10,1600,1600,1},
                   {10,1600,1600,13}, {16,512,512,1} };
    printf("attention speed: naive-CPU(oracle) vs flash-CPU vs flash-GPU, steady-state\n");
    printf("%5s %6s %6s %3s   %12s %12s %12s\n","H","Tq","Tk","B",
           "naiveCPU us","flashCPU us","flashGPU us");
    for (auto &s : shapes) {
        int it = 30, wu = 10;
        double ncpu = time_attn(s.H, s.Tq, s.Tk, s.B, false, false, it, wu, 1);
        double fcpu = time_attn(s.H, s.Tq, s.Tk, s.B, false, true,  it, wu, 1);
        double fgpu = time_attn(s.H, s.Tq, s.Tk, s.B, true,  true,  it, wu, 1);
        if (ncpu <= 0 || fcpu <= 0 || fgpu <= 0) {
            printf("%5d %6d %6d %3d   %s\n", s.H,s.Tq,s.Tk,s.B, " (err)");
            continue;
        }
        printf("%5d %6d %6d %3d   %12.1f %12.1f %12.1f   (flashGPU %4.1fx vs flashCPU; %4.1fx vs naiveCPU)\n",
               s.H,s.Tq,s.Tk,s.B, ncpu, fcpu, fgpu, fcpu/fgpu, ncpu/fgpu);
    }
    return 0;
}
