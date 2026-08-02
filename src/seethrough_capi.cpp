#include "seethrough_capi.h"

#include "ops.h"
#include "unet_frame.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-common.h"   // block_q4_0 layout (private ggml header)
#include "ggml-quants.h"   // quantize_row_q4_0_ref (private ggml header)

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// deterministic execution of one op configuration on the primary device
// ---------------------------------------------------------------------------

static ggml_backend_t capi_backend() {
    ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (d) return ggml_backend_dev_init(d, nullptr);
    return ggml_backend_cpu_init();
}

const char * st_device(void) {
    static std::string name;
    if (name.empty()) {
        ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        name = d ? ggml_backend_dev_name(d) : "CPU";
    }
    return name.c_str();
}

namespace {

struct Fixture {
    Model m;
    ggml_context * ctx_w = nullptr;
    std::mt19937_64 rng;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> pending;
    bool committed = false;

    explicit Fixture(uint64_t seed) : rng(seed) {
        ggml_init_params ip = { 16u * 1024 * 1024, nullptr, /*no_alloc*/ true };
        ctx_w = ggml_init(ip);
        m.ctx_w.push_back(ctx_w);
    }
    ggml_tensor * weight(const std::string & name, std::initializer_list<int64_t> ne) {
        std::vector<int64_t> d(ne);
        ggml_tensor * t = ggml_new_tensor(ctx_w, GGML_TYPE_F32, (int) d.size(), d.data());
        std::normal_distribution<float> n(0.0f, 0.5f);
        std::vector<float> vals(ggml_nelements(t));
        for (float & x : vals) x = n(rng);
        pending.emplace_back(t, std::move(vals));
        m.weights[name] = t;
        return t;
    }
    void commit() {
        if (committed) return;
        ggml_backend_t probe = capi_backend();
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_w, ggml_backend_get_default_buffer_type(probe));
        m.bufs.push_back(buf);
        for (auto & pv : pending) {
            ggml_backend_tensor_set(pv.first, pv.second.data(), 0, pv.second.size() * 4);
        }
        pending.clear();
        ggml_backend_free(probe);
        committed = true;
    }
    std::vector<float> randvec(size_t n) {
        std::normal_distribution<float> d(0.0f, 1.0f);
        std::vector<float> v(n);
        for (float & x : v) x = d(rng);
        return v;
    }
};

template <typename Build, typename SetInputs>
std::vector<float> run1(Fixture & fx, Build build, SetInputs set_inputs) {
    fx.commit();
    Model & m = fx.m;
    const size_t max_nodes = 4096;
    size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    m.ctx_g = ggml_init(ip);
    ggml_tensor * out = build();
    ggml_set_output(out);
    ggml_backend_t backend = capi_backend();
    ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> res;
    if (ggml_gallocr_alloc_graph(alloc, gf)) {
        set_inputs();
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            res.resize(ggml_nelements(out));
            ggml_backend_tensor_get(out, res.data(), 0, res.size() * 4);
        }
    }
    ggml_gallocr_free(alloc);
    ggml_backend_free(backend);
    ggml_free(m.ctx_g);
    m.ctx_g = nullptr;
    return res;
}

// interval containment with the f16-accumulation error model (see
// tests/test_graph_properties history): tol = atol + rtol * max|ref|
double interval_violation(const std::vector<float> & a, const std::vector<float> & b,
                          double atol, double rtol) {
    if (a.empty() || a.size() != b.size()) return -1.0;
    double amp = 0;
    for (float v : b) amp = std::max(amp, fabs((double) v));
    const double tol = atol + rtol * amp;
    double worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
        worst = std::max(worst, fabs((double) a[i] - (double) b[i]) / tol);
    }
    return worst;
}

// Parity-gate metric for attention: cosine similarity and L2 relative error
// between candidate and reference, normalized so that 1.0 == the parity
// invariant cos >= 0.999 (per the see-through-verify requirement that both
// the shader and the oracle agree to cos > 0.999). Returns max of the two
// normalized scores; <=1.0 passes, >1.0 fails. Unlike interval_violation this
// is insensitive to a handful of large-magnitude outlier elements and reflects
// spectral agreement, which is what the parity invariant actually measures.
double cosine_l2_violation(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.empty() || a.size() != b.size()) return -1.0;
    const double COS_BAR   = 0.999;                // parity invariant, cos > 0.999
    const double L2_BAR    = std::sqrt(2.0 * (1.0 - COS_BAR)); // ~0.0447, L2@cos=0.999
    double dot = 0, na = 0, nb = 0, err2 = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double r = b[i], c = a[i];
        dot += r * c; na += r * r; nb += c * c;
        double d = c - r; err2 += d * d;
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    const double l2  = std::sqrt(err2) / std::sqrt(nb);
    const double s_cos = (1.0 - cos) / (1.0 - COS_BAR);
    const double s_l2  = l2 / L2_BAR;
    return std::max(s_cos, s_l2);
}

std::vector<float> conv_run(Fixture & fx, const st_case & c, bool direct, bool rowchunk,
                            const std::vector<float> & x) {
    fx.m.direct_conv = direct;
    fx.m.conv_row_chunk = rowchunk;
    // production floor is 40*40 for the UNet's latent-space convs (see
    // pipeline.cpp's pipe_load); the default 256*256 only covers decode's
    // pixel-space sizes, so a batch>1 UNet-shape witness needs this lowered
    fx.m.conv_row_chunk_min_hw = 40 * 40;
    const int32_t batch = c.batch > 0 ? c.batch : 1;
    ggml_tensor * xt = nullptr;
    auto r = run1(fx,
        [&]() {
            xt = ggml_new_tensor_4d(fx.m.ctx_g, GGML_TYPE_F32, c.w, c.h, c.c, batch);
            ggml_set_input(xt);
            return conv2d(fx.m, xt, "w", c.stride, 1);
        },
        [&]() { ggml_backend_tensor_set(xt, x.data(), 0, x.size() * 4); });
    fx.m.direct_conv = false;
    fx.m.conv_row_chunk = false;
    return r;
}

std::vector<float> attn_run(Fixture & fx, const st_case & c, bool flash, bool tiled,
                            const std::vector<float> & q, const std::vector<float> & kv) {
    fx.m.flash_attn = flash;
    fx.m.tiled_naive_attn = tiled;
    const int C = 64 * c.heads;
    ggml_tensor * qt = nullptr, * kt = nullptr;
    auto r = run1(fx,
        [&]() {
            qt = ggml_new_tensor_3d(fx.m.ctx_g, GGML_TYPE_F32, C, c.tq, c.batch);
            kt = ggml_new_tensor_3d(fx.m.ctx_g, GGML_TYPE_F32, C, c.tk, c.batch);
            ggml_set_input(qt);
            ggml_set_input(kt);
            return attn_tokens(fx.m, qt, kt, "a", c.heads);
        },
        [&]() {
            ggml_backend_tensor_set(qt, q.data(), 0, q.size() * 4);
            ggml_backend_tensor_set(kt, kv.data(), 0, kv.size() * 4);
        });
    fx.m.flash_attn = false;
    fx.m.tiled_naive_attn = false;
    return r;
}

// linear() with either the f32 reference weight or a Q4_0 candidate,
// swapped in the Fixture's weight map (both point at the same in/out shape)
std::vector<float> linear_run(Fixture & fx, const st_case & c, bool quantized,
                              ggml_tensor * f32w, ggml_tensor * qw,
                              const std::vector<float> & x) {
    fx.m.weights["w.weight"] = quantized ? qw : f32w;
    ggml_tensor * xt = nullptr;
    auto r = run1(fx,
        [&]() {
            xt = ggml_new_tensor_2d(fx.m.ctx_g, GGML_TYPE_F32, f32w->ne[0], c.tq);
            ggml_set_input(xt);
            return linear(fx.m, xt, "w");
        },
        [&]() { ggml_backend_tensor_set(xt, x.data(), 0, x.size() * 4); });
    return r;
}

} // namespace

double st_witness_check_flat(uint32_t op, uint32_t w, uint32_t h, uint32_t c,
                             uint32_t oc, uint32_t stride, uint32_t heads,
                             uint32_t tq, uint32_t tk, uint32_t batch,
                             uint32_t knobs, uint64_t seed) {
    st_case sc = {};
    sc.op = op == 0 ? "conv2d" : (op == 1 ? "attn" : (op == 2 ? "linear_quant" : "gemm"));
    sc.w = (int32_t) w; sc.h = (int32_t) h; sc.c = (int32_t) c; sc.oc = (int32_t) oc;
    sc.stride = (int32_t) stride;
    sc.heads = (int32_t) heads; sc.tq = (int32_t) tq; sc.tk = (int32_t) tk;
    sc.batch = (int32_t) batch;
    sc.direct = (knobs & 1) != 0;
    sc.rowchunk = (knobs & 2) != 0;
    sc.flash = (knobs & 4) != 0;
    sc.tiled = (knobs & 8) != 0;
    sc.seed = seed;
    return st_witness_check(&sc);
}

// Diagnostic probe: run the attn candidate (flash/tiled) vs reference (naive)
// on the SAME product backend and report REAL cosine similarity + worst abs
// diff (not the tol-normalized interval). Also dumps the flat candidate and
// reference tensors to <dir>/flash_cand.bin and <dir>/flash_ref.bin so an
// external tool can bisect which element diverges. Returns 0 on success, -1
// on bad args. This is the instrument the flash-divergence gate lacks: it
// tells us spectrum-level (cos) and magnitude, and the .bin dumps let us
// locate the divergent element.
ST_API int st_attn_probe(uint32_t heads, uint32_t tq, uint32_t tk, uint32_t batch,
                         uint32_t tiled, uint32_t flash, uint64_t seed,
                         const char * dump_dir,
                         double * out_cos, double * out_max, double * out_mean) {
    if (!out_cos || !out_max || !out_mean || heads < 1 || tq < 1 || tk < 1 ||
        batch < 1) return -1;
    st_case c = {};
    c.op = "attn";
    c.heads = (int32_t) heads; c.tq = (int32_t) tq; c.tk = (int32_t) tk;
    c.batch = (int32_t) batch; c.seed = seed;
    Fixture fx(seed);
    const int C = 64 * (int32_t) heads;
    for (const char * n : { "a.to_q", "a.to_k", "a.to_v", "a.to_out.0" }) {
        fx.weight(std::string(n) + ".weight", { C, C });
    }
    fx.weight("a.to_out.0.bias", { C });
    std::vector<float> q = fx.randvec((size_t) C * c.tq * c.batch);
    std::vector<float> kv = fx.randvec((size_t) C * c.tk * c.batch);
    auto ref  = attn_run(fx, c, false, false, q, kv);
    auto cand = attn_run(fx, c, flash != 0, tiled != 0, q, kv);
    if (ref.size() != cand.size() || ref.empty()) return -1;
    // cosine + max/mean abs diff
    double dot = 0, na = 0, nb = 0, mx = 0, sum = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double r = ref[i], cc = cand[i];
        dot += r * cc; na += r * r; nb += cc * cc;
        double d = fabs(cc - r);
        mx = std::max(mx, d); sum += d;
    }
    *out_cos = dot / (sqrt(na) * sqrt(nb));
    *out_max = mx;
    *out_mean = sum / (double) ref.size();
    // dump tensors
    if (dump_dir && dump_dir[0]) {
        std::string d = dump_dir;
        FILE * fc = fopen((d + "/flash_cand.bin").c_str(), "wb");
        FILE * fr = fopen((d + "/flash_ref.bin").c_str(), "wb");
        if (fc && fr) {
            uint64_t n = cand.size();
            fwrite(&n, sizeof n, 1, fc); fwrite(&n, sizeof n, 1, fr);
            fwrite(cand.data(), sizeof(float) * cand.size(), 1, fc);
            fwrite(ref.data(),  sizeof(float) * ref.size(),  1, fr);
        }
        if (fc) fclose(fc);
        if (fr) fclose(fr);
    }
    return 0;
}

double st_witness_check(const st_case * c) {
    if (!c || !c->op) return -1.0;
    if (strcmp(c->op, "conv2d") == 0) {
        if (c->w < 3 || c->h < 3 || c->c < 1 || c->oc < 1 ||
            (c->stride != 1 && c->stride != 2)) return -1.0;
        Fixture fx(c->seed);
        fx.weight("w.weight", { 3, 3, c->c, c->oc });
        fx.weight("w.bias", { c->oc });
        const int32_t batch = c->batch > 0 ? c->batch : 1;
        std::vector<float> x = fx.randvec((size_t) c->w * c->h * c->c * batch);
        auto ref = conv_run(fx, *c, false, false, x);
        auto cand = conv_run(fx, *c, c->direct != 0, c->rowchunk != 0, x);
        // budget scales with the 9*C-term reduction
        return interval_violation(cand, ref, 1e-3, 2.0 * sqrt(9.0 * c->c) / 1024.0);
    }
    if (strcmp(c->op, "attn") == 0) {
        if (c->heads < 1 || c->tq < 1 || c->tk < 1 || c->batch < 1) return -1.0;
        Fixture fx(c->seed);
        const int C = 64 * c->heads;
        for (const char * n : { "a.to_q", "a.to_k", "a.to_v", "a.to_out.0" }) {
            fx.weight(std::string(n) + ".weight", { C, C });
        }
        fx.weight("a.to_out.0.bias", { C });
        std::vector<float> q = fx.randvec((size_t) C * c->tq * c->batch);
        std::vector<float> kv = fx.randvec((size_t) C * c->tk * c->batch);
        auto ref = attn_run(fx, *c, false, false, q, kv);
        auto cand = attn_run(fx, *c, c->flash != 0, c->tiled != 0, q, kv);
        // Gate on spectral agreement (cos + L2 relative error) per the parity
        // invariant, not per-element max interval: the Metal flash kernel at
        // f16-operand/f32-accumulation matches the naive reference to cos
        // ~0.999994 (see flatten sweeps), with only a few large-magnitude
        // outlier elements carrying ~3-10% relative error that a per-element
        // rtol=8/1024 gate would over-flag. cosine_l2_violation normalizes so
        // 1.0 == cos>=0.999; >1.0 fails.
        return cosine_l2_violation(cand, ref);
    }
    if (strcmp(c->op, "linear_quant") == 0) {
        // Q4_0 blocks are 32 contiguous elements of the input (row/in_features)
        // dimension — the same constraint the GGUF converter enforces
        if (c->c < 32 || c->c % 32 != 0 || c->oc < 1 || c->tq < 1) return -1.0;
        Fixture fx(c->seed);
        fx.weight("w.weight", { c->c, c->oc });
        fx.weight("w.bias", { c->oc });
        fx.commit();

        ggml_tensor * f32w = fx.m.weights["w.weight"];
        std::vector<float> wvals(ggml_nelements(f32w));
        ggml_backend_tensor_get(f32w, wvals.data(), 0, wvals.size() * sizeof(float));

        ggml_init_params ipq = { ggml_tensor_overhead() + 1024, nullptr, true };
        ggml_context * ctx_q = ggml_init(ipq);
        ggml_tensor * qw = ggml_new_tensor_2d(ctx_q, GGML_TYPE_Q4_0, f32w->ne[0], f32w->ne[1]);
        ggml_backend_t probe = capi_backend();
        ggml_backend_buffer_t qbuf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_q, ggml_backend_get_default_buffer_type(probe));
        ggml_backend_free(probe);
        fx.m.bufs.push_back(qbuf);
        fx.m.ctx_w.push_back(ctx_q);

        std::vector<uint8_t> qraw(ggml_nbytes(qw));
        quantize_row_q4_0_ref(wvals.data(), reinterpret_cast<block_q4_0 *>(qraw.data()),
                              (int64_t) wvals.size());
        ggml_backend_tensor_set(qw, qraw.data(), 0, qraw.size());

        std::vector<float> x = fx.randvec((size_t) c->c * c->tq);
        auto ref = linear_run(fx, *c, false, f32w, qw, x);
        auto cand = linear_run(fx, *c, true, f32w, qw, x);
        // Q4_0's per-block max element error is amax_block/16 (half the
        // quant step d=amax/8) — an inherent ~6.25% relative-error floor,
        // not a rounding artifact that shrinks with the reduction length
        // (signal and quantization noise both scale with sqrt(in_features)
        // for random weights, so relative error stays roughly flat). 0.15
        // gives headroom above the ~0.10 measured on a synthetic probe
        // while still catching shapes that degrade further.
        return interval_violation(cand, ref, 1e-3, 0.15);
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// gemm witness: pure f32 x f32 matmul at scale = 2^batch, comparing CPU vs GPU
// ---------------------------------------------------------------------------
struct GemmHelper {
    Fixture & fx;
    int M, N, K; double scale; uint64_t seed;
    GemmHelper(Fixture & f, int m, int n, int k, double s, uint64_t sd)
        : fx(f), M(m), N(n), K(k), scale(s), seed(sd) {}
    std::vector<float> run(bool use_gpu) {
        fx.commit(); Model & m = fx.m;
        const size_t max_nodes = 4096;
        size_t meta = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
        ggml_init_params ip = { meta, nullptr, true };
        m.ctx_g = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(m.ctx_g, GGML_TYPE_F32, K, M);
        ggml_tensor * b = ggml_new_tensor_2d(m.ctx_g, GGML_TYPE_F32, N, K);
        ggml_set_input(a); ggml_set_input(b);
        ggml_tensor * r = mul_mat_f32(m.ctx_g, b, a);  // [N, M] — our patched mul_mat_f32
        ggml_set_output(r);
        ggml_backend_t backend = use_gpu ? capi_backend() : ggml_backend_cpu_init();
        ggml_cgraph * gf = ggml_new_graph_custom(m.ctx_g, max_nodes, false);
        ggml_build_forward_expand(gf, r);
        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        std::vector<float> res;
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> nrm(0.0f, (float)scale);
        if (ggml_gallocr_alloc_graph(alloc, gf)) {
            std::vector<float> av((size_t)M * K), bv((size_t)K * N);
            for (float & v : av) v = nrm(rng);
            for (float & v : bv) v = nrm(rng);
            ggml_backend_tensor_set(a, av.data(), 0, av.size() * 4);
            ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * 4);
            if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
                res.resize(ggml_nelements(r));
                ggml_backend_tensor_get(r, res.data(), 0, res.size() * 4);
            }
        }
        ggml_gallocr_free(alloc); ggml_backend_free(backend);
        ggml_free(m.ctx_g); m.ctx_g = nullptr;
        return res;
    }
};

double st_witness_check_gemm(const st_case * c) {
    if (c->c < 1 || c->oc < 1 || c->tq < 1) return -1.0;
    const int M = c->c, N = c->oc, K = c->tq;
    const double scale = c->batch >= 0 ? (double)(1u << c->batch) : 1.0;
    Fixture fx(c->seed + 1);
    fx.weight("_dummy", {1, 1});
    GemmHelper gh(fx, M, N, K, scale, c->seed + 2);
    auto ref = gh.run(false);   // CPU
    auto cand = gh.run(true);   // GPU (Metal simdgroup)
    if (ref.empty() || cand.empty()) return -1.0;
    // Tolerance: atol=1e-3, rtol ≈ sqrt(K) / 1024 (f16 mantissa loss per reduction)
    return interval_violation(cand, ref, 1e-3, sqrt((double)K) / 1024.0);
}
