# 2026-08-14 — two streams per 4090: concurrency measured, 2.1x worse than serial

* Operator: Claude (agent session)
* Objective: establish **throughput per GPU** for fleet sizing — is a 4090 best
  used as one run at a time, or several concurrently?
* Outcome: **concurrency is strongly counterproductive.** Two 1280/30 runs took
  1496.0 s wall against 717.4 s for the same two run serially; aggregate
  efficiency 48%. Optimal batch per 4090 is **1**. One hypothesis refuted, one
  confirmed.

## Apparatus

Identical to
[2026-08-13-4090-f16-release-validation](2026-08-13-4090-f16-release-validation.md)
— RTX 4090 24 GB, driver 591.86, ggml Vulkan (KHR_coopmat, coopmat2
force-disabled), unmodified `v0.1.1` Windows/Vulkan binary, `v0.1.1` f16
weights, `art/concept/anime_with_caption_cc0_0023.jpg`, `--res 1280
--steps 30`.

Differences for this entry:

| Component | Value |
|---|---|
| Concurrency | 2 `see-through.exe` processes, launched 2 s apart |
| Seeds | 42 (stream A) and 43 (stream B) — distinct, to avoid any shared-cache artifact |
| Serial reference | 358.7 s, single stream, same binary and settings (previous entry) |

## Hypotheses

### H1 — Running two streams raises per-GPU throughput. **REFUTED**

*Why predicted:* fleet arithmetic quoted earlier assumed linear scaling at one
run per GPU, and that assumption had never been tested. Two arguments said
concurrency might win: `nvidia-smi` reported ~16.6 GB peak against 24.5 GB
available, which *looks* like it leaves room; and its "100% utilization" only
means at least one kernel is resident, not that the SMs are saturated — so a
second stream might fill occupancy gaps left by a single stream's serial
dependency chain. The counter-argument was that 2 x 16.6 GB exceeds 24 GB, but
peak is instantaneous and the two streams' peaks need not coincide.

*Result:* **refuted, decisively.** Two PSDs in 1496.0 s concurrent vs 717.4 s
serial — **2.09x worse**. Per-GPU throughput falls from 2.79e-3 to 1.34e-3
PSD/s.

The failure mode matters more than the ratio: **it does not OOM, it silently
crawls.** Both processes stayed alive at 23.73 GB / 24.5 GB (97%) with no
allocation failure and no error in either log. WDDM keeps them running by
evicting and re-fetching, converting a memory over-subscription into a ~10x
compute slowdown with no diagnostic. Anything scheduling this workload must
treat "it fits" as unproven by absence of an OOM.

Mechanism is memory, not compute, on three independent signals:

* **Power fell** to 216 W from 430 W single-stream — the card is stalling, not
  working harder.
* **Per-step time collapsed** for stream B: 3.72 -> **38.94 s/step** (10.5x);
  stream A degraded to 14.64 s/step. The two streams were affected unequally.
* **Recovery is immediate.** Stream A finished 266 s before B. B's *head* loop,
  which ran after A exited, clocked **3.72 s/step** — exactly the solo
  baseline — and its marigold at 2.65 s/step, also baseline. The penalty
  applies only while sharing, and vanishes the instant the card is free.

### H2 — The penalty is contention, not a fixed per-process cost. **CONFIRMED**

*Why predicted:* if the slowdown came from process-level overhead (driver
context switching, duplicated model loads, host-side serialisation) it would
persist for a stream's whole lifetime. If it came from memory contention it
would track card occupancy and disappear when the other process exits. These
predict opposite behaviour after the first stream finishes, so the test costs
nothing extra.

*Result:* confirmed by the recovery above — B returned to baseline step time
mid-run, in the same process, without restarting. This also means the measured
2.09x is **optimistic**: for its final 266 s B ran alone at full speed, so a
clean steady-state two-stream measurement would be worse.

## Observations

All *measured* unless labelled otherwise.

### Throughput (measured)

| configuration | wall | PSD/s per 4090 |
|---|---|---|
| 1 stream (serial, x2) | 717.4 s | 2.79e-3 |
| 2 concurrent | 1496.0 s | **1.34e-3** |

Individual `run` spans concurrent: 1230.3 s (A) and 1496.0 s (B). Both produced
valid output — 27 and 28 layers respectively (seeds differ, so layer counts
legitimately differ).

### Stage detail, concurrent (measured)

| span | stream A | stream B |
|---|---|---|
| `clip.body` | 9.1 s | 1.8 s |
| `layerdiff.body` | 594.6 s | 1304.2 s |
| `layerdiff.head` | 538.0 s | 147.3 s (ran mostly alone) |
| `marigold` | 87.9 s | 31.5 s (ran alone) |
| `run` | 1230.3 s | 1496.0 s |

UNet loop step times, concurrent vs the 4.16 / 3.69 s/step solo baseline:
A body 14.64, A head 12.51; B body **38.94**, B head 3.72 (alone).

### Fleet sizing (derived arithmetic, not measured)

At the measured 1-stream figure, 1 PSD/s requires ~359 RTX 4090s
(358.7 GPU-seconds per PSD; 5.88e6 shader-core-seconds at 16,384 cores). This
entry does not measure multi-GPU scaling — it only establishes that the
per-GPU term in that arithmetic should be the **1-stream** figure, which was
previously an assumption.

## Open threads

* **Two streams under Q4 weights.** The Q4 model set is 4.86 GB against f16's
  12.65 GB, so two Q4 streams plausibly fit under 24 GB without eviction. If
  the card has genuine occupancy headroom at 100% reported utilisation, that is
  the configuration where concurrency could still win, and it would roughly
  double PSD/s per GPU. Not attempted here: no Q4 weights on this host, and
  producing them means converting from the upstream torch models.
* Per-GPU throughput is **memory-bound, not compute-bound**, at this
  resolution. Every lever that reduces resident footprint (Q4, a smaller
  decoder, freeing stage weights earlier) buys throughput twice — once
  directly, once by making concurrency viable.
* Single trial per configuration. The direction is far outside run-to-run noise
  (2.09x against a measured 1.4 s spread on the serial baseline), but the exact
  ratio should not be quoted as precise.
