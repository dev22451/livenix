# Livenix — Phase 1 Plan (v2, evidence-based)

> Project = passive RGB face liveness SDK. Brand: **Livenix**.
> Reference baseline = Minivision Silent-Face-Anti-Spoofing (2020, Apache 2.0).
> **Phase 1 goal:** match/beat reference baseline on print + screen-replay attacks, with our own clean codebase and weights, mobile-first, server path designed in (built Phase 1.5, no rewrite).
> **Plan v2 corrections vs v1:** dataset license posture, evidence-based KPIs, dropped multi-scale crop fusion, added deepfake training class + OTA infra, Phase 1.5 uses Google ML Kit for active liveness.

---

## Phase 1 — Scope (locked)

| Item | Decision | Notes |
|---|---|---|
| Attack scope | **Print + Screen Replay** only | iBeta PAD L1 territory |
| Match-list defended | Photo, video call, camera call, recorded clip, paper cutout, **deepfake-on-screen, AIGC-on-screen** | All collapse to print + replay physics |
| NOT in scope Phase 1 | Silicone, 3D rigid masks, virtual-camera injection | Phase 1.5 covers injection; silicone Phase 2 |
| Topology Phase 1 | **Mobile-first, on-device only** | Server-ready API stubs included |
| Topology Phase 1.5 | **Hybrid**: on-device + server authoritative + **active liveness via Google ML Kit** | Drop-in upgrade, no SDK rewrite |
| Platforms Phase 1 | Web (1st), Android, iOS | Huawei deferred to Phase 2 |
| Input | Single RGB frame, 128×128 | No active liveness in Phase 1, no depth, no IR |
| Classes | **3-class** output: `real` / `print_spoof` / `replay_spoof` | Matches reference SDK |
| Output | `{ isLive, spoofType, confidence, source, latencyMs, modelVersion }` | `source` reserved for server-referred path |
| Budget Phase 1 | ~$26 GPU (RunPod 4090 Secure for reliability) | |
| Budget Phase 1.5 | +$5 GPU + dev time | Server model + active liveness + anti-injection |
| Timeline | 4–6 weeks Phase 1; +1–2 weeks Phase 1.5 | |
| License posture | Clean reimplementation, our own IP | No file copied from reference repo |

---

## Phase 1 — Output Coverage Hierarchy (P0/P1/P1.5/P2)

| Priority | Deliverable | Acceptance KPI (evidence-backed) |
|---|---|---|
| **P0** | Trained PyTorch model (v0.1 research prototype) | **TPR ≥ 96% @ FPR=1e-4** intra-dataset (CelebA-Spoof) |
| **P0** | ONNX export | Numerical parity vs PyTorch ≤ 1e-3 on 1000 fixed inputs |
| **P0** | INT8 quantized model | Accuracy drop vs FP32 < 1% (QAT fallback if PTQ insufficient) |
| **P0** | Cross-dataset eval (LODO) | **BPCER ≤ 15% @ APCER = 5%** |
| **P0** | OULU-NPU eval (Protocol 1, 4) — **internal only, academic EULA** | **P4 ACER ≤ 8%** (realistic for 1.5–4M model at 128×128) |
| **P0** | Webcam desktop demo (Python) | Real-time, bbox + label |
| **P0** | Per-attack confusion matrix + ROC at FPR=1e-3, 1e-4 | Documented, reproducible |
| **P0** | **Model OTA infrastructure** (signed bundles, version pinning, kill-switch) | Architectural — must exist in Phase 1 SDK |
| **P0** | **Versioned preprocessing pipeline** bundled with model artifact | Prevents the "silent regression" failure mode |
| **P0** | Threat model + claims policy doc | Audit-defensible |
| **P1** | TFLite export (Android) | Parity vs PyTorch ≤ 1e-2 |
| **P1** | CoreML export (iOS) | Parity vs PyTorch ≤ 1e-2 |
| **P1** | ONNX Runtime Web demo | Loads <1.7s, runs <100ms/frame on M1 (WebGPU) |
| **P1** | Fairness eval (skin tone × age × gender) | Max-cell BPCER ratio ≤ 1.5× |
| **P1** | Adversarial robustness (FGSM, PGD-20) | Documented robust BPCER @ APCER=1% |
| **P1.5** | MobileNetV4-Conv-Large server model | TPR ≥ 98% @ FPR=1e-4; private server-side weights |
| **P1.5** | FastAPI server inference + telemetry + drift monitoring | Authoritative scoring, rate-limited |
| **P1.5** | Mobile→server referral logic | "Refer when confidence ∈ [0.35, 0.65]" |
| **P1.5** | **Active liveness via Google ML Kit Face Detection** | Blink + head-pose challenge, server-issued nonce |
| **P1.5** | **Anti-injection** (Play Integrity, AppAttest, server frame integrity) | Defeats virtual-camera attacks (OBS, DeepFaceLive) |
| **P2** | NCNN export (Huawei) | Deferred |
| **P2** | Native SDK wrappers (Kotlin, Swift, TS) | Phase 2 |
| **P2** | Silicone-mask claim (after L2-grade data + bigger model) | Phase 2 with paid dataset license |
| **P2** | Multi-frame temporal model | Phase 2 |

---

## Reference baseline + Livenix targets (evidence-based)

| Metric | Reference open SDK (2020) | Livenix Phase 1 (mobile) | Livenix Phase 1.5 (server) |
|---|---|---|---|
| TPR @ FPR=1e-4 intra-dataset | not reported at this op-point | **≥ 96%** | **≥ 98%** |
| Cross-dataset LODO BPCER @ APCER=5% | not reported | **≤ 15%** | **≤ 10%** |
| OULU-NPU P4 ACER | not reported | **≤ 8%** | **≤ 5%** |
| TPR @ FPR=1e-5 (their reported) | 97.8% (their internal test) | Not our op-point (see note) | Not our op-point |
| Model size FP32 | ~3.4 MB | ≤ 18 MB (3.8M params) | ~50 MB (12M params, server) |
| Model size INT8 | not released | ≤ 5 MB | n/a (server runs FP16) |
| Mobile latency | ~20 ms (Kirin 990) | ≤ 30 ms mid-range Android | n/a |
| Server latency | n/a | n/a | ≤ 50 ms on T4/A10 |
| Architecture | MiniFASNetV1/V2 (closed) | MobileNetV4-Conv-Small (timm, Apache) | MobileNetV4-Conv-Large |
| Eval benchmarks | none reported | OULU-NPU + cross-dataset LODO | same |
| Fairness | not reported | Reported per demographic cell | same |

**Note on TPR@FPR=1e-5:** the reference SDK reports this on their internal test set (~57k bona fide samples), where the estimator has wide confidence intervals at 1e-5. We use **TPR@FPR=1e-4** as our headline op-point because it has stable published comparables in academic literature.

---

## Architecture (locked, v2)

### Mobile model (Phase 1)
| Component | Choice | Reason |
|---|---|---|
| Backbone | `timm.mobilenetv4_conv_small.e2400_r224_in1k` (3.8M params, Apache 2.0) | 2024 SOTA mobile, ONNX-clean, INT8-friendly |
| Stem | CDC refactored = Conv − θ·mean-Conv (fused), θ=0.7 | Catches print micro-textures; converts cleanly cross-runtime |
| Aux 1 (train-only) | FFT magnitude branch | Catches screen-replay moiré; stripped at export |
| Aux 2 (train-only) | SigLIP-So400m feature distillation (Apache 2.0) | Generalization to unseen variants; stripped at export |
| Aux 3 (train-only) | Patch contrastive (NT-Xent) | +3–5% cross-domain; stripped at export |
| Main head | 3-class classifier + cosine margin (m=0.3) | Tightens decision boundary |
| Loss | Focal (γ=2) + asymmetric + aux weighted (0.5·SigLIP + 0.2·FFT + 0.3·patch) | Pushes FPR=1e-4 op-point |
| Variants | Mobile (1.0×), Web (0.5×) | Same training pipeline, two configs |

### Server model (Phase 1.5)
| Component | Choice | Delta from mobile |
|---|---|---|
| Backbone | MobileNetV4-Conv-Large (~12M params) | Single config swap; same training code |
| Quantization | None (FP16 on GPU, FP32 on CPU) | Private weights, no shipping concern |
| Everything else | Same as mobile | Maximum code reuse |

### Dropped from v1 plan (after benchmark review)
- ❌ Classical multi-scale crop fusion (Minivision-style 2.7× + 4× crops) — **2019-era technique, superseded** by ViT-patch features and internal CDC layers in 2024-2026 SOTA. Our CDC stem + augmentation does the work.
- ❌ TPR@FPR=1e-5 as headline op-point — replaced with TPR@FPR=1e-4 (industry standard with published comparables).

### Banned ops in inference graph
FFT, dynamic shapes, custom autograd, `grid_sample`, `GroupNorm`, per-channel `PReLU`, custom `Hardswish` expressions. Verified at QA Gate 1 **before training**.

---

## Datasets — research prototype path (locked, v2)

**Legal posture:** Phase 1 weights are labeled **"v0.1 research prototype, not for commercial deployment."** Commercial weights come in Phase 2 after dataset licensing, paid data purchase, or in-house collection ramp.

### Training mix — research prototype phase (Phase 1)
| Dataset | Mix % | Purpose | License | Use |
|---|---|---|---|---|
| CelebA-Spoof | 50% | Volume, print, replay | Research-only | Prototype training, do not redistribute weights commercially |
| WMCA / CASIA-CeFA | 25% | Cross-ethnicity, robustness | Research/Idiap EULA | Same |
| HiFiMask | 10% | Silent silicone exposure (not claimed) | Research-only | Same |
| **Deepfake-on-screen samples** | 10% | SiW-Mv2 + recorded DeepFaceLive playback | Research-only | Same |
| In-house collected | 5% | Deployment realism, legally clean baseline | OURS | Ramps over Phase 1 weeks 2-6 |

### Eval-only (no training)
| Dataset | Purpose | License posture |
|---|---|---|
| OULU-NPU | Public credibility benchmark, 4 cross-domain protocols | Internal benchmarking only under academic EULA; do not publish numbers in commercial marketing |
| WFAS (Wild Face Anti-Spoofing) | Cross-domain robustness, 1.38M images | Same |

### Phase 2 commercial path (when revenue justifies)
| Source | Cost | Solves |
|---|---|---|
| Unidata Anti-Spoofing Real Videos (98K samples) | ~$500-2000 | Commercial-licensed PAD data |
| AxonData face anti-spoofing | varies | Same |
| CASIA-SURF / HiFiMask paid commercial license via SurfingTech | ~$3-10k | Re-enables CeFA + HiFiMask commercially |
| In-house collection at scale (200-500 subjects) | dev time only | Bulletproof legal position |

**Splits**: identity-disjoint enforced via FaceNet cosine sim < 0.6 cross-set. Verified at QA Gate 12.

---

## QA Gates (must pass in order)

| # | Gate | Pass criterion |
|---|---|---|
| 1 | Export gate (untrained) | ONNX + TFLite + CoreML export succeed before any training |
| 2 | Data sanity | 200 random samples per class manually reviewed |
| 3 | Overfit | 100 images → ≥99% acc in 100 epochs |
| 4 | Baseline | 1 epoch full data → loss decreases monotonically |
| 5 | Convergence | Tensorboard stable, per-attack confusion matrix reviewed |
| 6 | Cross-dataset (LODO) | BPCER ≤ 15% @ APCER = 5% |
| 7 | OULU-NPU | P1 and P4 ACER reported; P4 ≤ 8% target |
| 8 | Fairness | Max-cell BPCER ratio ≤ 1.5× across demographic cells |
| 9 | Adversarial | Robust BPCER documented under PGD-20 ε=8/255 |
| 10 | Quantization | INT8 vs FP32 accuracy drop < 1% (QAT fallback ready) |
| 11 | Numerical parity | PyTorch vs ONNX vs TFLite vs CoreML ≤ 1e-2 on 1000 inputs |
| 12 | Contamination audit | No identity overlap between splits |
| 13 | End-to-end golden | Full SDK pipeline matches training pipeline on 100 reference images |
| 14 | **Preprocessing version-pin** | Preprocessing pipeline version bundled with model artifact; regression tests pass on device-captured frames |

---

## Deployment topology

```
Phase 1 (ship first):
┌─────────────────┐         ┌──────────────────────┐
│ Client          │  decide │ On-device model      │
│ (mobile/web)    ├────────►│ INT8 ~3.5 MB         │
│                 │  locally│ + versioned preproc  │
│                 │         │ + OTA-capable        │
└─────────────────┘         └──────────────────────┘

Phase 1.5 (drop-in upgrade, no SDK rewrite):
┌─────────────────┐         ┌──────────────────────┐
│ Client          │  fast   │ On-device model      │
│                 ├────────►│ + Google ML Kit Face │
│                 │         │   Detection (blink,  │
│                 │  refer  │   head pose)         │
│                 │  if     └──────────┬───────────┘
│                 │  uncertain         │
└─────────────────┘                    ▼
                            ┌──────────────────────┐
                            │ Livenix API (FastAPI)│
                            │ - Conv-Large server  │
                            │ - Active liveness    │
                            │   challenge nonces   │
                            │ - Anti-injection     │
                            │   (Play Integrity,   │
                            │    AppAttest)        │
                            │ - Telemetry + drift  │
                            └──────────────────────┘
```

### Active liveness in Phase 1.5 — Google ML Kit
- Android: **Google ML Kit Face Detection** API — gives eye-open probability, smile probability, head Euler angles (yaw/pitch/roll)
- iOS: **Apple Vision framework** (`VNDetectFaceLandmarksRequest`) — same primitives
- Web: **MediaPipe Face Landmarker** — same primitives via WASM
- Server issues random challenge nonce: e.g. `"blink twice within 3s"` or `"turn head left then center"`
- SDK verifies eye state change / head pose change within window
- Server cross-verifies action result + nonce match → authoritative `isLive`

### API contract baked into Phase 1 SDK (server-ready)
```ts
type LivenessResult = {
  isLive: boolean
  spoofType: "none" | "print" | "replay"
  confidence: number
  source: "on-device" | "server-referred"
  latencyMs: number
  modelVersion: string         // OTA-managed
  preprocessingVersion: string // bundled with model artifact
}

// Phase 1.5 adds, doesn't replace:
type LivenessChallenge = {
  nonce: string
  challenge: "blink" | "head_turn_left" | "head_turn_right" | "look_up"
  windowMs: number
}
```

---

## Production capabilities — what we add vs defer

| Capability | Phase 1 | Phase 1.5 | Phase 2 |
|---|---|---|---|
| Passive PAD (print + screen replay) | ✅ Full | ✅ + server boost | ✅ + temporal |
| 3-class output (per-attack type) | ✅ | ✅ | ✅ |
| Deepfake-on-screen | ✅ (as replay class) | ✅ | ✅ |
| Model OTA / kill-switch | ✅ Infrastructure | ✅ + active rollouts | ✅ |
| Versioned preprocessing | ✅ | ✅ | ✅ |
| Active liveness (Google ML Kit) | ❌ | ✅ Blink + head pose | ✅ + randomized prompts |
| Anti-injection (Play Integrity, AppAttest) | ❌ | ✅ | ✅ + frame integrity chain |
| Server authoritative scoring | ❌ | ✅ | ✅ |
| Drift monitoring | ❌ | ✅ | ✅ |
| Silicone mask | ❌ (silent train only) | ❌ | ✅ with commercial license |
| Multi-frame temporal | ❌ | ❌ | ✅ |
| Huawei NCNN target | ❌ | ❌ | ✅ |

---

## Threat model + claims policy

**Phase 1 defends (CLAIM allowed):**
- Paper print attacks (matte/glossy)
- Phone/tablet screen replay (recorded video, live video call, AI-generated face shown on screen)
- Paper cutouts
- Deepfake video shown on a screen (caught as replay class)

**Phase 1 does NOT defend (DO NOT CLAIM):**
- Silicone/3D rigid masks
- Deepfake puppeting via virtual camera injection (Phase 1.5 covers)
- Live face on screen via virtual camera bypass
- Injection attacks against camera pipeline
- Partial occlusion attacks

**Phase 1.5 adds:**
- Virtual-camera injection defense (via Play Integrity + AppAttest + server nonces)
- Pre-recorded video replay (via active challenge — pre-recorded can't respond to random prompts)
- Static 3D rigid mask (via active challenge — mask can't blink/articulate)

See [CLAIMS.md](CLAIMS.md) for the full allowed/disallowed list and audit-defense script.

---

## Budget & timeline (v2)

| Week | Milestone | GPU |
|---|---|---|
| 1 | Scaffold, architecture, data pipeline, **export gate (untrained)**, OTA scaffolding | $0 |
| 2 | First full training + convergence gate | ~$8 (RunPod 4090 Secure, 30-min checkpoints) |
| 3 | Tuning iterations + aux head ablations + deepfake samples added | ~$10 |
| 4 | Cross-dataset + OULU-NPU + fairness + adversarial gates | ~$5 |
| 5 | Quantization (PTQ + QAT fallback), exports, parity tests | ~$3 |
| 6 | Webcam + web demos, threat model doc, Phase 1 handoff | $0 |
| **Phase 1 total** | | **~$26** |
| 7 | Conv-Large server model + FastAPI + Google ML Kit active liveness | ~$5 |
| 8 | Anti-injection (Play Integrity, AppAttest) + drift monitoring | $0 |
| **Phase 1.5 total** | | **+$5** |

In-house data collection runs **in parallel** weeks 2-6 (no GPU cost, your time).

---

## IP hygiene (non-negotiable)

- No file from `/Users/hamdev/Silent-Face-Anti-Spoofing` is copied/referenced/imported
- No pretrained `.pth` from reference repo is loaded
- Backbone from `timm` (upstream PyTorch ecosystem, Apache 2.0)
- Aux teacher (SigLIP-So400m): Apache 2.0, commercial distillation allowed
- Git history starts at our first commit in `/Users/hamdev/livenix`
- Concepts reimplemented from published papers (CDCN, MobileNetV4, CelebA-Spoof, PatchNet)
- README cites published papers; does not cite Minivision repo
- Brand name: **Livenix**. Package name: `livenix`
- v0.1 weights labeled "research prototype, not for commercial deployment" until commercial dataset path lands

---

## Decisions log (locked, v2)

| Date | Decision |
|---|---|
| 2026-05-17 | Brand = Livenix |
| 2026-05-17 | Phase 1 = print + screen replay (incl. deepfake-on-screen, AIGC-on-screen); no silicone/injection claims |
| 2026-05-17 | Mobile-first; hybrid server path designed-in, built Phase 1.5 |
| 2026-05-17 | Server model = MobileNetV4-Conv-Large (single config swap) |
| 2026-05-17 | OULU-NPU = internal eval only, not commercial marketing numbers |
| 2026-05-17 | Backbone = MobileNetV4-Conv-Small (3.8M params); input 128×128 |
| 2026-05-17 | Aux heads (train-only): FFT + SigLIP distillation + patch contrastive |
| 2026-05-17 | Quantization = INT8 PTQ first, QAT fallback if >1% drop |
| 2026-05-17 | Python tooling = uv; git initialized from first commit |
| 2026-05-17 | 3-class output (real/print_spoof/replay_spoof) |
| 2026-05-17 | Multi-scale crop fusion DROPPED (2019-era, superseded) |
| 2026-05-17 | HiFiMask in training mix for silent silicone resilience (not claimed) |
| 2026-05-17 | **KPIs rewritten to TPR@FPR=1e-4 (96-98% Phase 1; 98-99% Phase 1.5)** — evidence-based |
| 2026-05-17 | **Dataset path = research prototype + in-house collection; commercial license Phase 2** |
| 2026-05-17 | **Deepfake-on-screen samples added to Phase 1 training data** |
| 2026-05-17 | **Model OTA infrastructure = P0 in Phase 1** (signed bundles, version pin, kill-switch) |
| 2026-05-17 | **Preprocessing pipeline versioned with model artifact** (prevents silent regression) |
| 2026-05-17 | **Phase 1.5 active liveness uses Google ML Kit (Android) / Apple Vision (iOS) / MediaPipe (Web)** — no in-house gesture detector needed |
| 2026-05-17 | **Phase 1.5 anti-injection: Play Integrity (Android) + AppAttest (iOS) + server frame integrity** |
| 2026-05-17 | RunPod 4090 Secure (not Community) for training reliability; 30-min checkpoints |
| 2026-05-17 | Reference parity verified against actual reference code |
