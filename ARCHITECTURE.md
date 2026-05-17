# Livenix — Architecture (How It Works)

> Companion to [PLAN.md](PLAN.md). Shows end-to-end data + control flow across training, inference, and deployment.

---

## 1. System overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            LIVENIX SYSTEM                                │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  TRAINING    │───►│   MODEL      │───►│  DEPLOYMENT TARGETS      │   │
│  │  (offline,   │    │  (one        │    │  - Web (ONNX RT Web)     │   │
│  │   one-time)  │    │   PyTorch    │    │  - Android (TFLite)      │   │
│  └──────────────┘    │   .pth)      │    │  - iOS (CoreML)          │   │
│                      └──────────────┘    │  - Server (Phase 1.5)    │   │
│                                          └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Training pipeline (offline, runs once per release)

```
Research datasets (Phase 1 v0.1 prototype — weights labeled non-commercial)
   │
   ├── CelebA-Spoof (50%)              [print, replay, volume]
   ├── WMCA / CASIA-CeFA (25%)         [cross-ethnicity, robustness]
   ├── HiFiMask (10%)                  [silent silicone exposure, not claimed]
   ├── Deepfake-on-screen (10%)        [SiW-Mv2 + recorded DeepFaceLive playback]
   └── In-house collection (5%)        [our own data, legally clean, ramps weeks 2-6]
   │
   ▼
┌──────────────────────────────────────────┐
│ PREPROCESSING                            │
│  - Face detection: YuNet ONNX            │
│    (~1ms, MIT-licensed, cross-platform)  │
│  - Single-scale patch crop (face bbox    │
│    expanded by 1.3× margin)              │
│  - ISP-aware augmentation:               │
│    · Random JPEG q ∈ [40, 95]            │
│    · Color/gamma jitter                  │
│    · Simulated demosaic artifacts        │
│    · Random brightness, blur             │
│  - Resize to 128×128                     │
│  - Normalize (RGB, ImageNet mean/std)    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ MODEL (training mode)                    │
│                                          │
│  Input: 128×128 RGB                      │
│    │                                     │
│  CDC stem (refactored as standard Conv)  │
│    │                                     │
│  MobileNetV4-Conv-Small backbone (timm)  │
│    │                                     │
│  Feature map                             │
│    ├──► Main head: 3-class               │
│    │    (real / print_spoof / replay_spoof)
│    │    + cosine margin (ArcFace m=0.3)  │
│    │                                     │
│    ├──► Aux 1: FFT magnitude branch      │
│    │    (training-only, stripped at export)
│    │                                     │
│    ├──► Aux 2: SigLIP-So400m distillation│
│    │    (training-only, stripped)        │
│    │                                     │
│    └──► Aux 3: Patch contrastive         │
│         (training-only, stripped)        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│ LOSS                                     │
│  Focal CE (γ=2) + Asymmetric on main head│
│  + 0.5 × MSE on SigLIP distillation      │
│  + 0.2 × BCE on FFT branch               │
│  + 0.3 × NT-Xent on patch contrastive    │
└──────────────┬───────────────────────────┘
               │
               ▼
        Checkpoint .pth
```

**Output of training:** one `.pth` file containing only the backbone + main head. Aux heads exist in code for training but are NOT in the saved checkpoint.

---

## 3. Build pipeline (offline, runs per release)

```
PyTorch checkpoint (.pth)
      │
      ▼
┌──────────────────────────────────────┐
│ EXPORT TO ONNX                       │
│  - torch.onnx.export, opset 17       │
│  - Strip aux heads (already stripped)│
│  - Constant-fold                     │
│  - Verify shape on test input        │
└──────────────┬───────────────────────┘
               │  livenix.onnx (~6 MB FP32)
               │
               ├──► onnx2tf      ──► livenix.tflite       (~6 MB FP32)
               │                       │
               │                       ▼ PTQ INT8 calib
               │                     livenix_int8.tflite  (~2 MB)
               │
               ├──► coremltools  ──► livenix.mlpackage    (~6 MB FP16)
               │
               ├──► (no convert) ──► livenix.onnx         (~6 MB) for Web
               │
               └──► pnnx         ──► livenix.ncnn         (Phase 2)
                                      │
                                      ▼ INT8 quant
                                    livenix.ncnn.int8     (~2 MB)
```

**Parity gate:** all targets must match PyTorch outputs within tolerance on a fixed 1000-image test bundle.

---

## 4. Inference — Phase 1 (mobile / web, on-device only)

```
┌──────────────────┐
│  CLIENT APP      │
│  (mobile/web)    │
└────────┬─────────┘
         │
         ▼ camera frame (or video frame)
┌──────────────────────────────────────┐
│ SDK preprocessing (NATIVE per platform)│
│  - Face detect (YuNet ONNX, ~1ms)    │
│  - Crop face bbox + 1.3× margin      │
│  - Resize to 128×128                 │
│  - Normalize (RGB, ImageNet mean/std)│
│  - Convert to runtime tensor format  │
│  - Preprocessing version pinned to   │
│    model artifact (Gate 14)          │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ ON-DEVICE MODEL                      │
│  - INT8 TFLite/CoreML/ONNX           │
│  - Outputs 3 logits: real, print, rep│
│  - Softmax → probabilities           │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│ DECISION LAYER (in SDK)              │
│  isLive  = (argmax == real)          │
│  spoofType = "print"|"replay"|"none" │
│  confidence = softmax[argmax]        │
│  source = "on-device"                │
└────────────────┬─────────────────────┘
                 │
                 ▼
            Return to app
```

**API contract (same Phase 1 and Phase 1.5):**
```ts
type LivenessResult = {
  isLive: boolean
  spoofType: "none" | "print" | "replay"
  confidence: number       // 0..1
  source: "on-device" | "server-referred"
  latencyMs: number
  modelVersion: string
}
```

---

## 5. Inference — Phase 1.5 (hybrid, drop-in upgrade, no SDK rewrite)

```
┌──────────────────┐
│  CLIENT APP      │
└────────┬─────────┘
         │ Step A: passive PAD
         ▼
┌──────────────────────────────────────┐
│ On-device model (same as Phase 1)    │
└────────────────┬─────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │ Confidence routing:          │
    │   conf > 0.8 → trust local   │
    │   conf < 0.2 → trust local   │
    │   else → refer + challenge   │
    └────────────────┬─────────────┘
       local         │     refer/challenge
       ┌─────────────┴──────────────────┐
       ▼                                ▼
    Return            ┌──────────────────────────────────┐
                      │ Step B: ACTIVE LIVENESS          │
                      │ Server issues random challenge:  │
                      │  { nonce, "blink_twice",         │
                      │    windowMs: 3000 }              │
                      └──────────────────┬───────────────┘
                                         ▼
                      ┌──────────────────────────────────┐
                      │ SDK runs Google ML Kit (Android) │
                      │ / Apple Vision (iOS)             │
                      │ / MediaPipe (Web)                │
                      │ - Eye open/closed state per frame│
                      │ - Head Euler angles (yaw/pitch)  │
                      │ - Timing within window           │
                      └──────────────────┬───────────────┘
                                         ▼
                      ┌──────────────────────────────────┐
                      │ POST /v1/verify                  │
                      │  { passive_frames, active_result,│
                      │    nonce, integrity_tokens }     │
                      └──────────────────┬───────────────┘
                                         ▼
                      ┌──────────────────────────────────┐
                      │ SERVER (FastAPI)                 │
                      │ - Anti-injection: verify         │
                      │   Play Integrity / AppAttest     │
                      │   tokens                         │
                      │ - Run Conv-Large on passive frames│
                      │ - Verify challenge result + nonce│
                      │ - Combine scores                 │
                      │ - Log telemetry                  │
                      └──────────────────┬───────────────┘
                                         ▼
                          Return authoritative isLive
                          + spoofType + telemetry
```

**Active liveness uses platform-native primitives — no custom gesture detector needed:**
- **Android**: [Google ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection) — eye open probability, head Euler angles, smile probability
- **iOS**: [Apple Vision Framework](https://developer.apple.com/documentation/vision) — `VNDetectFaceLandmarksRequest` gives same primitives
- **Web**: [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) — WASM-based, same primitives

**What active liveness defeats that passive alone doesn't:**
| Attack | Passive (Phase 1) | + Active (Phase 1.5) |
|---|---|---|
| Pre-recorded video on screen | ✅ Caught (replay) | ✅ Stronger — can't blink on cue |
| Live video call | ✅ Caught (replay) | ✅ Stronger — challenge timing |
| Static 3D mask | ⚠️ Hard | ✅ Mask can't articulate |
| Virtual-camera injection (OBS, DeepFaceLive) | ❌ Misses | ✅ Defeats — injected feed can't respond to server-issued random challenge in real time |

**Phase 1 SDK already returns `confidence` and `source`. Phase 1.5 just adds `verifyChallenge()` call + Google ML Kit dependency. Zero rewrite of Phase 1 code.**

---

## 6. Repo layout (to be scaffolded Week 1)

```
livenix/
├── PLAN.md                  ← strategy + decisions log
├── ARCHITECTURE.md          ← this file
├── CLAIMS.md                ← allowed/disallowed public claims
├── README.md                ← project intro (later)
├── pyproject.toml           ← uv-managed Python project
├── .gitignore
├── LICENSE                  ← our own license (TBD: proprietary or MIT)
│
├── src/livenix/
│   ├── __init__.py
│   ├── models/
│   │   ├── backbone.py      ← MobileNetV4 + CDC stem wrapper
│   │   ├── heads.py         ← main 3-class head + aux heads
│   │   └── full_model.py    ← combined module
│   ├── data/
│   │   ├── celeba_spoof.py  ← loader
│   │   ├── wmca.py
│   │   ├── hifimask.py
│   │   ├── oulu_npu.py      ← eval-only loader
│   │   ├── transforms.py    ← ISP-aware augmentation
│   │   └── splits.py        ← identity-disjoint split + contamination audit
│   ├── train/
│   │   ├── losses.py        ← focal + asymmetric + NT-Xent
│   │   ├── trainer.py       ← Lightning-style loop
│   │   └── config.py        ← YAML configs per variant (mobile/web/server)
│   ├── inference/
│   │   ├── predictor.py     ← reference Python inference
│   │   └── preprocess.py    ← face detect + crop + normalize
│   ├── export/
│   │   ├── to_onnx.py
│   │   ├── to_tflite.py
│   │   ├── to_coreml.py
│   │   ├── quantize.py      ← INT8 PTQ
│   │   └── parity_test.py   ← QA Gate 11: cross-runtime numerical parity
│   ├── eval/
│   │   ├── metrics.py       ← APCER/BPCER/TPR@FPR/ECE
│   │   ├── fairness.py      ← per-demographic breakdown
│   │   ├── adversarial.py   ← FGSM, PGD-20
│   │   └── benchmarks.py    ← OULU-NPU protocols 1, 4
│   └── server/              ← Phase 1.5
│       ├── api.py           ← FastAPI app
│       ├── routing.py       ← confidence-band referral
│       └── telemetry.py
│
├── configs/
│   ├── mobile_small.yaml    ← MobileNetV4-Conv-Small variant
│   ├── web_small.yaml       ← 0.5× variant for browser
│   └── server_large.yaml    ← Phase 1.5 Conv-Large
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── export_all.py
│   ├── run_qa_gates.py      ← orchestrates all QA gates in order
│   └── webcam_demo.py       ← real-time desktop demo
│
├── tests/
│   ├── test_export_gate.py        ← QA Gate 1
│   ├── test_overfit_gate.py       ← QA Gate 3
│   ├── test_parity.py             ← QA Gate 11
│   └── test_contamination.py      ← QA Gate 12
│
└── demos/
    ├── webcam/                    ← Python desktop demo
    └── web/                       ← ONNX Runtime Web demo
```

---

## 7. Phase 1 deliverable summary

At end of Week 6, this is what gets handed over:

| Artifact | Form | Purpose |
|---|---|---|
| `livenix.onnx` | FP32 ONNX, ~6 MB | Source of truth, hub format |
| `livenix.tflite` (INT8) | Android | Mobile SDK Phase 2 wraps this |
| `livenix.mlpackage` (FP16) | iOS | Mobile SDK Phase 2 wraps this |
| `livenix.onnx` for Web | Browser via ONNX Runtime Web | Web demo + future Web SDK |
| `webcam_demo.py` | Python | Real-time testing, manual QA |
| `web/` demo | Static site | Browser-based real-time testing |
| `eval_report.md` | Markdown | All KPI numbers, ROC plots, confusion matrices, fairness breakdowns |
| `THREAT_MODEL.md` | Markdown | What we defend, what we don't, allowed claims |
| `CHANGELOG.md` | Markdown | Versions + KPIs per release |

Phase 1.5 adds the server: `server/api.py` + Conv-Large model checkpoint + deployment scripts.
