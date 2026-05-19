# Livenix — Android Integration Handoff

> Self-contained brief for a new Claude session (or developer) to integrate the
> trained Livenix v0.1 binary model into an Android app. **Do NOT discuss
> training/data acquisition in that session — the model already exists.**

---

## What's already done

✅ **Model trained**: MobileNetV4-Conv-Small + CDC stem, ~1.26M params, 5 MB
✅ **Binary classifier**: outputs 2 logits — `[real, spoof]`
✅ **96.41% accuracy** on CelebA-Spoof cropped (caveat: train/val same data; expect 75-88% in production)
✅ **Exported to ONNX + CoreML** — files exist at `runs/v0.1-binary/export/`

## What's NOT done (out of scope for the Android session)

❌ Cross-dataset evaluation (WMCA, OULU-NPU)
❌ Proper identity-disjoint train/val split
❌ Adversarial testing
❌ Fairness audit across skin tones
❌ Active liveness (use Google ML Kit separately)
❌ Anti-injection (use Play Integrity API)

---

## Files you need

### Model file for Android
```
runs/v0.1-binary/export/livenix.onnx        ← 5.04 MB, use this on Android
runs/v0.1-binary/export/livenix.mlpackage   ← iOS only
runs/v0.1-binary/checkpoint_best.pth        ← original PyTorch weights
```

### Model contract
| Property | Value |
|---|---|
| Framework | ONNX (opset 17) |
| Input name | `input` |
| Input shape | `(1, 3, 128, 128)` — NCHW, batch dynamic |
| Input dtype | float32 |
| Input range | ImageNet normalization (see below) — **NOT [-1,1]** |
| Output name | `logits` |
| Output shape | `(1, 2)` |
| Output | Raw logits — apply softmax for probabilities |
| Class 0 | `real` (live face) |
| Class 1 | `spoof` (print, replay, paper mask, etc.) |

### Inference pseudocode (any language)

> ⚠️ **CRITICAL**: the model is trained with **ImageNet** normalization,
> NOT `(x-0.5)/0.5`. Using the wrong norm produces near-random outputs.
> A previous version of this doc had this bug — confirmed on Android
> deployment 2026-05-20.

```
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # RGB order
IMAGENET_STD  = [0.229, 0.224, 0.225]

face_bbox = run_face_detector(camera_frame)            # ML Kit / MediaPipe
crop = crop_and_resize(camera_frame, face_bbox, 128, 128)  # uint8 HWC RGB

x = crop.astype(float32) / 255.0                       # [0, 1]
x = (x - IMAGENET_MEAN) / IMAGENET_STD                 # per-channel
x = x.transpose(HWC -> CHW).expand_dims(batch=0)       # (1, 3, 128, 128)

logits = onnx_model.run(x)                             # (1, 2)
probs  = softmax(logits)                               # [P_real, P_spoof]
is_real = probs[0] > THRESHOLD                         # 0.6-0.7 tuned per device
```

**Color order:** ML Kit / Android Bitmaps give you ARGB. Strip alpha and
ensure RGB order (NOT BGR) before normalizing.

---

## Android integration — recommended stack

| Layer | Library | Why |
|---|---|---|
| ONNX runtime | **ONNX Runtime Mobile** (`onnxruntime-android`) | Official, supports NNAPI/XNNPACK acceleration |
| Camera | **CameraX** | Modern Android camera API |
| Face detection | **Google ML Kit Face Detection** | Free, on-device, fast |
| Active liveness | **Google ML Kit (existing in your app)** | Blink/turn-head challenges |
| Anti-injection | **Play Integrity API** | Verifies device + app integrity |
| Image processing | **ML Kit InputImage** + bitmap utilities | Avoid manual pixel math |

### Maven dependency
```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
```

### Asset placement
Copy `livenix.onnx` to: `app/src/main/assets/livenix.onnx`

---

## Architecture flow (what to build)

```
┌──────────────────────────────────────────────────────────────┐
│  Camera (CameraX) → frame                                    │
│      ↓                                                       │
│  ML Kit Face Detector → face bbox (or null if no face)       │
│      ↓                                                       │
│  Crop + resize 128×128, normalize → input tensor             │
│      ↓                                                       │
│  ONNX Runtime (livenix.onnx) → logits [2]                    │
│      ↓                                                       │
│  Softmax → P(real), P(spoof)                                 │
│      ↓                                                       │
│  Threshold (e.g. P(real) > 0.7) → liveness OK / FAIL         │
│      ↓                                                       │
│  Combine with: Active liveness (ML Kit) +                    │
│                Play Integrity attestation                    │
│      ↓                                                       │
│  Final decision: allow user through identity flow            │
└──────────────────────────────────────────────────────────────┘
```

---

## Tasks for the Android session (in order)

1. **Project skeleton**
   - New Android Studio project, Kotlin, minSdk 24, CameraX
   - Add ONNX Runtime Mobile + ML Kit Face Detection deps

2. **Camera + face detection**
   - CameraX preview with front camera
   - Bind ML Kit face detector to image analyzer
   - Draw bbox on overlay (debug)

3. **ONNX inference wrapper**
   - Singleton `LivenixDetector` class
   - Load `livenix.onnx` from assets on init
   - Method: `detect(bitmap: Bitmap): LivenixResult(realProb, spoofProb)`

4. **Crop + preprocess**
   - Use the ML Kit face bbox to crop the input frame
   - Resize to 128×128 with bilinear interpolation
   - Normalize: `(pixel/255 - 0.5)/0.5`, layout NCHW

5. **Decision logic**
   - Threshold: `realProb > 0.7` → pass (tunable)
   - Smoothing: require N consecutive passes (avoid frame flicker)

6. **Demo UI**
   - Live preview with "LIVE" / "SPOOF" overlay + confidence %
   - Test against: own face, printed photo of own face, phone screen showing own face

7. **(Phase 1.5 — separate sprint)**
   - Add ML Kit active liveness challenges (already in your existing app)
   - Add Play Integrity API call before allowing the liveness check
   - Add server-side nonce flow if backend exists

---

## Critical contract notes for the Android dev

- **The model expects a PRE-CROPPED face**, NOT a full camera frame. Without face detection upstream, accuracy will collapse.
- **Front camera is the deployment target.** The training data was front-facing portraits.
- **Output is raw logits** — apply softmax to get probabilities. Don't compare logits directly to 0.5.
- **128×128 is small** — keep the face bbox tight (just face, not torso) before resizing.
- **Phase 1 = passive only.** Combine with active liveness (blink/turn) for production-grade liveness.

---

## What to ask the new Claude session

Open the new session with something like:

> I have a trained ONNX face-liveness model at `runs/v0.1-binary/export/livenix.onnx`. It takes a 128×128 RGB face crop normalized to [-1,1] and outputs 2 logits (real / spoof).
>
> Help me build an Android demo app using CameraX + ML Kit face detection + ONNX Runtime Mobile that:
> 1. Shows live camera preview
> 2. Detects a face, crops it
> 3. Runs the model
> 4. Displays LIVE / SPOOF + confidence on screen
>
> Reference: `ANDROID_INTEGRATION_HANDOFF.md` in this repo has the full model contract and architecture.

That session will have all the info it needs without re-learning the model side.

---

## Useful repo references

| Path | What it has |
|---|---|
| `src/livenix/models/full_model.py` | The exact model architecture |
| `src/livenix/data/transforms.py` | The training-time normalization (mean/std) |
| `scripts/export_checkpoint.py` | The export script — useful for re-exports if model changes |
| `PLAN.md` | Overall product plan, KPIs, claims |
| `CLAIMS.md` | What we can/cannot say about the model |
| `ARCHITECTURE.md` | End-to-end system architecture |

---

## Cost of Android session

- No GPU needed
- No RunPod
- Pure local development on Mac → Android Studio → physical phone
- ~free except your time

---

## When the Android session is done, you'll have

- ✅ APK that runs on your phone
- ✅ Live demo of liveness detection
- ✅ Reusable `LivenixDetector` SDK class
- ✅ Foundation for the full government identity flow
