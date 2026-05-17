# Livenix — Claims Policy

> What we can say publicly about Livenix without failing an audit.
> Companion to [PLAN.md](PLAN.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ✅ Allowed claims (Phase 1)

Marketing, product pages, sales decks, technical briefings.

### Attack coverage
- "Detects photo (printed paper) spoof attempts"
- "Detects screen replay spoof attempts (video call, recorded clip, camera-feed replay)"
- "Passive face liveness — no user action required"
- "Single-frame inference, sub-50ms on mobile"

### Performance (only with measured numbers attached)
- "X% TPR at FPR=1e-4 on CelebA-Spoof test split" (with the actual number — TPR@FPR=1e-4 is industry standard with published comparables)
- "ACER X% on OULU-NPU Protocol 4" (internal benchmark only — do not publish in commercial datasheets without commercial dataset license)
- "INT8 quantized, ~5 MB on disk"

### Phase 1.5 capabilities (after they ship)
- "Active liveness challenge: blink and head-pose detection via Google ML Kit / Apple Vision / MediaPipe"
- "Server-side anti-injection: Play Integrity (Android) + AppAttest (iOS) verification"
- "Defeats virtual-camera injection attacks (OBS, DeepFaceLive)"
- "Hybrid on-device + server scoring"

### Deployment
- "Cross-platform: web, iOS, Android"
- "On-device inference; optional server-side authoritative scoring (Phase 1.5)"
- "ONNX-based pipeline, exports to TFLite, CoreML"

### Origin / IP
- "Developed in-house"
- "Modern (2024+) mobile architecture (MobileNetV4 family)"
- Phase 1 (v0.1 prototype): "Trained on public research datasets for technical validation; commercial deployment uses licensed or in-house data"
- Phase 2 (v1.0 commercial): "Trained on commercially licensed datasets and in-house collected data" (after dataset path lands)

---

## ❌ Disallowed claims (Phase 1)

Do not say these in any public-facing material, sales deck, or marketing copy.

### Attack coverage we cannot defend
- ❌ "Defeats silicone masks"
- ❌ "Defeats 3D-printed masks"
- ❌ "Defeats 3D human masks"
- ❌ "Deepfake-proof"
- ❌ "Real-time deepfake detection"
- ❌ "AI-generated face detection"
- ❌ Unqualified "anti-spoofing" (always pair with attack list)
- ❌ "Multi-modal" (we are RGB-only)
- ❌ "Active liveness" (we are passive only)

### Certifications we don't have yet
- ❌ "ISO/IEC 30107-3 compliant" (only after iBeta certifies)
- ❌ "iBeta PAD Level 1 certified" (only after passing)
- ❌ "iBeta PAD Level 2 certified" (Phase 2 target at earliest)
- ❌ "NIST FRVT PAD listed" (only after submission + listing)
- ❌ "Government certified" (no government has certified us)

### Origin claims to avoid
- ❌ "Based on Minivision Silent-Face-Anti-Spoofing" (we reimplemented, no need to cite)
- ❌ Any reference to "Minivision," "MiniFASNet," or their trademarks
- ❌ Calling it "Apache 2.0 based" (we don't redistribute their code)

---

## Phase 1.5 / Phase 2 — claims that unlock later

These become allowed **only when** we have measured numbers and pass relevant tests.

| Claim | Unlocks when |
|---|---|
| "Silicone-mask resilience" | When HiFiMask cross-domain TPR@FPR=1e-3 ≥ 80%, with documented numbers |
| "Deepfake-resistant (screen replay class)" | Today, but only worded as "detects deepfake content shown on a screen" — caught as replay, not as deepfake semantic detection |
| "iBeta PAD Level 1 certified" | After iBeta lab tests pass and certificate issued |
| "ISO/IEC 30107-3 compliant" | Same as above |
| "Hybrid on-device + server" | When Phase 1.5 ships |
| "Active liveness challenge" | When Phase 1.5 ships |
| "Defeats virtual-camera injection" | When Phase 1.5 anti-injection ships |
| "Multi-frame temporal model" | Phase 2 |

---

## Defensibility cheat-sheet

If a customer or auditor asks **"is this really your product or did you copy something?"**, the answer is:

> "Livenix is developed in-house. We use the open-source PyTorch ecosystem (timm for backbone, standard PyTorch for training and export). The conceptual approach — passive RGB liveness with Fourier-based auxiliary supervision — is well-established in published academic literature (see CDCN, CelebA-Spoof, MobileNetV4 papers). Our implementation, training data, weights, and SDK are entirely our own. No third-party proprietary code is incorporated. All dependencies are listed in `pyproject.toml`."

If asked **"why don't you claim silicone-mask defense like Vendor X does?"**:

> "Passive RGB single-frame face liveness has a known capability ceiling against silicone masks, even at much larger model sizes. We chose to be precise about what we defend in Phase 1 (print and screen replay), and to ship silicone-mask defense in Phase 2 when our architecture supports it credibly. Vendors who claim it today should be asked for iBeta PAD Level 2 results — that's the standard test for silicone."

This is the position. It's true, it's defensible, it's marketable.
