# Changelog

## [0.1.0-week1-scaffold] — 2026-05-17

### Added
- Repo scaffold: pyproject.toml (uv-managed), .gitignore, LICENSE (proprietary), README.md
- Package skeleton: `src/livenix/` with `models/`, `data/`, `train/`, `inference/`, `export/`, `eval/`, `server/`
- **Backbone**: MobileNetV4-Conv-Small (timm, Apache 2.0) with CDC stem (refactored as standard Conv − θ·mean-Conv ops, no custom autograd)
- **Main head**: 3-class classifier (real / print_spoof / replay_spoof)
- **Export pipeline**: PyTorch → ONNX (opset 17) → TFLite (onnx2tf) + CoreML (coremltools 8.x)
- **Cross-runtime parity test**: 100 fixed-seed inputs, ≤1e-2 tolerance across PyTorch / ONNX / TFLite / CoreML
- **QA Gate 1 (Export gate)**: untrained model exports cleanly to all targets before any training spend
- Documentation: [PLAN.md](PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md), [CLAIMS.md](CLAIMS.md), [WEEK_1.md](WEEK_1.md)

### Intentionally deferred
- Aux heads (FFT, SigLIP distillation, patch contrastive) — Week 2
- Dataset loaders (CelebA-Spoof, WMCA, HiFiMask) — Week 2
- Training loop — Week 2
- INT8 quantization — Week 5
- Server (FastAPI) — Phase 1.5
- Active liveness (Google ML Kit / Apple Vision / MediaPipe) — Phase 1.5
- Anti-injection (Play Integrity / AppAttest) — Phase 1.5

### Notes
- v0.1 weights are labeled "research prototype, not for commercial deployment"
- Zero GPU spend in Week 1
- Clean reimplementation — no code copied from any reference repo
