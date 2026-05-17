# Livenix — Week 1 Handoff (Scaffold + QA Gate 1)

> **For the next implementation session.** Self-contained — assumes you've read [PLAN.md](PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md), [CLAIMS.md](CLAIMS.md), but nothing more.
> **Goal of Week 1:** scaffold the repo + verify the architecture exports cleanly to ONNX/TFLite/CoreML on an UNTRAINED model. Zero GPU spend. Sets up Week 2 (training) for success.
> **Verifier session (Opus):** uses [QA_GATE_1_CHECKLIST.md](#qa-gate-1-acceptance-checklist) below to independently verify Week 1 output.

---

## Operating context (do not skip)

- **Repo root:** `/Users/hamdev/livenix/`
- **Package name:** `livenix` (lowercase, throughout the codebase)
- **Brand name:** Livenix (capital L, in docs/UI)
- **Python tooling:** `uv` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` if not present)
- **License:** All Rights Reserved (proprietary). Create `LICENSE` file with `Copyright (c) 2026 [owner]. All Rights Reserved.`
- **Git:** init from first commit; never copy files from `/Users/hamdev/Silent-Face-Anti-Spoofing/`
- **GPU target later (Week 2):** RunPod 4090 **Secure** (not Community) for reliability
- **Hard constraint:** NO TRAINING in Week 1. Only scaffold + untrained export gate.

---

## Week 1 task list (in order)

### Task 1 — Install uv and init project
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart shell
cd /Users/hamdev/livenix
uv init --package livenix --python 3.11
```

### Task 2 — Configure pyproject.toml dependencies

Add to `pyproject.toml`:
```toml
[project]
name = "livenix"
version = "0.1.0"
description = "Livenix face liveness SDK — Phase 1 prototype"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "torchvision>=0.18.0",
    "timm>=1.0.7",               # backbone source (Apache 2.0)
    "onnx>=1.16.0",
    "onnxruntime>=1.18.0",
    "onnx2tf>=1.22.0",           # ONNX → TFLite
    "coremltools>=8.0",          # ONNX → CoreML
    "opencv-python>=4.10.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0.0",
    "pillow>=10.0.0",
    "albumentations>=1.4.0",     # ISP-aware aug
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-xdist>=3.5.0",
    "ruff>=0.5.0",
    "ipython>=8.0.0",
]
train = [
    "pytorch-lightning>=2.3.0",
    "tensorboard>=2.16.0",
    "wandb>=0.17.0",             # optional
    "transformers>=4.40.0",      # for SigLIP teacher
]
```

Then: `uv sync --extra dev` (lightweight install for Week 1; `--extra train` added Week 2).

### Task 3 — Scaffold repo layout

Create the directory structure from [ARCHITECTURE.md §6](ARCHITECTURE.md#6-repo-layout-to-be-scaffolded-week-1). Empty `__init__.py` in each Python package, stub modules with `pass` or `raise NotImplementedError` where appropriate.

**Critical Week 1 files (must have actual code, not stubs):**
- `src/livenix/models/backbone.py` — MobileNetV4-Conv-Small + CDC stem (see Task 4)
- `src/livenix/models/heads.py` — Main 3-class head only (aux heads can be NotImplementedError for Week 1)
- `src/livenix/models/full_model.py` — Combined module with inference-mode forward (no aux heads)
- `src/livenix/export/to_onnx.py` — ONNX export script
- `src/livenix/export/to_tflite.py` — TFLite via onnx2tf
- `src/livenix/export/to_coreml.py` — CoreML via coremltools
- `src/livenix/export/parity_test.py` — Cross-runtime numerical parity
- `tests/test_export_gate.py` — QA Gate 1 test
- `scripts/run_qa_gate_1.py` — Orchestrates Gate 1

**Stubs OK for Week 1 (implemented Week 2+):**
- `src/livenix/data/*` — empty stubs
- `src/livenix/train/*` — empty stubs
- `src/livenix/eval/*` — empty stubs
- `src/livenix/server/*` — empty stubs

### Task 4 — Implement backbone + CDC stem + main head (untrained)

`src/livenix/models/backbone.py`:
- Use `timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=0, global_pool='')` to get the feature extractor (no head, no global pool)
- Wrap with CDC stem **prepended** to the first conv
- CDC stem implementation: `output = stem_conv(x) - theta * mean_conv(x)` where `mean_conv` has same weight shape but values = `weight.mean(dim=(2,3), keepdim=True).expand_as(weight)`, theta=0.7
- **Banned ops** (must NOT appear in forward graph): FFT, `grid_sample`, `GroupNorm`, custom autograd, dynamic shapes, custom `Hardswish` expressions

`src/livenix/models/heads.py`:
- `MainHead`: takes pooled feature vector, outputs 3 logits (real / print_spoof / replay_spoof)
- Implement with `nn.Linear(in_features, 3)` — cosine margin / ArcFace is applied at LOSS time, not in head (keep head simple for export)
- Aux heads (`FFTHead`, `SigLIPDistillHead`, `PatchContrastiveHead`): raise `NotImplementedError("Implemented in Week 2")` — they don't need to exist for Week 1 export gate

`src/livenix/models/full_model.py`:
```python
class LivenixModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = CDCStem() + MobileNetV4ConvSmall()
        self.head = MainHead(in_features=..., num_classes=3)
        # aux heads only constructed in training mode (Week 2)

    def forward(self, x):
        # Inference-mode forward — what gets exported.
        # Input: (B, 3, 128, 128) FP32 RGB normalized
        # Output: (B, 3) logits
        f = self.backbone(x)
        f = f.mean(dim=(2, 3))  # global avg pool — use plain mean, not AdaptiveAvgPool
        return self.head(f)
```

**Note:** Use `f.mean(dim=(2,3))` instead of `nn.AdaptiveAvgPool2d` — adaptive pool sometimes serializes poorly in TFLite/CoreML. Plain reduce-mean is portable.

### Task 5 — Implement export scripts

`src/livenix/export/to_onnx.py`:
```python
def export_to_onnx(model: nn.Module, output_path: str, input_size=(1, 3, 128, 128)):
    model.eval()
    dummy = torch.randn(*input_size)
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
```

`src/livenix/export/to_tflite.py`:
- Use `onnx2tf.convert` with `output_signaturedefs=True`
- No quantization in Week 1 — FP32 only (quantization is Week 5)

`src/livenix/export/to_coreml.py`:
- Use `coremltools.convert(model="livenix.onnx", source="onnx", convert_to="mlprogram")`
- Set `compute_units=coremltools.ComputeUnit.ALL` and `minimum_deployment_target=ct.target.iOS16`

`src/livenix/export/parity_test.py`:
- Generate 100 fixed-seed random inputs
- Run through PyTorch, ONNX (onnxruntime), TFLite (tf.lite.Interpreter), CoreML (coremltools)
- Assert max absolute difference ≤ 1e-2 between any pair
- Return per-runtime mean/max diff for the report

### Task 6 — Run QA Gate 1

`scripts/run_qa_gate_1.py`:
1. Instantiate untrained `LivenixModel`
2. Forward a dummy input — assert output shape `(1, 3)` and no NaN
3. Export to ONNX → `build/livenix_untrained.onnx`
4. Convert to TFLite → `build/livenix_untrained.tflite`
5. Convert to CoreML → `build/livenix_untrained.mlpackage`
6. Run parity test — must pass ≤1e-2
7. Print pass/fail summary

This is **QA Gate 1**. If anything fails, fix architecture before training (saves all GPU spend).

### Task 7 — First clean git commit

```bash
cd /Users/hamdev/livenix
git init
# Add a .gitignore for Python/PyTorch/ML (uv tools, __pycache__, *.pth, build/, .venv/, etc.)
git add .
git commit -m "Initial scaffold: Livenix Phase 1 — clean reimplementation

- MobileNetV4-Conv-Small backbone + CDC stem (untrained)
- 3-class main head (real / print_spoof / replay_spoof)
- Export scripts: PyTorch → ONNX → TFLite + CoreML
- QA Gate 1 (untrained export gate) implementation
- Docs: PLAN.md, ARCHITECTURE.md, CLAIMS.md, WEEK_1.md

No code or weights from any third-party repo. All concepts reimplemented
from published papers (CDCN, MobileNetV4, CelebA-Spoof, PatchNet)."
```

---

## QA Gate 1 acceptance checklist

**For the verifier session (Opus) — independently verify Week 1 output meets these criteria. Run each check and report pass/fail.**

### Repo hygiene
- [ ] `/Users/hamdev/livenix/` exists with the directory layout from [ARCHITECTURE.md §6](ARCHITECTURE.md#6-repo-layout-to-be-scaffolded-week-1)
- [ ] `pyproject.toml` exists with deps from Task 2; `uv lock` succeeded
- [ ] `.venv/` exists after `uv sync`
- [ ] No file in the repo references `/Users/hamdev/Silent-Face-Anti-Spoofing/` or contains string `Minivision`
- [ ] No `.pth` file imported from the reference repo
- [ ] `git log` shows clean history starting at first Livenix commit (no merged history from reference)
- [ ] `LICENSE` is "All Rights Reserved" or equivalent proprietary; not Apache 2.0
- [ ] `.gitignore` excludes `.venv/`, `__pycache__/`, `*.pth`, `build/`, `*.onnx`, `*.tflite`, `*.mlpackage`

### Architecture correctness
- [ ] `LivenixModel.forward(input=torch.randn(1,3,128,128))` returns tensor of shape `(1, 3)` without error
- [ ] Output has no NaN/Inf for an untrained model
- [ ] Backbone is `timm.mobilenetv4_conv_small` family (verify by `print(model.backbone)`)
- [ ] CDC stem is implemented as `Conv − θ·mean_Conv` (NOT a custom autograd function — must be standard ops)
- [ ] `theta=0.7` per [PLAN.md architecture table](PLAN.md)
- [ ] Aux heads exist as classes but raise `NotImplementedError` for Week 1 (intentional — Week 2 work)
- [ ] No FFT, `grid_sample`, `GroupNorm`, or custom `Hardswish` ops appear in the inference forward graph

### Export gate (the actual gate)
- [ ] `python scripts/run_qa_gate_1.py` exits 0
- [ ] `build/livenix_untrained.onnx` is created (~6 MB)
- [ ] `build/livenix_untrained.tflite` is created
- [ ] `build/livenix_untrained.mlpackage/` directory is created
- [ ] Parity test passes: max abs diff ≤ 1e-2 between PyTorch, ONNX, TFLite, CoreML on 100 fixed-seed inputs
- [ ] Each export script can be run independently (smoke test)

### Documentation parity
- [ ] No contradictions between PLAN.md, ARCHITECTURE.md, CLAIMS.md (re-read after Week 1 work)
- [ ] CHANGELOG.md exists with a "0.1.0-week1-scaffold" entry summarizing what was built

### Banned ops check (manual inspection of forward graph)
- [ ] `print([n.op_type for n in onnx.load(onnx_path).graph.node])` does NOT contain: `FFT`, `STFT`, `GridSample`, `GroupNormalization`, `If`, `Loop`, `Scan`
- [ ] No `aten::pixel_shuffle`, `aten::einsum`, `aten::scatter` in trace

### Week 1 spend
- [ ] GPU spend so far = $0 (no training ran)
- [ ] uv install was the only system-level change (no apt/brew/pip global installs)

---

## What Week 1 does NOT include (deferred to Week 2+)

- ❌ Training data loaders (CelebA-Spoof / WMCA / HiFiMask / etc.) — Week 2
- ❌ Aux heads implementation (FFT, SigLIP distillation, patch contrastive) — Week 2
- ❌ Loss functions (focal, asymmetric, NT-Xent) — Week 2
- ❌ Training loop — Week 2
- ❌ Any GPU training — Week 2 ($8)
- ❌ INT8 quantization — Week 5
- ❌ Server (FastAPI) — Phase 1.5
- ❌ Active liveness / Google ML Kit integration — Phase 1.5
- ❌ Anti-injection — Phase 1.5

---

## If something blocks during Week 1

| Symptom | Likely cause | Fix |
|---|---|---|
| `timm` doesn't have `mobilenetv4_conv_small.e2400_r224_in1k` | Old timm version | `uv add 'timm>=1.0.7'` |
| ONNX export fails on CDC stem | CDC implemented as custom autograd | Refactor to `Conv − Conv` with shared weights; standard ops only |
| TFLite conversion fails | onnx2tf version mismatch with onnx | Pin `onnx==1.16.0`, `onnx2tf==1.22.0` |
| CoreML conversion fails | coremltools version too old | `uv add 'coremltools>=8.0'` |
| Parity test fails >1e-2 | Likely AdaptiveAvgPool or hardswish op | Replace with `.mean(dim=(2,3))` and `nn.ReLU6` |
| uv install fails | Network or curl missing | Fallback: `pip install uv` then `uv sync` |

---

## Handoff to Week 2

When QA Gate 1 passes:
1. Implementation session commits all Week 1 work
2. Opus verifier session runs the [QA Gate 1 checklist](#qa-gate-1-acceptance-checklist) and reports pass/fail
3. On full pass: Week 2 begins — implement aux heads + training loop + dataset loaders
4. On any fail: fix surfaced items before moving forward (saves GPU spend)

**No training spend until Gate 1 passes.** This is the single most important rule of Week 1.
