# Livenix — Atomic Task Breakdown

> **Purpose:** every remaining piece of work, broken into self-contained tasks small enough to be executed by a smaller/cheaper model (Sonnet, Haiku) and independently verified by Opus.
> **Companion to:** [PLAN.md](PLAN.md), [ARCHITECTURE.md](ARCHITECTURE.md), [CLAIMS.md](CLAIMS.md), [WEEK_1.md](WEEK_1.md).

---

## Execution model

### Implementer role (Sonnet / Haiku)
- Reads the task's **inputs** (specific files only — not the whole repo).
- Writes the **deliverables** exactly as specified.
- Self-checks against the **acceptance criteria** before declaring done.
- Commits per the [commit policy](#commit-policy) below.

### Verifier role (Opus)
- Re-reads ONLY the **deliverables + QA verification commands**.
- Runs the QA verification commands listed in the task.
- Reports pass/fail per criterion, no opinions, no rewrites.
- If fail: lists the specific failure, hands back to Implementer.

### Why this split saves money
- ~80% of work is mechanical (file edits, well-specified). Sonnet handles it.
- Opus is invoked ~1× per task for verification — 5-10× cheaper than letting Opus write everything.

---

## Commit policy

Every task ends with **one commit** on `feature/<short-task-id>` branch (e.g. `feature/t1.1-fix-tflite-deps`). Conventional commit subject. Co-Authored-By trailer. After QA verification passes, merge to `feature/week-1-scaffold` (or current integration branch).

---

## Priority legend

- **P0**: blocks the next gate. Must complete before any downstream task starts.
- **P1**: feature work for the current week.
- **P2**: nice-to-have, can slip a week.
- **Phase 1.5/2**: explicitly deferred.

---

## TASKS

### Week 1 — Finish QA Gate 1

#### T1.1 — Fix TFLite export deps (P0)
- **Inputs**: `pyproject.toml`, error trace `No module named 'tf_keras'`
- **Deliverables**:
  - Add `tf-keras>=2.16.0` to runtime deps in `pyproject.toml`
  - `uv sync --extra dev` succeeds
- **Acceptance criteria**:
  - `uv run python -c "import onnx2tf; print(onnx2tf.__version__)"` exits 0
- **QA verification (Opus)**:
  - `cd /Users/hamdev/livenix && uv run python -c "import onnx2tf, tf_keras; print('OK')"` returns "OK"
- **Effort**: 5 minutes
- **Depends on**: nothing

#### T1.2 — Fix CoreML export deps (P0)
- **Inputs**: `pyproject.toml`, error `Keras cannot be imported`
- **Deliverables**:
  - Verify coremltools 8 actually needs keras (some 8.x versions require it transitively). Either: (a) add `keras>=3.0.0` to deps, OR (b) downgrade coremltools to a version that doesn't require keras.
  - Document the chosen path in code comment.
- **Acceptance criteria**:
  - `uv run python -c "import coremltools as ct; print(ct.__version__)"` exits 0
- **QA verification (Opus)**:
  - `cd /Users/hamdev/livenix && uv run python -c "import coremltools as ct; m = ct.models.MLModel; print('OK')"` returns "OK"
- **Effort**: 15 minutes
- **Depends on**: nothing

#### T1.3 — Re-run QA Gate 1 to full pass (P0)
- **Inputs**: working ONNX + TFLite + CoreML deps after T1.1, T1.2
- **Deliverables**:
  - `build/livenix_untrained.onnx` (exists, ~5 MB)
  - `build/tflite_untrained/livenix*_float32.tflite` (exists)
  - `build/livenix_untrained.mlpackage/` (exists)
  - Parity test passes ≤1e-2 across all 3 runtimes
- **Acceptance criteria**:
  - `uv run python scripts/run_qa_gate_1.py` exits 0 with "RESULT: PASS"
  - All three runtime exports succeed (no FAILED in output)
- **QA verification (Opus)**:
  - Run `uv run python scripts/run_qa_gate_1.py` and verify exit code 0
  - Verify all 3 artifacts exist via `ls -la build/`
  - Run `uv run pytest tests/test_export_gate.py -v` — all tests pass
- **Effort**: 5 minutes after T1.1/T1.2
- **Depends on**: T1.1, T1.2

#### T1.4 — Validate banned-ops absence in ONNX graph (P0)
- **Inputs**: `build/livenix_untrained.onnx` from T1.3
- **Deliverables**:
  - Script `scripts/check_banned_ops.py` that loads ONNX, prints all op types, asserts banned set is absent
  - Banned set: `FFT`, `STFT`, `GridSample`, `GroupNormalization`, `If`, `Loop`, `Scan`
- **Acceptance criteria**:
  - `uv run python scripts/check_banned_ops.py build/livenix_untrained.onnx` exits 0
  - Output lists all op types found
- **QA verification (Opus)**:
  - Run the script, verify exit 0
  - Inspect printed op-types list — confirm only ONNX-clean ops present
- **Effort**: 10 minutes
- **Depends on**: T1.3

---

### Week 2 — Aux heads + losses + data + first training

#### T2.1 — Implement FFT auxiliary head (P0)
- **Inputs**: `src/livenix/models/heads.py`, [PLAN.md architecture table](PLAN.md)
- **Deliverables**:
  - Replace `FFTHead` NotImplementedError stub with working implementation
  - Takes feature-map input (B, C, H, W), applies 2D FFT, returns magnitude spectrum (B, 1, H, W) supervised by BCE against spoof label
  - Used only during training (`if model.training`); never in inference graph
- **Acceptance criteria**:
  - `uv run python -c "from livenix.models.heads import FFTHead; h = FFTHead(in_channels=128); print(h(torch.randn(2,128,4,4)).shape)"` returns `torch.Size([2, 1, ...])`
  - FFTHead is NOT instantiated in `LivenixModel` inference path
- **QA verification (Opus)**:
  - Unit test: instantiate, forward pass, shape match
  - Verify `LivenixModel(...).forward(x)` does NOT call FFTHead (use `model.training = False`)
- **Effort**: 1 hour
- **Depends on**: T1.3

#### T2.2 — Implement SigLIP distillation head (P0)
- **Inputs**: `src/livenix/models/heads.py`
- **Deliverables**:
  - `SigLIPDistillHead`: 1×1 projection from backbone feature dim → SigLIP-So400m dim (1152)
  - Loss in training: MSE between projected backbone features and frozen SigLIP teacher features
  - Teacher model loaded once from HuggingFace `google/siglip-so400m-patch14-384`
  - Teacher run with bf16, no_grad
- **Acceptance criteria**:
  - Forward shape correct
  - Memory under 12GB on RTX 4090 with batch 256 (use bf16 teacher)
- **QA verification (Opus)**:
  - Unit test forward pass
  - Memory profile snippet (mock teacher OK if download slow)
- **Effort**: 2 hours
- **Depends on**: T1.3

#### T2.3 — Implement patch contrastive head (P0)
- **Inputs**: `src/livenix/models/heads.py`
- **Deliverables**:
  - `PatchContrastiveHead`: takes feature map (B, C, H, W) → patches → NT-Xent loss
  - 4×4 patch tokens, temperature τ=0.07
- **Acceptance criteria**:
  - Forward + loss compute works on dummy input
- **QA verification (Opus)**:
  - Unit test
- **Effort**: 1.5 hours
- **Depends on**: T1.3

#### T2.4 — Implement loss functions (P0)
- **Inputs**: `src/livenix/train/__init__.py`, [PLAN.md loss spec](PLAN.md)
- **Deliverables**:
  - `src/livenix/train/losses.py` with:
    - `FocalLoss(gamma=2.0)`
    - `AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)`
    - `NTXentLoss(temperature=0.07)`
    - `CosineMarginLoss(margin=0.3, scale=30)` — applied to main logits before CE
    - `CombinedLoss` that weights main + 0.5·SigLIP + 0.2·FFT + 0.3·patch
- **Acceptance criteria**:
  - Unit tests for each loss: shape + monotonic decrease on overfit
- **QA verification (Opus)**:
  - `uv run pytest tests/test_losses.py -v`
- **Effort**: 2 hours
- **Depends on**: T2.1, T2.2, T2.3

#### T2.5 — Implement CelebA-Spoof dataset loader (P0)
- **Inputs**: CelebA-Spoof README / structure
- **Deliverables**:
  - `src/livenix/data/celeba_spoof.py` with `CelebASpoofDataset(root, split, transforms)`
  - Returns `(image, label, attack_type)` where attack_type ∈ {0=real, 1=print_spoof, 2=replay_spoof}
  - Maps CelebA-Spoof's fine-grained spoof types to our 3-class scheme
  - **Does NOT download data** — assumes user-provided root path
- **Acceptance criteria**:
  - Loader instantiates with a fake directory structure (use tmp_path in test)
  - Mapping table is documented
- **QA verification (Opus)**:
  - `uv run pytest tests/test_celeba_spoof_loader.py -v`
  - Inspect mapping table in code
- **Effort**: 2 hours
- **Depends on**: nothing

#### T2.6 — Implement WMCA dataset loader (P0)
- Same structure as T2.5 for WMCA
- **Effort**: 1.5 hours
- **Depends on**: T2.5 (pattern reuse)

#### T2.7 — Implement HiFiMask dataset loader (P0)
- Same structure for HiFiMask. Label all as spoof but include attack-subtype field for analysis.
- **Effort**: 1.5 hours
- **Depends on**: T2.5

#### T2.8 — Implement deepfake-on-screen dataset stub (P0)
- **Inputs**: planning notes on deepfake training data
- **Deliverables**:
  - `src/livenix/data/deepfake_screen.py` — loads pre-staged directory of frames extracted from DeepFaceLive recordings played on a screen
  - Documentation in code on how to collect these (record DeepFaceLive output, play on phone, capture with another camera)
- **Acceptance criteria**:
  - Loader works on empty directory (returns empty dataset, no crash)
- **QA verification (Opus)**:
  - Inspect loader code + run with empty dir
- **Effort**: 1 hour
- **Depends on**: T2.5

#### T2.9 — Implement ISP-aware augmentation pipeline (P0)
- **Inputs**: [ARCHITECTURE.md preprocessing](ARCHITECTURE.md)
- **Deliverables**:
  - `src/livenix/data/transforms.py` using `albumentations`
  - Pipeline: random JPEG q∈[40,95], color/gamma jitter, simulated demosaic artifacts, brightness, blur, then resize to 128×128, ImageNet normalize
  - Two versions: `train_transforms()` and `eval_transforms()`
- **Acceptance criteria**:
  - Round-trip through pipeline preserves shape (3, 128, 128)
  - Stochastic for train, deterministic for eval
- **QA verification (Opus)**:
  - Unit test on 10 random images
- **Effort**: 1.5 hours
- **Depends on**: nothing

#### T2.10 — Implement identity-disjoint split utility (P0)
- **Inputs**: [PLAN.md QA Gate 12](PLAN.md)
- **Deliverables**:
  - `src/livenix/data/splits.py` with:
    - `make_identity_disjoint_split(samples, train_frac, val_frac, identity_fn)`
    - `audit_split_contamination(train, val, test, facenet_model)` — FaceNet cosine sim
- **Acceptance criteria**:
  - On synthetic data with known IDs, no ID overlap across splits
  - Audit returns 0 contamination pairs above threshold
- **QA verification (Opus)**:
  - `uv run pytest tests/test_splits.py -v`
- **Effort**: 2 hours
- **Depends on**: nothing

#### T2.11 — Implement training loop (P0)
- **Inputs**: all of T2.1–T2.10
- **Deliverables**:
  - `src/livenix/train/trainer.py` with a PyTorch Lightning-style loop
  - YAML-config-driven (consumes `configs/mobile_small.yaml`)
  - Logs to tensorboard
  - Saves checkpoint every 30 min (for RunPod Secure)
- **Acceptance criteria**:
  - Trainer instantiates from config
  - 1 epoch on a tiny dummy dataset completes without crash
- **QA verification (Opus)**:
  - Run 1 epoch on 100-image dummy dataset; check loss curve makes sense
- **Effort**: 3 hours
- **Depends on**: T2.4, T2.5, T2.9, T2.10

#### T2.12 — QA Gate 2: Overfit gate (P0)
- **Inputs**: T2.11 training loop
- **Deliverables**:
  - `scripts/run_qa_gate_2.py`: train on 100 hand-picked CelebA-Spoof images for 100 epochs
  - Must hit ≥99% train accuracy
- **Acceptance criteria**:
  - Train acc ≥99% at epoch 100
  - If fails: model or loss is broken — stop, debug before Gate 4
- **QA verification (Opus)**:
  - Run the script, inspect final accuracy
- **Effort**: 1 hour + ~10 min of CPU training
- **Depends on**: T2.11

#### T2.13 — Acquire CelebA-Spoof + WMCA datasets (manual)
- **Inputs**: dataset request forms
- **Deliverables**: data on disk at agreed paths
- **Acceptance criteria**: `ls <path>` shows expected directory structure
- **QA verification (Opus)**: file-count sanity check
- **Effort**: 1-4 weeks calendar (mostly waiting on EULAs)
- **Depends on**: nothing — runs in parallel
- **Action item for human**: file requests in Week 1

#### T2.14 — First full training run (P0)
- **Inputs**: trained-loop + datasets ready
- **Deliverables**:
  - `runs/v0.1-baseline/checkpoint_final.pth`
  - Tensorboard log
- **Acceptance criteria**:
  - Training completes without OOM
  - Final val loss < initial val loss
  - Checkpoint exports to ONNX successfully
- **QA verification (Opus)**:
  - Run inference on 10 validation samples, check outputs are sensible
- **Effort**: 4-6 hours GPU, ~$8 on RunPod 4090 Secure
- **Depends on**: T2.11, T2.12, T2.13

---

### Week 3 — Convergence + cross-dataset gates

#### T3.1 — Implement metrics module (P0)
- **Inputs**: nothing
- **Deliverables**:
  - `src/livenix/eval/metrics.py`: TPR@FPR, APCER, BPCER, ACER, HTER, ECE
- **Acceptance criteria**: unit tests on known-output cases
- **QA verification (Opus)**: pytest
- **Effort**: 2 hours
- **Depends on**: nothing

#### T3.2 — QA Gate 3: Convergence gate (P0)
- **Deliverables**:
  - `scripts/run_qa_gate_3.py`: load v0.1 checkpoint, compute per-attack confusion matrix, ROC at FPR=1e-3 and 1e-4
  - Output saved to `runs/v0.1-baseline/eval_report.md`
- **Acceptance criteria**: TPR ≥ 96% at FPR=1e-4 on CelebA-Spoof intra-dataset
- **QA verification (Opus)**: read the report, verify numbers
- **Effort**: 2 hours
- **Depends on**: T2.14, T3.1

#### T3.3 — QA Gate 4: Cross-dataset LODO (P0)
- **Deliverables**:
  - `scripts/run_qa_gate_4.py`: train-on-CelebA-Spoof, test-on-WMCA + vice versa
  - BPCER ≤ 15% @ APCER=5%
- **Acceptance criteria**: numbers meet target; if not, document gap and proceed
- **QA verification (Opus)**: read numbers
- **Effort**: 2 hours + GPU
- **Depends on**: T3.2

#### T3.4 — QA Gate 7: OULU-NPU eval (internal) (P0)
- **Deliverables**:
  - `scripts/run_oulu_eval.py`: protocols 1 and 4
  - P4 ACER ≤ 8% target
- **Acceptance criteria**: number reported
- **QA verification (Opus)**: read number
- **Effort**: 2 hours + GPU
- **Depends on**: T3.2

---

### Week 4 — Tuning + fairness + adversarial

#### T4.1 — Hyperparameter tuning iterations (P1)
- **Deliverables**: 3-5 training runs varying aux weights, focal γ, ISP aug strength; pick best
- **Acceptance criteria**: best run improves vs baseline on cross-dataset
- **QA verification (Opus)**: comparison table
- **Effort**: 1-2 days + GPU
- **Depends on**: T3.3

#### T4.2 — QA Gate 5: Fairness eval (P0)
- **Deliverables**:
  - `scripts/run_qa_gate_5.py`: per-demographic BPCER (Fitzpatrick × age × gender)
  - max-cell BPCER ratio ≤ 1.5× target
- **Acceptance criteria**: report generated with per-cell numbers
- **QA verification (Opus)**: read fairness report
- **Effort**: 3 hours
- **Depends on**: T4.1 (uses best model)

#### T4.3 — QA Gate 6: Adversarial robustness (P0)
- **Deliverables**:
  - `scripts/run_qa_gate_6.py`: FGSM ε=4/255, PGD-20 ε=8/255
  - Robust BPCER@APCER=1% reported
- **Acceptance criteria**: numbers reported, even if poor
- **QA verification (Opus)**: read numbers
- **Effort**: 3 hours + GPU
- **Depends on**: T4.1

---

### Week 5 — Quantization + exports

#### T5.1 — INT8 PTQ pipeline (P0)
- **Deliverables**:
  - `src/livenix/export/quantize.py`: PTQ with 500-frame calibration
- **Acceptance criteria**: INT8 accuracy drop < 1% vs FP32
- **QA verification (Opus)**: run comparison, read numbers
- **Effort**: 3 hours
- **Depends on**: T4.1

#### T5.2 — QAT fallback (P1 conditional)
- **Only runs if T5.1 fails the 1% bar**
- **Deliverables**: QAT training loop with LSQ, per-channel weights, 5 epochs
- **Effort**: 4 hours + GPU
- **Depends on**: T5.1

#### T5.3 — Export full pipeline + parity gate (P0)
- **Deliverables**: ONNX/TFLite/CoreML for the best v1.0 model with full parity validation on 1000 inputs
- **Acceptance criteria**: parity ≤ 1e-2 across all runtimes
- **QA verification (Opus)**: run `scripts/run_qa_gate_11.py`
- **Effort**: 2 hours
- **Depends on**: T5.1

#### T5.4 — End-to-end golden test (P0)
- **Deliverables**:
  - `scripts/run_qa_gate_13.py`: full SDK pipeline (load → preprocess → infer) on 100 device-captured reference images, compare with training pipeline outputs
- **Acceptance criteria**: per-image cosine similarity > 0.99
- **QA verification (Opus)**: read report
- **Effort**: 3 hours
- **Depends on**: T5.3

---

### Week 6 — Demos + handoff

#### T6.1 — Webcam Python demo (P0)
- **Deliverables**: `demos/webcam/webcam_demo.py` — opens webcam, draws bbox + isLive/spoofType label live
- **Acceptance criteria**: runs on macOS, ≥10 FPS
- **QA verification (Opus)**: visual inspection or screenshot
- **Effort**: 3 hours
- **Depends on**: T5.3

#### T6.2 — Web demo (ONNX Runtime Web) (P0)
- **Deliverables**: `demos/web/` — static site, browser webcam → ONNX Runtime Web (WebGPU) → label
- **Acceptance criteria**: loads in <1.7s, runs <100ms/frame on M1
- **QA verification (Opus)**: open in browser, check console + timing
- **Effort**: 1 day
- **Depends on**: T5.3

#### T6.3 — THREAT_MODEL.md (P0)
- **Deliverables**: full threat model doc with attack table, achieved KPIs, fairness numbers, adversarial numbers
- **Acceptance criteria**: matches numbers from gates 3-6
- **QA verification (Opus)**: cross-reference numbers
- **Effort**: 2 hours
- **Depends on**: T4.3

#### T6.4 — eval_report.md (P0)
- **Deliverables**: full evaluation report combining outputs of gates 3-7, fairness, adversarial, parity
- **Acceptance criteria**: numbers + ROC plots + confusion matrices
- **QA verification (Opus)**: cross-reference all gate outputs
- **Effort**: 3 hours
- **Depends on**: T5.4

#### T6.5 — Phase 1 handoff package (P0)
- **Deliverables**: tagged release `v0.1.0-phase1` with all artifacts + reports
- **Acceptance criteria**: all P0 tasks above are complete; tag pushed
- **QA verification (Opus)**: run the QA Gate 1 checklist from WEEK_1.md against the final repo state
- **Effort**: 1 hour
- **Depends on**: T6.1, T6.2, T6.3, T6.4

---

### Phase 1.5 (after Phase 1 ships)

#### T7.1 — Train Conv-Large server model (+$5 GPU)
- Same pipeline as T2.14 but with `configs/server_large.yaml`
- **Acceptance**: TPR ≥ 98% @ FPR=1e-4
- **Effort**: 1 day GPU

#### T7.2 — FastAPI server inference
- `src/livenix/server/api.py` with POST `/v1/verify`
- Rate-limiting, telemetry, drift monitor
- **Effort**: 2 days

#### T7.3 — Active liveness via Google ML Kit (Android) / Apple Vision (iOS) / MediaPipe (Web)
- SDK wrapper that runs the platform-native gesture detector after passive scoring
- Server-issued challenge nonces with HMAC
- **Effort**: 3-4 days

#### T7.4 — Anti-injection
- Android: Play Integrity API
- iOS: AppAttest
- Server: frame integrity chain + nonce verification
- **Effort**: 3-5 days

---

## QA verification commands cheat-sheet (for Opus)

Drop-in commands the Verifier can run cold per task:

```bash
# Setup (once per session)
cd /Users/hamdev/livenix
source $HOME/.local/bin/env

# Gate 1 (Week 1)
uv run python scripts/run_qa_gate_1.py             # exit 0 + "RESULT: PASS"
uv run python scripts/check_banned_ops.py build/livenix_untrained.onnx
uv run pytest tests/test_export_gate.py -v

# Gate 2 (Week 2)
uv run python scripts/run_qa_gate_2.py             # train acc ≥ 99% on 100 images

# Gate 3 (Week 3)
uv run python scripts/run_qa_gate_3.py             # TPR ≥ 96% @ FPR=1e-4

# Gate 4
uv run python scripts/run_qa_gate_4.py             # BPCER ≤ 15% @ APCER=5%

# Gate 5
uv run python scripts/run_qa_gate_5.py             # max-cell BPCER ratio ≤ 1.5×

# Gate 6
uv run python scripts/run_qa_gate_6.py             # robust BPCER reported

# Gate 7 (OULU-NPU, internal)
uv run python scripts/run_oulu_eval.py             # P4 ACER ≤ 8%

# Gate 10 (Quantization)
uv run python -m livenix.export.quantize \
  --fp32 runs/v1.0/checkpoint.pth \
  --calib data/calibration_500/ \
  --out build/livenix_int8.tflite
# Then compare INT8 vs FP32 metric drop < 1%

# Gate 11 (Parity)
uv run python scripts/run_parity_test.py --num-samples 1000

# Gate 12 (Contamination audit)
uv run python scripts/run_contamination_audit.py --threshold 0.6

# Gate 13 (Golden image)
uv run python scripts/run_qa_gate_13.py            # per-image cosine sim > 0.99
```

---

## Status tracking

Mark tasks as `pending`, `in_progress`, `done`, `blocked` here as work progresses, or use the TodoWrite list. This table is the authoritative status.

| Task | Status | Branch | Commit |
|---|---|---|---|
| T1.1 — Fix TFLite deps | pending | | |
| T1.2 — Fix CoreML deps | pending | | |
| T1.3 — Re-run Gate 1 full pass | pending | | |
| T1.4 — Banned-ops check | pending | | |
| T2.1 — FFTHead | pending | | |
| T2.2 — SigLIPDistillHead | pending | | |
| T2.3 — PatchContrastiveHead | pending | | |
| T2.4 — Losses | pending | | |
| T2.5 — CelebA-Spoof loader | pending | | |
| T2.6 — WMCA loader | pending | | |
| T2.7 — HiFiMask loader | pending | | |
| T2.8 — Deepfake stub | pending | | |
| T2.9 — ISP transforms | pending | | |
| T2.10 — Splits + contamination | pending | | |
| T2.11 — Trainer | pending | | |
| T2.12 — Gate 2 overfit | pending | | |
| T2.13 — Datasets acquired | pending | | |
| T2.14 — First training run | pending | | |
| T3.1 — Metrics | pending | | |
| T3.2 — Gate 3 convergence | pending | | |
| T3.3 — Gate 4 LODO | pending | | |
| T3.4 — Gate 7 OULU | pending | | |
| T4.1 — Tuning | pending | | |
| T4.2 — Gate 5 fairness | pending | | |
| T4.3 — Gate 6 adversarial | pending | | |
| T5.1 — INT8 PTQ | pending | | |
| T5.2 — QAT fallback | pending | | |
| T5.3 — Full export + parity | pending | | |
| T5.4 — Golden test | pending | | |
| T6.1 — Webcam demo | pending | | |
| T6.2 — Web demo | pending | | |
| T6.3 — THREAT_MODEL.md | pending | | |
| T6.4 — eval_report.md | pending | | |
| T6.5 — Phase 1 handoff | pending | | |
| T7.1 — Conv-Large training | Phase 1.5 | | |
| T7.2 — FastAPI server | Phase 1.5 | | |
| T7.3 — Active liveness | Phase 1.5 | | |
| T7.4 — Anti-injection | Phase 1.5 | | |
