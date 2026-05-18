# Livenix — RunPod Setup Checklist (T2.14)

> Step-by-step provisioning for the first full Phase 1 training run on a
> cloud GPU. Target: **RTX 4090 24GB on RunPod Secure**, ~$8 for one run,
> ~4–6h wall clock. MPS / Mac not supported by the trainer.

---

## Prereqs (on your laptop, before booking the pod)

- [ ] **Datasets acquired** (T2.13). EULA-signed downloads in hand:
  - CelebA-Spoof (~78 GB, click-through license — research only, weights labeled v0.1 prototype per CLAIMS.md)
  - WMCA (signed Idiap EULA from institutional email, ~75 GB preprocessed)
  - HiFiMask (CASIA-SURF agreement, ~75 GB) — *optional for first run*
  - Deepfake-on-screen (~few hundred frames you collect) — *optional*
- [ ] **Git remote set up** (`git remote add origin …` then `git push -u origin develop`) so the pod can clone the repo
- [ ] **RunPod account** with a payment method and credits (≥ $20 buffer)
- [ ] **SSH key registered** in your RunPod account
- [ ] **HuggingFace token** (free) — needed for the SigLIP-So400m teacher download. Save in your password manager or env vars.

---

## 1. Spin up the pod

On runpod.io:

1. **GPU Pods → Deploy**
2. Filter: **RTX 4090 (24GB)**, **Secure Cloud** (NOT Community — Secure won't be preempted)
3. Template: **PyTorch 2.4 (CUDA 12.1, Ubuntu 22.04)** — closest match to our `torch>=2.3` requirement
4. **Storage**: 100 GB container disk + 200 GB network volume (mount at `/workspace/data` — datasets persist across pod restarts, big saving on re-runs)
5. **Expose port**: 22 (SSH), 6006 (TensorBoard)
6. **Environment variables** (set in the pod config UI):
   ```
   HF_HOME=/workspace/.cache/huggingface
   HF_TOKEN=<your token>
   CELEBA_SPOOF_ROOT=/workspace/data/CelebA-Spoof
   WMCA_ROOT=/workspace/data/WMCA
   HIFIMASK_ROOT=/workspace/data/HiFiMask
   DEEPFAKE_SCREEN_ROOT=/workspace/data/deepfake_screen
   ```
7. **Deploy**. ~30s to provision.

---

## 2. SSH in + clone

```bash
ssh root@<pod-public-ip> -p <pod-ssh-port>

# Once in:
cd /workspace
git clone <your remote url> livenix
cd livenix
git checkout develop

# Install uv and sync deps
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra dev --extra train     # --extra train pulls transformers for SigLIP
```

Verify CUDA + tests pass:

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest tests/ -q
```

Expected: `cuda: True NVIDIA GeForce RTX 4090` and `110+ passed`.

---

## 3. Upload datasets

Two options:

**(a) Direct upload from laptop** (slow but simple):
```bash
# from your laptop, replacing pod ssh details:
rsync -avzP -e "ssh -p <port>" \
    /path/to/local/CelebA-Spoof/ \
    root@<ip>:/workspace/data/CelebA-Spoof/
```

**(b) From RunPod network storage** (if you've used it before, dataset is already there): no action needed.

Quick sanity:
```bash
ls /workspace/data/CelebA-Spoof/Data/train/ | wc -l   # should be many subject dirs
ls /workspace/data/WMCA/preprocessed-images/ | head  # should show frames
```

---

## 4. (Optional) Smoke run on a tiny slice

Before paying for the full run, verify the pipeline works end-to-end with a small batch:

```bash
uv run python scripts/train.py \
    --config configs/mobile_small.yaml \
    --output-dir runs/smoke \
    --max-train-samples 500 \
    --no-siglip
```

Expected: `[fit] starting training…` then loss decreasing for a few iterations, then `[fit] done.`. <5 min on the 4090. If it crashes, fix before paying for the full run.

---

## 5. The real training run

In a `tmux` or `screen` session (so SSH disconnects don't kill the run):

```bash
tmux new -s train

# Inside tmux:
cd /workspace/livenix
source $HOME/.local/bin/env

uv run python scripts/train.py \
    --config configs/mobile_small.yaml \
    --output-dir runs/v0.1-baseline \
    --device cuda \
    2>&1 | tee runs/v0.1-baseline.log
```

First run also downloads SigLIP-So400m (~1.6 GB → `/workspace/.cache/huggingface/`). Persists across pod restarts thanks to the network volume.

Detach with `Ctrl-b d`. Reattach later with `tmux attach -t train`.

**Expected wall clock:** 4–6 hours for 25 epochs on combined CelebA + WMCA.

---

## 6. Monitoring

**TensorBoard** (in a separate SSH session):
```bash
cd /workspace/livenix
source $HOME/.local/bin/env
uv run tensorboard --logdir runs/v0.1-baseline --host 0.0.0.0 --port 6006
```

Open `http://<pod-public-ip>:6006` in your browser.

Watch:
- `train/loss_total` should decrease monotonically
- `train/loss_main` is the discriminative signal
- `val/accuracy` should rise above 90% within a few epochs

**Checkpoints** auto-save every 30 min (per trainer config) to `runs/v0.1-baseline/checkpoints/`. Safe to disconnect — checkpoints persist.

---

## 7. Pulling results back

When `[fit] done.` prints:

```bash
# from your laptop:
rsync -avzP -e "ssh -p <port>" \
    root@<ip>:/workspace/livenix/runs/v0.1-baseline/ \
    ./runs/v0.1-baseline/
```

Verify the final checkpoint:
```bash
ls runs/v0.1-baseline/checkpoints/      # *.pth files
ls runs/v0.1-baseline/                  # events.out.tfevents.*, *.log
```

---

## 8. Tear down the pod

**Do NOT skip this** — every hour the pod stays up costs money even if idle.

In RunPod web UI: **My Pods → ⋮ → Stop** (preserves the network volume so datasets stay)
OR **Terminate** (deletes everything including the network volume — only if you don't need re-runs).

Recommendation: **Stop**, not Terminate. Re-spinning a stopped pod takes ~30 seconds and your data is still there for the tuning iterations (Week 3+).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync` fails with TLS error | Reset network in RunPod UI; try again. Pod images sometimes have stale CA bundles. |
| `CUDA out of memory` | Lower `batch_size` in `configs/mobile_small.yaml` (256 → 128). 4090 has 24GB so 256 should fit even with SigLIP teacher in bf16. |
| SigLIP download hangs | Verify `$HF_TOKEN` is set; SigLIP is gated. Or test with `--no-siglip` first. |
| Loss is NaN within first 100 iters | Lower learning rate (`lr: 3.0e-4` → `1.0e-4`) in config. Check for label issues. |
| Tests fail with `tf-keras` import error | Already a known dep issue tracked in deferred TFLite work — does NOT affect training. Re-run with `uv run pytest tests/ -q --ignore=tests/test_export_gate.py`. |
| Pod disconnects mid-training | Reattach `tmux attach -t train` (assuming you used tmux). Latest checkpoint resumes from `runs/v0.1-baseline/checkpoints/`. |

---

## Cost ceiling

| Item | Cost |
|---|---|
| RTX 4090 Secure | ~$0.69/hr |
| One training run (5h) | ~$3.50 |
| Network volume (200 GB, kept stopped) | ~$0.10/day |
| One full run + 2 tuning iterations (Week 3) | ~$10–12 |
| Phase 1 GPU budget total | **~$26** per [PLAN.md](PLAN.md) |

If costs trend higher than budget: pause, share `runs/v0.1-baseline.log` for review, decide whether to continue.
