# Livenix

Passive RGB face liveness SDK — Phase 1 research prototype.

**Status:** v0.1 scaffold (Week 1). Not for commercial deployment.

## Documentation

- [PLAN.md](PLAN.md) — Phase 1 strategy, KPIs, decisions log
- [ARCHITECTURE.md](ARCHITECTURE.md) — End-to-end training, build, inference flow
- [CLAIMS.md](CLAIMS.md) — Allowed/disallowed marketing claims
- [WEEK_1.md](WEEK_1.md) — Scaffold + export gate handoff

## Phase 1 scope (locked)

Defends: paper print, screen replay (video call, recorded clip, camera-call replay), paper cutouts, deepfake-on-screen.
Does NOT defend (Phase 1): silicone masks, virtual-camera injection, multi-frame attacks.

## Quick start (Week 1)

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install deps
uv sync --extra dev

# Run QA Gate 1 (verifies architecture exports cleanly, untrained)
uv run python scripts/run_qa_gate_1.py
```

## Repo layout

See [ARCHITECTURE.md §6](ARCHITECTURE.md#6-repo-layout-to-be-scaffolded-week-1).

## License

Proprietary. See [LICENSE](LICENSE).
