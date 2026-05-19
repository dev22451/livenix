"""Audit a face anti-spoofing dataset for size, attack mix, and demographics.

Works on:
  (a) The pre-cropped Kaggle CelebA-Spoof variant we trained on:
        --root /workspace/data/CelebA-Spoof   (or local path)
        --variant cropped
  (b) The full CelebA-Spoof distribution (74 zip parts → extracted):
        --root /path/to/CelebA-Spoof-full     (containing Data/train, Data/test)
        --variant full
  (c) Any directory tree of {live, spoof} subdirs — generic mode:
        --variant generic

Reports:
  - Total samples, real vs spoof breakdown
  - For 'full' variant: per-attack-type counts, identity counts, per-split sizes
  - For 'cropped'/'generic': basic counts only
  - Skin-tone heuristic: face crop mean L channel histogram (rough proxy)
  - Saves N_SAMPLE random thumbnails to <out>/eyeball/ for manual review
    (separated by class so you can scroll through and sanity-check)

This is intentionally NOT a fairness audit by itself. The 'eyeball/' folder
is the actual deliverable — open it in Finder and look at the distribution.
The L channel histogram is a *hint*, not a verdict.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


# CelebA-Spoof spoof type IDs → human name (from official paper)
SPOOF_TYPE_NAMES = {
    0:  "Live",
    1:  "Photo",
    2:  "Poster",
    3:  "A4",
    4:  "FaceMask (paper)",
    5:  "UpperBodyMask",
    6:  "RegionMask",
    7:  "PC (monitor)",
    8:  "Pad (tablet)",
    9:  "Phone",
    10: "3D Mask",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Dataset root")
    p.add_argument("--variant", choices=["cropped", "full", "generic"], default="cropped")
    p.add_argument("--out", default="runs/dataset-audit",
                   help="Where to write the report and eyeball thumbnails")
    p.add_argument("--sample", type=int, default=200,
                   help="Number of random images to save for manual review")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_spoof_type(sidecar: Path) -> int | None:
    try:
        lines = [ln.strip() for ln in sidecar.read_text().splitlines() if ln.strip()]
        return int(lines[-1])
    except Exception:
        return None


def audit_full(root: Path) -> dict:
    """Parse CelebA-Spoof's metas/intra_test/*_label.txt files.

    Each line: ``Data/<split>/<subj>/<live|spoof>/<file>.<ext> <0_or_1>``
    Binary labels only (0=live, 1=spoof). Fine-grained spoof types live
    in protocol1/2 — out of scope for this audit; we don't need them to
    eyeball skin-tone distribution.
    """
    metas = root / "metas" / "intra_test"
    if not metas.is_dir():
        raise FileNotFoundError(f"Expected {metas} (CelebA-Spoof intra_test labels)")

    report: dict = {
        "variant": "full",
        "splits": {},
        "identities_total": 0,
        "attack_type_counts": Counter(),
        "image_paths": [],
    }

    identity_ids: set[str] = set()
    for label_file in sorted(metas.glob("*_label.txt")):
        split_name = label_file.stem.replace("_label", "")  # "train" or "test"
        split_info = {
            "subjects": set(),
            "live_samples": 0,
            "spoof_samples": 0,
            "missing_on_disk": 0,
        }
        with label_file.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rel_path, label_str = parts[0], parts[-1]
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                # rel_path like Data/train/2623/live/000000.jpg
                segs = rel_path.split("/")
                if len(segs) >= 3:
                    subj_id = f"{segs[1]}/{segs[2]}"
                    split_info["subjects"].add(subj_id)
                    identity_ids.add(subj_id)
                if label == 0:
                    split_info["live_samples"] += 1
                else:
                    split_info["spoof_samples"] += 1
                report["attack_type_counts"][label] += 1
                report["image_paths"].append((root / rel_path, label, split_name))

        split_info["subjects"] = len(split_info["subjects"])
        report["splits"][split_name] = split_info

    report["identities_total"] = len(identity_ids)
    return report


def audit_cropped(root: Path) -> dict:
    """Cropped Kaggle variant: root/CelebA_Spoof/<split>/live|spoof/*.jpg (no sidecar)."""
    base_candidates = [root / "CelebA_Spoof", root]
    base = next((c for c in base_candidates if (c / "test").is_dir() or (c / "train").is_dir()), None)
    if base is None:
        raise FileNotFoundError(f"Could not find live/spoof split dirs under {root}")

    report: dict = {
        "variant": "cropped",
        "splits": {},
        "image_paths": [],
    }
    for split_dir in sorted(base.iterdir()):
        if not split_dir.is_dir():
            continue
        info = {"live_samples": 0, "spoof_samples": 0}
        for label, sub in [(0, "live"), (1, "spoof")]:
            d = split_dir / sub
            if not d.is_dir():
                continue
            for p in d.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    if label == 0:
                        info["live_samples"] += 1
                    else:
                        info["spoof_samples"] += 1
                    report["image_paths"].append((p, label, split_dir.name))
        report["splits"][split_dir.name] = info
    return report


def audit_generic(root: Path) -> dict:
    """Generic: any dir with {live, spoof} subdirs."""
    report = {"variant": "generic", "image_paths": [], "splits": {"_": {}}}
    info = {"live_samples": 0, "spoof_samples": 0}
    for label, sub in [(0, "live"), (1, "spoof")]:
        d = root / sub
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                if label == 0:
                    info["live_samples"] += 1
                else:
                    info["spoof_samples"] += 1
                report["image_paths"].append((p, label, "_"))
    report["splits"]["_"] = info
    return report


def brightness_histogram(samples: list[tuple[Path, int, str]], n: int = 500) -> dict:
    """Rough proxy for skin-tone distribution: mean L channel of a center crop.

    NOT a real skin tone classifier — just enough signal to spot if the
    distribution is heavily lighter-skewed.
    """
    if not samples:
        return {}
    picks = random.sample(samples, min(n, len(samples)))
    L_live: list[float] = []
    L_spoof: list[float] = []
    for img_path, label, _ in picks:
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")  # luminance
                # center 60% crop to avoid background dominating
                w, h = im.size
                cx, cy = w // 2, h // 2
                hw, hh = int(w * 0.3), int(h * 0.3)
                box = (cx - hw, cy - hh, cx + hw, cy + hh)
                arr = np.array(im.crop(box))
                m = float(arr.mean())
            if label == 0:
                L_live.append(m)
            else:
                L_spoof.append(m)
        except (UnidentifiedImageError, OSError):
            continue

    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    return {"live_L": stats(L_live), "spoof_L": stats(L_spoof)}


def save_eyeball_thumbs(samples, out_dir: Path, n_per_class: int, variant: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict = {}
    for img_path, label, split in samples:
        if variant == "full":
            key = SPOOF_TYPE_NAMES.get(label, f"type_{label}")
        else:
            key = "live" if label == 0 else "spoof"
        by_class.setdefault(key, []).append((img_path, split))

    for cls, items in by_class.items():
        random.shuffle(items)
        cls_dir = out_dir / cls.replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)
        for i, (img_path, split) in enumerate(items[:n_per_class]):
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((256, 256))
                    im.save(cls_dir / f"{i:03d}_{split}_{img_path.name}")
            except (UnidentifiedImageError, OSError):
                continue


def format_report(report: dict, brightness: dict, out_root: Path) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"Dataset audit — variant={report['variant']}")
    lines.append("=" * 70)
    lines.append("")
    total_imgs = len(report["image_paths"])
    lines.append(f"Total images discovered: {total_imgs:,}")

    if report["variant"] == "full":
        lines.append(f"Unique identities (subject dirs): {report['identities_total']:,}")
        lines.append("")
        lines.append("Per-split breakdown:")
        for split, info in report["splits"].items():
            lines.append(f"  {split:>10s}: subjects={info['subjects']:,}  "
                         f"live={info['live_samples']:,}  spoof={info['spoof_samples']:,}")
        lines.append("")
        lines.append("Binary label breakdown (intra_test protocol):")
        for t in sorted(report["attack_type_counts"]):
            name = "Live" if t == 0 else "Spoof"
            n = report["attack_type_counts"][t]
            pct = 100.0 * n / max(total_imgs, 1)
            lines.append(f"  {t}: {name:<10s} {n:>8,}  ({pct:5.1f}%)")
        lines.append("")
        lines.append("Note: fine-grained spoof types (Photo/Poster/Phone/PC/etc.) live in")
        lines.append("metas/protocol1 and protocol2 — out of scope for this audit.")
    else:
        lines.append("")
        lines.append("Per-split breakdown:")
        for split, info in report["splits"].items():
            lines.append(f"  {split:>10s}: live={info.get('live_samples', 0):,}  "
                         f"spoof={info.get('spoof_samples', 0):,}")

    if brightness:
        lines.append("")
        lines.append("Brightness (L channel) histogram — rough skin-tone proxy:")
        lines.append("  Lower mean L = darker faces on average. Compare live vs spoof.")
        lines.append("  CelebA-Spoof typically shows live_L mean in the 130-160 range.")
        lines.append("  Production target population (e.g. South Asian users)")
        lines.append("  would typically have lower L values (110-140).")
        lines.append("")
        for cls, s in brightness.items():
            if s.get("n", 0):
                lines.append(f"  {cls}: n={s['n']}  mean={s['mean']:.1f}  "
                             f"p10={s['p10']:.0f}  p50={s['p50']:.0f}  p90={s['p90']:.0f}")

    lines.append("")
    lines.append("Manual review:")
    lines.append(f"  Open {out_root}/eyeball/  in Finder.")
    lines.append("  Scroll through each class folder. Note: are the faces dominantly light-skinned?")
    lines.append("  East Asian? Caucasian? Any brown / dark skin? Children? Glasses? Beards?")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[audit] root={root}  variant={args.variant}")
    if args.variant == "full":
        report = audit_full(root)
    elif args.variant == "cropped":
        report = audit_cropped(root)
    else:
        report = audit_generic(root)

    print(f"[audit] discovered {len(report['image_paths']):,} images")
    print(f"[audit] computing brightness histogram on 500 random samples…")
    brightness = brightness_histogram(report["image_paths"], n=500)

    eyeball_dir = out / "eyeball"
    if eyeball_dir.exists():
        shutil.rmtree(eyeball_dir)
    n_per_class = max(20, args.sample // max(len(set(p[1] for p in report["image_paths"])), 1))
    print(f"[audit] saving {n_per_class} thumbnails per class to {eyeball_dir}/")
    save_eyeball_thumbs(report["image_paths"], eyeball_dir, n_per_class, args.variant)

    text = format_report(report, brightness, out)
    print()
    print(text)
    (out / "report.txt").write_text(text)
    print(f"[audit] full report written to {out}/report.txt")


if __name__ == "__main__":
    main()
