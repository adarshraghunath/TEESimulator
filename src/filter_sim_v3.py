"""
Filter the new realistic-simulator patient set using the DATASET'S OWN
curated list of "full" LAA segmentations, not a pixel-count heuristic.

This matches exactly what Phase 2b used (685 patients), because that number
comes from ImageCAS-STACOM's info/all_full_laa_segmentations_id.txt file --
not from any filter we applied ourselves.

Reads:
    data/laa_segmentations/ImageCAS-STACOM2025-02-10-2025/info/
        all_full_laa_segmentations_id.txt      (685 curated patient IDs)
    data/simulations_v2/synthetic_tee_v3_realistic/labels/{pid:06d}.npy

Writes:
    data/simulations_v2/synthetic_tee_v3_realistic/patient_list.txt
    data/simulations_v2/synthetic_tee_v3_realistic/stats.csv
"""

from pathlib import Path

import numpy as np


DATA_ROOT = Path("/home/woody/iwi5/iwi5370h/generative_modelling/data")
ID_LIST_PATH = (DATA_ROOT / "laa_segmentations" /
                "ImageCAS-STACOM2025-02-10-2025" / "info" /
                "all_full_laa_segmentations_id.txt")

LBL_DIR = DATA_ROOT / "simulations_v2" / "synthetic_tee_v3_realistic" / "labels"
OUT_ROOT = LBL_DIR.parent

LAA_CLASS = 8


def main():
    # Parse the curated ID list. Format is like "100.img" per line.
    raw_ids = [line.strip() for line in ID_LIST_PATH.read_text().splitlines() if line.strip()]
    pids = []
    for r in raw_ids:
        try:
            pid = int(r.split(".")[0])
            pids.append(pid)
        except ValueError:
            continue
    print(f"Curated 'full LAA' ID list: {len(pids)} patients", flush=True)

    kept = []
    missing = []
    laa_px_list = []

    for pid in sorted(pids):
        pid_padded = f"{pid:06d}"
        lbl_path = LBL_DIR / f"{pid_padded}.npy"
        if not lbl_path.exists():
            missing.append(pid_padded)
            continue
        lbl = np.load(lbl_path)
        laa_px = int((lbl == LAA_CLASS).sum())
        laa_px_list.append(laa_px)
        kept.append(pid_padded)

    (OUT_ROOT / "patient_list.txt").write_text("\n".join(kept) + "\n")

    print(f"Found in new simulator output: {len(kept)}", flush=True)
    print(f"Missing from new simulator output: {len(missing)}", flush=True)
    if missing:
        print(f"  First few missing: {missing[:10]}", flush=True)
    if laa_px_list:
        print(f"LAA px stats: min={min(laa_px_list)}  "
              f"median={int(np.median(laa_px_list))}  "
              f"mean={np.mean(laa_px_list):.0f}  max={max(laa_px_list)}", flush=True)

    print(f"\nOutput: {OUT_ROOT / 'patient_list.txt'}")


if __name__ == "__main__":
    main()
