"""
Check the geometry/label part of the simulator against the bundled expected output.

Needs only the STACOM segmentations (no CT): the segmentation file is passed as its own
"CT", which is enough for probe pose, ray casting and label scan conversion. It does NOT
check B-mode images, which depend on the CT.

Pass criteria per patient:
  - laa_pixels_in_fan and laa_voxels_in_volume equal the expected manifest values
  - label map equals the expected one except on the 45-degree fan edge (|x - W/2| == y),
    where a float tie in scan_convert_v2's `abs(theta) <= half` test is version dependent
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tee_simulator import TEEConfig  # noqa: E402
from scan_convert_v2 import simulate_labels_only  # noqa: E402


def find_seg(seg_dir, pid):
    hits = list(seg_dir.rglob(f"{pid}.img.nii.gz"))
    if not hits:
        raise FileNotFoundError(f"{pid}.img.nii.gz not found under {seg_dir}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg_dir", type=Path, required=True, help="folder with STACOM {pid}.img.nii.gz")
    ap.add_argument("--expected", type=Path, default=ROOT / "examples" / "expected_output")
    a = ap.parse_args()

    cfg = TEEConfig(fan_angle_deg=90.0)
    cfg.center_freq_mhz = 5.0

    manifest = pd.read_csv(a.expected / "manifest.csv", dtype={"pid_padded": str})
    failed = 0
    for _, row in manifest.iterrows():
        seg_path = find_seg(a.seg_dir, int(row.pid_raw))
        lab, meta = simulate_labels_only(str(seg_path), str(seg_path), cfg=cfg)
        ref = np.load(a.expected / "labels" / f"{row.pid_padded}.npy")

        ys, xs = np.nonzero(ref != lab)
        off_edge = int((np.abs(xs - ref.shape[1] // 2) != ys).sum())
        counts_ok = (meta["laa_pixels_in_fan"] == row.laa_pixels_in_fan
                     and meta["laa_voxels_in_volume"] == row.laa_voxels_in_volume)
        ok = counts_ok and off_edge == 0
        failed += not ok
        print(f"{row.pid_padded}  counts_match={counts_ok}  fan_edge_px={len(ys)}  "
              f"other_mismatch_px={off_edge}  {'PASS' if ok else 'FAIL'}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
