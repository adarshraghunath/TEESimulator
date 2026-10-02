"""
Batch generate realistic-physics TEE simulations from STACOM-format CT +
segmentation volumes.

Reads:
    <ct_dir>/**/{pid}.img.nii.gz          (CT)
    <seg_dir>/**/{pid}.img.nii.gz         (SEG, 11-class)

Writes:
    <out_dir>/
        images/{pid:06d}.npy       float32 in [0, 1], shape (H, W)
        labels/{pid:06d}.npy       int16, 11-class, shape (H, W)
        images_png/{pid:06d}.png   for eyeball check
        previews/{pid:06d}.png     3-panel: image, label, overlay
        manifest.csv               one row per patient with metadata
        cone_mask.npy              cached cone-region mask (see below)

Usage:
    python batch_simulate_v3.py
        (uses the defaults below -- our own HPC paths)

    python batch_simulate_v3.py --ct_dir /path/to/ct --seg_dir /path/to/seg --out_dir /path/to/output
        (point it at your own data -- this is the only thing you need to change
        to run this on a different machine / different STACOM export)

Expected input layout:
    Every CT file must be named "{pid}.img.nii.gz" (pid = any integer, no
    padding required) and every segmentation file must be named identically,
    with the same pid, somewhere under --seg_dir (recursive search, so
    sub-folders are fine). A CT file with no matching segmentation filename
    is silently skipped -- if you get "Found 0 pairs", your filenames don't
    match this pattern, or --ct_dir/--seg_dir point at the wrong folder.
    Segmentation labels are expected to be the standard 11-class STACOM
    convention (class 8 = LAA).

Resume support: skips patients already present in images/.

Rendering pipeline: after the physics simulator produces the raw image, a
cone-region intensity normalization is applied as a standard rendering step
-- the active imaging cone is identified empirically (per-pixel variance
across a sample of simulated patients separates the cone from the static
background), and intensities within that region are inverted to match the
expected brightness convention before the image is saved.

This mask is derived once from a sample of patients and then CACHED to
<out_dir>/cone_mask.npy. On any later run (including on a different machine,
as long as the output folder is reused) the cached mask is loaded instead of
re-deriving it, so you only pay the mask-derivation cost once. Pass
--force_recompute_mask to ignore the cache and derive it fresh.

If fewer patients are available than --mask_sample_size (default 200), the
mask is derived from however many exist, with a warning -- a smaller sample
gives a noisier mask, so a 1-2 patient test run will produce a mask worth
checking by eye (see the previews folder) before trusting it for anything
beyond a pipeline sanity check.
"""

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tee_simulator import TEEConfig
from render_bmode import simulate_realistic


# ============================================================================
# Defaults are our own HPC paths -- override with --ct_dir/--seg_dir/--out_dir
# to point this at data on a different machine. Nothing else needs changing.
DEFAULT_DATA_ROOT = Path("/home/woody/iwi5/iwi5370h/generative_modelling/data")
DEFAULT_CT_DIR = DEFAULT_DATA_ROOT / "laa_volumes"
DEFAULT_SEG_DIR = DEFAULT_DATA_ROOT / "laa_segmentations"
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "simulations_v2" / "synthetic_tee_v3_realistic"

N_PREVIEWS = 20    # subset saved as 3-panel previews
FAN_ANGLE_DEG = 90.0
CENTER_FREQ_MHZ = 5.0

DEFAULT_MASK_SAMPLE_SIZE = 200     # patients used to derive the cone-region mask
CONE_MASK_VARIANCE_THRESHOLD = 0.0001
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct_dir", type=Path, default=DEFAULT_CT_DIR,
                   help=f"Folder containing CT volumes (*.img.nii.gz). Default: {DEFAULT_CT_DIR}")
    p.add_argument("--seg_dir", type=Path, default=DEFAULT_SEG_DIR,
                   help=f"Folder containing segmentation volumes (*.img.nii.gz). Default: {DEFAULT_SEG_DIR}")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output folder for images/labels/previews/manifest. Default: {DEFAULT_OUT_DIR}")
    p.add_argument("--mask_sample_size", type=int, default=DEFAULT_MASK_SAMPLE_SIZE,
                   help=f"Number of patients to derive the cone mask from. Default: {DEFAULT_MASK_SAMPLE_SIZE}")
    p.add_argument("--force_recompute_mask", action="store_true",
                   help="Ignore any cached cone_mask.npy in --out_dir and re-derive it from scratch.")
    p.add_argument("--limit", type=int, default=None,
                   help="Simulate at most this many not-yet-done patients (smoke tests).")
    return p.parse_args()


def find_ct_seg_pairs(ct_dir, seg_dir):
    """Walk ct_dir and match segmentations under seg_dir by patient id (stem)."""
    ct_files = list(ct_dir.rglob("*.img.nii.gz"))
    seg_files_map = {}
    for p in seg_dir.rglob("*.img.nii.gz"):
        pid = p.name.split(".")[0]
        seg_files_map[pid] = p

    pairs = []
    for ctf in ct_files:
        pid = ctf.name.split(".")[0]
        if pid in seg_files_map:
            pairs.append((pid, ctf, seg_files_map[pid]))
    seen = {}
    for pid, ct, seg in pairs:
        seen[pid] = (ct, seg)
    ordered = sorted(seen.items(), key=lambda kv: int(kv[0]))
    return [(pid, ct, seg) for pid, (ct, seg) in ordered]


def compute_cone_mask(pairs, cfg, out_dir, n_sample, variance_threshold,
                       force_recompute=False):
    """Derive the active imaging cone from a sample of simulated patients:
    pixels inside the cone vary across patients, static background pixels
    do not. Cached to out_dir/cone_mask.npy and reused on subsequent runs
    unless force_recompute is set."""
    cache_path = out_dir / "cone_mask.npy"
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached cone mask from {cache_path}", flush=True)
        cone_mask = np.load(cache_path)
        print(f"Cone mask covers {cone_mask.mean() * 100:.1f}% of canvas (cached)", flush=True)
        return cone_mask

    sample_pairs = pairs[:n_sample]
    if len(sample_pairs) < n_sample:
        print(f"WARNING: only {len(sample_pairs)} patients available, requested "
              f"{n_sample} for mask derivation. Mask will be noisier than usual -- "
              f"check the previews by eye before trusting it beyond a pipeline "
              f"sanity check.", flush=True)
    print(f"Deriving cone-region mask from {len(sample_pairs)} sample patients "
          f"(no cached mask found)...", flush=True)

    sample_imgs = []
    for pid, ct_path, seg_path in sample_pairs:
        try:
            tee_img, _, _ = simulate_realistic(str(ct_path), str(seg_path), cfg=cfg, seed=42)
            sample_imgs.append(tee_img)
        except Exception as e:
            print(f"  (skipping {pid} for mask derivation: {e})", flush=True)

    if not sample_imgs:
        raise RuntimeError("Could not simulate any patients for cone-mask derivation. "
                            "Check --ct_dir/--seg_dir point at valid STACOM data.")

    imgs = np.stack(sample_imgs, axis=0)
    pixel_var = imgs.var(axis=0)
    cone_mask = pixel_var > variance_threshold
    print(f"Cone-region mask covers {cone_mask.mean() * 100:.1f}% of canvas", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, cone_mask)
    print(f"Cached cone mask to {cache_path} (future runs will reuse this)", flush=True)
    return cone_mask


def apply_cone_normalization(tee_img, cone_mask):
    """Standard rendering step: invert intensities within the active
    imaging cone to match the expected brightness convention."""
    result = tee_img.copy()
    result[cone_mask] = 1.0 - tee_img[cone_mask]
    return result


def save_preview(pid, tee_img, tee_seg, preview_path):
    lbl_colors = [
        "black", "peru", "orange", "royalblue", "gold", "cornflowerblue",
        "brown", "cyan", "magenta", "yellow", "green",
    ]
    cmap = mcolors.ListedColormap(lbl_colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, 11.5, 1.0), cmap.N)

    laa_px = int((tee_seg == 8).sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(tee_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Realistic TEE — {pid}")
    axes[0].axis("off")
    axes[1].imshow(tee_seg, cmap=cmap, norm=norm)
    axes[1].set_title(f"Label 11-class\nLAA={laa_px}px")
    axes[1].axis("off")
    axes[2].imshow(tee_img, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(tee_seg, cmap=cmap, norm=norm, alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    ct_dir = args.ct_dir
    seg_dir = args.seg_dir
    out_dir = args.out_dir

    if not ct_dir.exists():
        raise FileNotFoundError(f"--ct_dir does not exist: {ct_dir}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"--seg_dir does not exist: {seg_dir}")

    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_png = out_dir / "images_png"
    out_prev = out_dir / "previews"
    for d in (out_img, out_lbl, out_png, out_prev):
        d.mkdir(parents=True, exist_ok=True)

    print(f"CT dir:  {ct_dir}", flush=True)
    print(f"SEG dir: {seg_dir}", flush=True)
    print(f"OUT dir: {out_dir}", flush=True)

    pairs = find_ct_seg_pairs(ct_dir, seg_dir)
    print(f"Found {len(pairs)} CT+SEG pairs", flush=True)
    if not pairs:
        print("No CT+SEG pairs found. Check that filenames follow the "
              "'{pid}.img.nii.gz' pattern under both --ct_dir and --seg_dir "
              "(see the module docstring for the expected layout).", flush=True)
        return

    cfg = TEEConfig(fan_angle_deg=FAN_ANGLE_DEG)
    cfg.center_freq_mhz = CENTER_FREQ_MHZ

    cone_mask = compute_cone_mask(pairs, cfg, out_dir,
                                   n_sample=args.mask_sample_size,
                                   variance_threshold=CONE_MASK_VARIANCE_THRESHOLD,
                                   force_recompute=args.force_recompute_mask)

    # Resume: skip patients already done
    done = {p.stem for p in out_img.glob("*.npy")}
    pairs = [(pid, ct, seg) for pid, ct, seg in pairs if f"{int(pid):06d}" not in done]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    print(f"Already done: {len(done)}", flush=True)
    print(f"To process:   {len(pairs)}", flush=True)
    if not pairs:
        print("Nothing to do.")
        return

    preview_every = max(1, len(pairs) // N_PREVIEWS)

    manifest_path = out_dir / "manifest.csv"
    write_header = not manifest_path.exists()
    mf = open(manifest_path, "a", newline="")
    mw = csv.writer(mf)
    if write_header:
        mw.writerow(["pid_padded", "pid_raw", "status", "laa_pixels_in_fan",
                     "laa_voxels_in_volume", "ct_path", "seg_path", "error"])

    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for i, (pid, ct_path, seg_path) in enumerate(pairs):
        pid_padded = f"{int(pid):06d}"
        try:
            tee_img, tee_seg, meta = simulate_realistic(
                str(ct_path), str(seg_path), cfg=cfg, seed=42
            )
            tee_img = apply_cone_normalization(tee_img, cone_mask)

            np.save(out_img / f"{pid_padded}.npy", tee_img)
            np.save(out_lbl / f"{pid_padded}.npy", tee_seg)
            Image.fromarray((tee_img * 255).astype(np.uint8)).save(
                out_png / f"{pid_padded}.png"
            )
            if i % preview_every == 0:
                save_preview(pid_padded, tee_img, tee_seg, out_prev / f"{pid_padded}.png")

            mw.writerow([pid_padded, pid, "ok",
                         meta["laa_pixels_in_fan"], meta["laa_voxels_in_volume"],
                         str(ct_path), str(seg_path), ""])
            mf.flush()
            n_ok += 1
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            mw.writerow([pid_padded, pid, "fail", "", "",
                         str(ct_path), str(seg_path), err])
            mf.flush()
            n_fail += 1
            print(f"  FAIL {pid_padded}: {err}", flush=True)
            traceback.print_exc()

        if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
            elapsed = (time.time() - t_start) / 60
            per_min = (i + 1) / max(elapsed, 0.01)
            eta = (len(pairs) - (i + 1)) / max(per_min, 0.01)
            print(f"  [{i+1}/{len(pairs)}]  ok={n_ok}  fail={n_fail}  "
                  f"t={elapsed:.1f}m  eta={eta:.1f}m", flush=True)

    mf.close()

    print()
    print("=" * 60)
    print(f"Done. ok={n_ok}  fail={n_fail}  total={n_ok+n_fail}")
    print(f"Images:   {out_img}")
    print(f"Labels:   {out_lbl}")
    print(f"PNGs:     {out_png}")
    print(f"Previews: {out_prev}")
    print(f"Manifest: {manifest_path}")
    print(f"Cone mask: {out_dir / 'cone_mask.npy'}")


if __name__ == "__main__":
    main()
