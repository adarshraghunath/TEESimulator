"""
Label-only TEE-style B-mode simulation using PyMUST/SIMUS (pip install pymust, LGPL-2.1).

Input : one or more 2D fan label maps (H, W), STACOM 11-class, apex top-center, 90 deg fan,
        fan depth = 0.95 * H pixels spanning 90 mm (see examples/input).
Output: per label, in --out_dir: <stem>.npy (float32 in [0, 1]), <stem>.png (simulated image),
        <stem>_label.png (colored label) and <stem>_compare.png (label | simulated image).

The per-class reflectivities below are hand-chosen.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pymust

# Relative scatterer amplitude per STACOM class (0 = unlabelled tissue).
CLASS_AMP = {0: 0.35, 1: 1.0, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03,
             6: 0.03, 7: 0.03, 8: 0.03, 9: 0.3, 10: 0.03}
BLOOD = (2, 3, 4, 5, 6, 7, 8, 10)
WALL_AMP = 1.0
WALL_PX = 3


def make_scatterers(label, px_m, density_per_mm2, rng):
    H, W = label.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs + 0.5 - W / 2
    dy = ys + 0.5
    r_max = 0.95 * H
    inside = (np.abs(np.arctan2(dx, dy)) <= np.deg2rad(45)) & (np.hypot(dx, dy) <= r_max)

    amp = np.vectorize(CLASS_AMP.get)(label).astype(np.float32)
    blood = np.isin(label, BLOOD)
    wall = binary_dilation(blood, iterations=WALL_PX) & ~blood & (label != 1)
    amp[wall] = WALL_AMP

    n_per_px = density_per_mm2 * (px_m * 1e3) ** 2
    counts = rng.poisson(n_per_px, size=label.shape) * inside
    iy, ix = np.nonzero(counts)
    rep = counts[iy, ix]
    iy, ix = np.repeat(iy, rep), np.repeat(ix, rep)
    x = (ix + rng.random(ix.size) - W / 2) * px_m
    z = (iy + rng.random(iy.size)) * px_m
    rc = amp[iy, ix] * rng.standard_normal(ix.size)
    return x, z, rc, inside


def simulate(label, depth_mm=90.0, density=6.0, n_tx=21, seed=0, fc=None):
    rng = np.random.default_rng(seed)
    H, W = label.shape
    px_m = depth_mm * 1e-3 / (0.95 * H)
    x, z, rc, inside = make_scatterers(label, px_m, density, rng)
    print(f"{x.size} scatterers", flush=True)

    param = pymust.getparam("P4-2v")
    if fc:
        param.fc = fc
    param.fs = 4 * param.fc
    param.attenuation = 0.5

    tilts = np.deg2rad(np.linspace(-40, 40, n_tx))
    width = np.deg2rad(2 * 80 / max(n_tx - 1, 1))

    xi = (np.arange(W)[None, :] + 0.5 - W / 2) * px_m * np.ones((H, 1))
    zi = (np.arange(H)[:, None] + 0.5) * px_m * np.ones((1, W))

    acc = np.zeros((H, W), dtype=np.complex128)
    for k, t in enumerate(tilts):
        t0 = time.time()
        delays = pymust.txdelay(param, t, width)
        param.TXdelay = delays
        rf, _ = pymust.simus(x, z, rc, delays, param)
        rf, _ = pymust.tgc(rf)
        iq = pymust.rf2iq(rf, param.fs, param.fc)
        m = pymust.dasmtx(iq, xi, zi, param)
        acc += (m @ iq.flatten(order="F")).reshape(xi.shape, order="F")
        print(f"tx {k + 1}/{n_tx} {time.time() - t0:.1f}s", flush=True)

    img = pymust.bmode(acc, 40).astype(np.float32)
    img = np.clip((img - img.min()) / (np.ptp(img) + 1e-9), 0, 1)
    img[~inside] = 0
    return img


LABEL_COLORS = ["black", "peru", "orange", "royalblue", "gold", "cornflowerblue",
                "brown", "cyan", "magenta", "yellow", "green"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_tx", type=int, default=21)
    ap.add_argument("--density", type=float, default=4.0, help="scatterers per mm^2")
    ap.add_argument("--fc", type=float, default=None, help="override center frequency (Hz)")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(LABEL_COLORS)
    for path in a.labels:
        print(f"== {path.stem}", flush=True)
        label = np.load(path)
        img = simulate(label, density=a.density, n_tx=a.n_tx, fc=a.fc)
        stem = a.out_dir / path.stem
        np.save(f"{stem}.npy", img)
        plt.imsave(f"{stem}.png", img, cmap="gray", vmin=0, vmax=1)
        plt.imsave(f"{stem}_label.png", label, cmap=cmap, vmin=-0.5, vmax=10.5)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(label, cmap=cmap, vmin=-0.5, vmax=10.5, interpolation="nearest")
        ax[0].set_title("input label")
        ax[1].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[1].set_title("PyMUST SIMUS (label-only)")
        for x in ax:
            x.axis("off")
        fig.tight_layout()
        fig.savefig(f"{stem}_compare.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
