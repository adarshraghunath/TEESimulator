"""
TEE Simulator — wrapped as a single function for batch use.

Usage in a notebook:
    from tee_simulator import simulate_from_paths, TEEConfig
    image, seg_img, meta = simulate_from_paths("ct.nii.gz", "seg.nii.gz")
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TEEConfig:
    # Cone geometry
    fan_angle_deg: float = 60.0
    n_rays: int = 192
    depth_mm: float = 90.0
    n_depth_samples: int = 480

    # Probe placement
    probe_offset_mm: float = 25.0

    # HU -> echo mapping
    hu_clip: Tuple[float, float] = (-1000.0, 1500.0)

    # Physics-lite knobs (tuned to let bulk tissue through)
    attenuation_db_per_cm: float = 0.3
    reflection_gain: float = 2.0
    bulk_scatter: float = 1.0
    wall_echo_strength: float = 0.8
    wall_blur_depth: float = 1.0
    wall_blur_ray: float = 0.8
    speckle_strength: float = 0.5
    speckle_grain_px: float = 1.2
    shadow_strength: float = 0.7
    shadow_threshold_hu: float = 400.0

    # Output
    out_size: Tuple[int, int] = (256, 256)
    gamma: float = 0.3

    # Reproducibility
    seed: Optional[int] = 42


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_case(ct_path: str, seg_path: str):
    """Load CT + segmentation, return arrays, spacing, and orientation axcodes."""
    ct_nii = nib.load(ct_path)
    seg_nii = nib.load(seg_path)
    ct = ct_nii.get_fdata().astype(np.float32)
    seg = seg_nii.get_fdata().astype(np.int16)
    spacing = np.array(ct_nii.header.get_zooms()[:3], dtype=np.float32)
    axcodes = nib.aff2axcodes(ct_nii.affine)
    return ct, seg, spacing, axcodes


# -----------------------------------------------------------------------------
# Probe placement (orientation-aware)
# -----------------------------------------------------------------------------
def _centroid(mask):
    return np.argwhere(mask).mean(axis=0)


def _posterior_direction_from_axcodes(axcodes):
    """Unit vector in voxel space pointing posteriorly (toward spine)."""
    d = np.zeros(3, dtype=np.float32)
    for i, code in enumerate(axcodes):
        if code == 'A':
            d[i] = -1.0; return d
        if code == 'P':
            d[i] = +1.0; return d
    d[1] = -1.0
    return d


def compute_probe_pose(seg, spacing, axcodes, cfg):
    la_mask = seg == 2
    laa_mask = seg == 8
    if la_mask.sum() == 0 or laa_mask.sum() == 0:
        raise ValueError("Missing LA (2) or LAA (8) in segmentation.")

    la_c = _centroid(la_mask)
    laa_c = _centroid(laa_mask)

    posterior_voxel = _posterior_direction_from_axcodes(axcodes)
    axis = int(np.argmax(np.abs(posterior_voxel)))
    offset_voxels = posterior_voxel * (cfg.probe_offset_mm / spacing[axis])
    probe_pos = la_c + offset_voxels

    view_phys = (laa_c - probe_pos) * spacing
    view_dir = view_phys / (np.linalg.norm(view_phys) + 1e-8)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, view_dir)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    up = up - np.dot(up, view_dir) * view_dir
    up = up / (np.linalg.norm(up) + 1e-8)

    return (probe_pos.astype(np.float32),
            view_dir.astype(np.float32),
            up.astype(np.float32),
            la_c, laa_c)


# -----------------------------------------------------------------------------
# Ray casting and sampling
# -----------------------------------------------------------------------------
def build_fan_coordinates(probe_pos, view_dir, up, spacing, cfg):
    right = np.cross(view_dir, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    half = np.deg2rad(cfg.fan_angle_deg / 2.0)
    angles = np.linspace(-half, half, cfg.n_rays)
    depths_mm = np.linspace(0.0, cfg.depth_mm, cfg.n_depth_samples)

    ray_dirs = (np.cos(angles)[:, None] * view_dir[None, :]
                + np.sin(angles)[:, None] * right[None, :])

    depth_vox = depths_mm[:, None, None] * ray_dirs[None, :, :] / spacing[None, None, :]
    coords = probe_pos[None, None, :] + depth_vox
    return coords.astype(np.float32), depths_mm.astype(np.float32)


def sample_volume(volume, coords, order=1, cval=-1000.0):
    shp = coords.shape[:2]
    flat = coords.reshape(-1, 3).T
    sampled = map_coordinates(volume, flat, order=order, mode="constant",
                              cval=cval, prefilter=False)
    return sampled.reshape(shp)


# -----------------------------------------------------------------------------
# Ultrasound physics-lite
# -----------------------------------------------------------------------------
def _hu_to_echo(hu):
    soft_tissue = 0.5 * np.exp(-((hu - 50.0)**2) / (2 * 100.0**2))
    blood       = -0.3 * np.exp(-((hu - 300.0)**2) / (2 * 80.0**2))
    air         = -0.4 * (hu < -500.0).astype(np.float32)
    bone        =  0.6 * np.clip((hu - 400.0) / 400.0, 0.0, 1.0)
    return np.clip(0.35 + soft_tissue + blood + air + bone, 0.0, 1.5)


def render_bmode(hu, seg_along_rays, depths_mm, rng, cfg):
    echo = _hu_to_echo(hu)

    seg_int = seg_along_rays.astype(np.int16)
    seg_changes = (np.diff(seg_int, axis=0, prepend=seg_int[:1, :]) != 0).astype(np.float32)
    seg_changes = gaussian_filter(seg_changes,
                                  sigma=(cfg.wall_blur_depth, cfg.wall_blur_ray))
    echo = echo + cfg.wall_echo_strength * seg_changes

    dE = np.abs(np.diff(echo, axis=0, prepend=echo[:1, :]))
    reflect = cfg.reflection_gain * dE + cfg.bulk_scatter * np.clip(echo, 0, 1)

    mu_per_mm = (cfg.attenuation_db_per_cm / 10.0) * (np.log(10) / 20.0)
    atten = np.exp(-mu_per_mm * depths_mm)[:, None]
    sig = reflect * atten

    dense = (hu > cfg.shadow_threshold_hu).astype(np.float32)
    shadow = np.cumprod(1.0 - cfg.shadow_strength * dense * 0.1, axis=0)
    sig = sig * shadow

    speckle = rng.gamma(shape=1.0, scale=1.0, size=sig.shape).astype(np.float32)
    speckle = gaussian_filter(speckle, sigma=cfg.speckle_grain_px)
    sig = sig * (1.0 + cfg.speckle_strength * (speckle - speckle.mean()))

    sig = np.clip(sig, 0, None)
    sig = np.log1p(sig)
    sig = sig / (sig.max() + 1e-8)
    return np.power(sig, cfg.gamma).astype(np.float32)


# -----------------------------------------------------------------------------
# Scan conversion (polar -> fan)
# -----------------------------------------------------------------------------
def scan_convert(polar_image, cfg, order=1, mask_outside=True):
    H, W = cfg.out_size
    n_depth, n_rays = polar_image.shape
    half = np.deg2rad(cfg.fan_angle_deg / 2.0)

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W / 2.0
    dx = (xs - cx) / (W / 2.0)
    dy = ys / H

    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dx, dy)

    inside = (np.abs(theta) <= half) & (r <= 1.0)
    depth_idx = np.clip(r * (n_depth - 1), 0, n_depth - 1)
    ray_idx = np.clip((theta + half) / (2 * half) * (n_rays - 1), 0, n_rays - 1)

    src = np.stack([depth_idx.ravel(), ray_idx.ravel()], axis=0)
    out = map_coordinates(polar_image, src, order=order, mode="constant", cval=0.0)
    out = out.reshape(H, W)
    if mask_outside:
        out[~inside] = 0.0
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Top-level
# -----------------------------------------------------------------------------
def simulate_tee(ct, seg, spacing, axcodes=('L', 'A', 'S'), cfg=None):
    """
    Generate one synthetic TEE image from a CT volume + segmentation.

    Returns
    -------
    tee_image : (H, W) float32 in [0, 1], cone-shaped on black background
    tee_seg   : (H, W) int16, segmentation labels in cone coords (0 outside)
    meta      : dict with probe pose, config, sanity stats
    """
    if cfg is None:
        cfg = TEEConfig()
    rng = np.random.default_rng(cfg.seed)

    # Track per-patient HU stats for debugging (not used in rendering)
    body_mask = ct > -500.0
    if body_mask.sum() > 1000:
        hu_lo = float(np.percentile(ct[body_mask], 1.0))
        hu_hi = float(np.percentile(ct[body_mask], 99.5))
        hu_median = float(np.median(ct[body_mask]))
        bright_fraction = float((ct[body_mask] > 200).sum() / body_mask.sum())
    else:
        hu_lo, hu_hi, hu_median, bright_fraction = float('nan'), float('nan'), float('nan'), float('nan')

    probe_pos, view_dir, up, la_c, laa_c = compute_probe_pose(seg, spacing, axcodes, cfg)
    coords, depths_mm = build_fan_coordinates(probe_pos, view_dir, up, spacing, cfg)

    hu = sample_volume(ct, coords, order=1, cval=cfg.hu_clip[0])
    seg_along = sample_volume(seg.astype(np.float32), coords, order=0, cval=0.0)

    bmode = render_bmode(hu, seg_along, depths_mm, rng, cfg)

    tee_image = scan_convert(bmode, cfg, order=1, mask_outside=True)
    tee_seg = scan_convert(seg_along, cfg, order=0, mask_outside=True).astype(np.int16)

    meta = {
        "probe_pos_voxel": probe_pos.tolist(),
        "view_dir": view_dir.tolist(),
        "up": up.tolist(),
        "la_centroid": la_c.tolist(),
        "laa_centroid": laa_c.tolist(),
        "spacing": spacing.tolist(),
        "axcodes": list(axcodes),
        "laa_pixels_in_fan": int((tee_seg == 8).sum()),
        "laa_voxels_in_volume": int((seg == 8).sum()),
        "hu_lo": hu_lo,
        "hu_hi": hu_hi,
        "hu_median": hu_median,
        "bright_fraction": bright_fraction,
    }
    return tee_image, tee_seg, meta


def simulate_from_paths(ct_path, seg_path, cfg=None):
    """Convenience wrapper: load files and simulate."""
    ct, seg, spacing, axcodes = load_case(ct_path, seg_path)
    return simulate_tee(ct, seg, spacing, axcodes, cfg)
