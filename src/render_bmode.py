"""
Realistic B-mode renderer — extracted verbatim from realistic_sim_LOCKED.ipynb.

DO NOT EDIT. If you want to change the physics, edit the LOCKED notebook, verify
visually, then re-extract to keep the two in sync.

Included physics:
  - HU -> tissue impedance (piecewise map)
  - Per-class scatter coefficients (LAA/myocardium bright, blood pool dark)
  - Rayleigh speckle with signal**0.7 grain
  - Interface highlighting at tissue transitions along ray
  - Depth attenuation (0.5 dB/cm/MHz)
  - Partial TGC (0.4 recovery)
  - Anisotropic PSF (sigma 0.7 lateral, 0.4 axial)
  - Electronic noise
  - Log compression, 45 dB dynamic range
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_dilation


CLASS_SCATTER = {
    0:  0.15, 1:  1.00, 2:  0.10, 3:  0.10, 4:  0.10, 5:  0.10,
    6:  0.80, 7:  0.60, 8:  0.90, 9:  0.60, 10: 0.50,
}


def hu_to_impedance(hu):
    hu = np.asarray(hu, dtype=np.float32)
    e = np.zeros_like(hu)
    e = np.where(hu < -500, 0.40, e)
    e = np.where((hu >= -500) & (hu < 0),   0.40 + (hu + 500) / 500 * 0.4, e)
    e = np.where((hu >=   0) & (hu < 100),  0.42 + hu / 100 * 0.2, e)
    e = np.where((hu >= 100) & (hu < 400),  0.62 + (hu - 100) / 300 * 0.23, e)
    e = np.where(hu >= 400,                 0.85 + np.clip((hu - 400) / 1000, 0, 0.15), e)
    return np.clip(e, 0.0, 1.0)


def render_bmode_realistic(hu, seg_along_rays, depths_mm, rng, cfg):
    n_depth, n_rays = hu.shape

    base_intensity = hu_to_impedance(hu)
    scatter_map = np.ones_like(base_intensity, dtype=np.float32) * 0.5
    for cls, s in CLASS_SCATTER.items():
        scatter_map = np.where(seg_along_rays == cls, s, scatter_map)
    signal = base_intensity * scatter_map

    g1 = rng.standard_normal(signal.shape).astype(np.float32)
    g2 = rng.standard_normal(signal.shape).astype(np.float32)
    speckle = np.sqrt(g1*g1 + g2*g2)
    rf_envelope = signal ** 0.7 * speckle

    seg_int = seg_along_rays.astype(np.int32)
    seg_shifted = np.roll(seg_int, -1, axis=0)
    boundary = (seg_int != seg_shifted) & (seg_int > 0) & (seg_shifted > 0)
    boundary = binary_dilation(boundary, iterations=1)
    rf_envelope = rf_envelope + 0.3 * boundary * rf_envelope.max()

    freq_mhz = getattr(cfg, 'center_freq_mhz', 5.0)
    alpha_np_per_cm = 0.5 * freq_mhz / 8.686
    depths_cm = depths_mm / 10.0
    attenuation = np.exp(-alpha_np_per_cm * depths_cm)
    rf_envelope = rf_envelope * attenuation[:, None]

    tgc_gain = np.exp(0.4 * alpha_np_per_cm * depths_cm)
    rf_envelope = rf_envelope * tgc_gain[:, None]

    rf_envelope = gaussian_filter1d(rf_envelope, sigma=0.7, axis=1, mode='reflect')
    rf_envelope = gaussian_filter1d(rf_envelope, sigma=0.4, axis=0, mode='reflect')

    electronic_noise = rng.standard_normal(rf_envelope.shape).astype(np.float32) * 0.02
    rf_envelope = np.abs(rf_envelope + electronic_noise)

    eps = 1e-6
    log_signal = 20.0 * np.log10(rf_envelope + eps)
    dynamic_range_db = 45.0
    max_db = log_signal.max()
    bmode = (log_signal - (max_db - dynamic_range_db)) / dynamic_range_db
    bmode = np.clip(bmode, 0.0, 1.0)
    return bmode.astype(np.float32)


def simulate_realistic(ct_path, seg_path, cfg=None, seed=42):
    """Full CT->TEE simulation using the realistic renderer above."""
    from tee_simulator import (
        load_case, compute_probe_pose, build_fan_coordinates, sample_volume,
        TEEConfig,
    )
    from scan_convert_v2 import scan_convert

    if cfg is None:
        cfg = TEEConfig()
    if not hasattr(cfg, 'center_freq_mhz'):
        cfg.center_freq_mhz = 5.0
    rng = np.random.default_rng(seed)

    ct, seg, spacing, axcodes = load_case(ct_path, seg_path)
    probe_pos, view_dir, up, la_c, laa_c = compute_probe_pose(seg, spacing, axcodes, cfg)
    coords, depths_mm = build_fan_coordinates(probe_pos, view_dir, up, spacing, cfg)

    ct_along = sample_volume(ct, coords, order=1, cval=-1024.0)
    seg_along = sample_volume(seg.astype(np.float32), coords, order=0, cval=0.0)

    bmode_polar = render_bmode_realistic(ct_along, seg_along.astype(np.int32),
                                          depths_mm, rng, cfg)

    tee_img = scan_convert(bmode_polar, cfg, order=1, mask_outside=True)
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
    }
    return tee_img.astype(np.float32), tee_seg, meta
