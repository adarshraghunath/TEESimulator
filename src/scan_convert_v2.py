"""
Fixed scan conversion with rounded fan geometry.

Fan layout:
  - Apex at top-center (x = W/2, y = 0)
  - Rays diverge from apex within +/- fan_angle/2 of vertical
  - Maximum depth = a configurable fraction of image height
  - The fan reaches a true ARC (not a straight line) at maximum depth
  - Everything outside the arc is masked black
"""

import numpy as np
from scipy.ndimage import map_coordinates


# Fraction of image height the fan should span. Default 0.95 leaves a small
# margin at the bottom. The arc at max depth touches the bottom of the image
# along its centerline and curves up toward the sides — that's the real fan shape.
FAN_DEPTH_FRAC = 0.95


def scan_convert(polar_image, cfg, order=1, mask_outside=True, depth_frac=FAN_DEPTH_FRAC):
    """
    Convert a polar (n_depth, n_rays) image into a rounded-fan 2D image.

    Geometry:
      - Apex at top-center, (x = W/2, y = 0)
      - r_max defines the maximum distance from the apex (depth in pixels)
      - Output fan is bounded by:
            r <= r_max  AND  |theta| <= fan_angle/2
      - Set r_max so that the centerline of the fan (theta=0) reaches
        a chosen fraction of the image height at maximum depth.
    """
    H, W = cfg.out_size
    n_depth, n_rays = polar_image.shape
    half = np.deg2rad(cfg.fan_angle_deg / 2.0)

    # Pixel grid
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

    # Apex at top-center
    x_apex = W / 2.0
    y_apex = 0.0

    # Vector from apex
    dx_pix = xs - x_apex
    dy_pix = ys - y_apex

    # Polar coords: r = distance, theta = angle from straight-down
    r_pix = np.sqrt(dx_pix ** 2 + dy_pix ** 2)
    theta = np.arctan2(dx_pix, dy_pix)

    # Set max radius so the fan centerline (theta=0) reaches depth_frac * H
    # at maximum depth. At theta=0, r = y, so r_max = depth_frac * H.
    r_max = float(depth_frac * H)

    # Inside fan: within angular range AND within radius
    inside = (np.abs(theta) <= half) & (r_pix <= r_max)

    # Map (r, theta) to source polar indices
    depth_idx_f = r_pix / r_max * (n_depth - 1)
    ray_idx_f = (theta + half) / (2 * half) * (n_rays - 1)

    depth_idx = np.clip(depth_idx_f, 0, n_depth - 1)
    ray_idx = np.clip(ray_idx_f, 0, n_rays - 1)

    src = np.stack([depth_idx.ravel(), ray_idx.ravel()], axis=0)
    out = map_coordinates(polar_image, src, order=order, mode="constant", cval=0.0)
    out = out.reshape(H, W)

    if mask_outside:
        out[~inside] = 0.0

    if order == 0:
        return out.astype(polar_image.dtype)
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Labels-only simulation (no physics, much faster)
# -----------------------------------------------------------------------------
def simulate_labels_only(ct_path, seg_path, cfg=None):
    """
    Produces ONLY the label map. Skips B-mode physics rendering entirely.
    Use this when only the segmentation is needed (label-conditional diffusion).
    """
    from tee_simulator import (
        load_case,
        compute_probe_pose,
        build_fan_coordinates,
        sample_volume,
        TEEConfig,
    )
    if cfg is None:
        cfg = TEEConfig()

    ct, seg, spacing, axcodes = load_case(ct_path, seg_path)
    probe_pos, view_dir, up, la_c, laa_c = compute_probe_pose(seg, spacing, axcodes, cfg)
    coords, _ = build_fan_coordinates(probe_pos, view_dir, up, spacing, cfg)

    seg_along = sample_volume(seg.astype(np.float32), coords, order=0, cval=0.0)
    tee_seg = scan_convert(seg_along, cfg, order=0, mask_outside=True).astype(np.int16)

    meta = {
        "probe_pos_voxel": probe_pos.tolist(),
        "view_dir": view_dir.tolist(),
        "la_centroid": la_c.tolist(),
        "laa_centroid": laa_c.tolist(),
        "spacing": spacing.tolist(),
        "axcodes": list(axcodes),
        "laa_pixels_in_fan": int((tee_seg == 8).sum()),
        "laa_voxels_in_volume": int((seg == 8).sum()),
    }
    return tee_seg, meta
