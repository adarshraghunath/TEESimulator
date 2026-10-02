# TEESimulator

Synthetic TEE (transesophageal echocardiography) B-mode images from CT and 11-class segmentation
volumes, by label-guided ray casting. Also includes a CT-free, label-only variant that uses the
open-source PyMUST (SIMUS) ultrasound simulator.

## Layout

```
src/
  tee_simulator.py        CT/seg loading, probe pose, fan ray casting
  render_bmode.py         realistic B-mode renderer (do not edit; extracted from the locked notebook)
  scan_convert_v2.py      polar -> 256x256 fan conversion, labels-only path
  batch_simulate_v3.py    batch driver (the entry point for the CT pipeline)
  filter_sim_v3.py        keeps patients with complete LAA (dataset's own ID list, HPC paths hard-coded)
  label_only_pymust.py    CT-free simulation from a 2D label map (PyMUST/SIMUS)
scripts/verify_labels.py  checks labels and LAA counts against examples/expected_output
examples/
  expected_output/        reference output for patients 1-3 (images, labels, manifest, cone_mask)
  label_only/             example output of the CT-free simulation for patient 1
requirements.txt          pins for the CT pipeline and verify script
requirements-labelonly.txt
```

## Setup

Tested with Python 3.11 on Windows.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Input

Per patient, one CT and one segmentation, both named `{pid}.img.nii.gz` (pid is any integer).
Segmentations use the STACOM 11-class labels (0 background, 1 myocardium, 2 LA, 3 LV, 4 RA, 5 RV,
6 aorta, 7 PA, 8 LAA, 9 coronary, 10 PV).

- Segmentations: the public STACOM LAA label maps built on ImageCAS
  (https://github.com/Bjonze/Public-Cardiac-CT-Dataset).
- CT volumes: the public ImageCAS dataset (Kaggle / ImageCAS GitHub). They are not in this repo.

No dataset files are included here. Check the dataset licenses before redistributing any derived data;
`examples/expected_output` contains output derived from patients 1-3.

## Run the CT pipeline

```powershell
.venv\Scripts\python.exe src\batch_simulate_v3.py `
    --ct_dir  <folder with CT {pid}.img.nii.gz> `
    --seg_dir <folder with seg {pid}.img.nii.gz> `
    --out_dir out `
    --mask_sample_size 3 --limit 3
```

`--mask_sample_size` only limits how many patients the cone mask is derived from; `--limit` caps the
patients simulated (added in this repo; the original script documented it but did not implement it).
The run resumes from `images/` if interrupted. Omitting `--ct_dir/--seg_dir/--out_dir`
uses the original author's HPC paths, so always pass them.

Output under `--out_dir`: `images/{pid:06d}.npy` (float32 in [0, 1], 256x256), `labels/{pid:06d}.npy`
(int16), `images_png/`, `previews/`, `manifest.csv` and the cached `cone_mask.npy`.

## Expected output

`examples/expected_output` holds the reference result for patients 1-3 from the original run:

| file | content |
|---|---|
| `images/{pid}.npy` | simulated TEE image, float32, 256x256, cone-normalized |
| `labels/{pid}.npy` | fan label map, int16 |
| `manifest.csv` | `laa_pixels_in_fan`, `laa_voxels_in_volume` per patient |
| `cone_mask.npy` | cone mask derived from the original 32-patient run |

Reference LAA counts: patient 1: 4389 px / 116916 voxels; patient 2: 789 / 24784; patient 3: 4284 / 153879.

Note: that `cone_mask.npy` came from 32 patients. A `--mask_sample_size 3` run derives a noisier mask, so
images will differ in the cone region. To compare images, copy `cone_mask.npy` into your `--out_dir` before running; it is
then reused instead of re-derived.

## Verify without CT (what is tested)

`scripts/verify_labels.py` needs only the STACOM segmentations. It runs the geometry and label path
with the segmentation standing in for the CT and compares with the expected labels and counts:

```powershell
.venv\Scripts\python.exe scripts\verify_labels.py --seg_dir <folder with seg {pid}.img.nii.gz>
```

Result on the reference environment (numpy 1.26.3, scipy 1.13.1):

```
000001  counts_match=True  fan_edge_px=131  other_mismatch_px=0  PASS
000002  counts_match=True  fan_edge_px=128  other_mismatch_px=0  PASS
000003  counts_match=True  fan_edge_px=137  other_mismatch_px=0  PASS
```

LAA counts match exactly. About 130 pixels per patient differ from the reference only on the 45-degree
fan edge (`|x - 128| == y`), probably a float tie in the `abs(theta) <= half` test of `scan_convert_v2.py`.
The script treats that as a pass.

**Not verified:** the B-mode images. They need the CT volumes, which were not available when this was
set up, so image reproduction against `expected_output/images` has not been run. The batch driver itself was
smoke-tested with the segmentation folder passed as both `--ct_dir` and `--seg_dir` (3 patients, `ok=3 fail=0`,
LAA counts matching the manifest); the images from that run are meaningless.

## Changes from the original code

- Added `--limit` to `batch_simulate_v3.py`.
- Added `requirements.txt` (the original README referenced one that did not exist).
- `render_bmode.py`, `tee_simulator.py` and `scan_convert_v2.py` are unchanged copies.

## Label-only simulation (no CT)

Uses PyMUST (SIMUS) with hand-chosen per-class reflectivities on a 2D fan label map. It is a different
simulator and does not reproduce the CT-based images.

```powershell
python -m venv .venv-labelonly
.venv-labelonly\Scripts\python.exe -m pip install -r requirements-labelonly.txt
.venv-labelonly\Scripts\python.exe src\label_only_pymust.py `
    --label examples\expected_output\labels\000001.npy --out out\label_only_000001 --n_tx 21 --density 4
```

About 4 minutes on CPU. Compare with `examples/label_only/000001.png`. The reflectivity table (`CLASS_AMP`),
the 2.7 MHz P4-2v probe and the wall model are heuristics, not derived from CT.

## How the CT pipeline works

1. Probe is placed 25 mm posterior of the LA centroid, aimed at the LAA centroid.
2. 192 rays over a 90-degree fan, 480 samples to 90 mm; CT sampled linearly, labels by nearest neighbour.
3. `render_bmode_realistic`: HU to impedance, per-class scatter, Rayleigh speckle, interface highlights,
   attenuation 0.5 dB/cm/MHz at 5 MHz with partial TGC, anisotropic blur, noise, 45 dB log compression.
4. Polar-to-fan scan conversion to 256x256, apex top-center.
5. Cone-region normalization: pixels inside the empirically derived cone are inverted (`1 - intensity`).

The physics constants are hand-tuned and no source method is cited in the code.

## Citations

If you use the label-only simulation, cite PyMUST and SIMUS:

- Bernardino G., Garcia D., "PyMUST: an open-Source Python Library for the Simulation and Analysis of
  Ultrasound", IEEE UFFC-JS 2024, doi:10.1109/uffc-js60046.2024.10793881
- Garcia D., "SIMUS: an open-source simulator for medical ultrasound imaging. Part I: theory & examples",
  Computer Methods and Programs in Biomedicine 218 (2022) 106726, doi:10.1016/j.cmpb.2022.106726
- Garcia D., "Make the most of MUST, an open-source MATLAB UltraSound Toolbox", IEEE IUS 2021,
  doi:10.1109/IUS52206.2021.9593605
- Perrot V. et al., "So you think you can DAS? A viewpoint on delay-and-sum beamforming",
  Ultrasonics 111 (2021) 106309, doi:10.1016/j.ultras.2020.106309

For the input labels cite Hansen et al., "A Public Cardiac CT Dataset Featuring the Left Atrial
Appendage", STACOM 2025, and the ImageCAS dataset it builds on.

## License

MIT, see `LICENSE`.
