# TEESimulator
CPU implementation
Simulates a TEE-style B-mode image from a 2D segmentation label map alone (no CT), using the
open-source [PyMUST](https://github.com/creatis-ULTIM/PyMUST) (SIMUS) ultrasound simulator.

## Files

```
simulate_tee.py       the simulator (single script)
requirements.txt      pinned versions
examples/input/       four input label maps (.npy)
examples/output/      simulated outputs for those inputs (.npy and .png)
```

## Setup

Tested with Python 3.11 on Windows.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.venv\Scripts\python.exe simulate_tee.py `
    --labels examples\input\000001.npy examples\input\000002.npy `
    --out_dir out --n_tx 21 --density 4
```

About 4 minutes per label on CPU. `--n_tx 3 --density 3` is a quick smoke test. `--fc` overrides the probe
center frequency (Hz).

## Input

A 2D label map, `.npy`, shape (256, 352), integer classes from the STACOM 11-class scheme:
0 unlabelled, 1 myocardium, 2 LA, 3 LV, 4 RA, 5 RV, 6 aorta, 7 PA, 8 LAA, 9 coronary, 10 PV.
Geometry: apex top-center, 90 degree fan, fan depth = 0.95 x image height (243 px), spanning 90 mm
(about 0.37 mm per pixel). The canvas must be at least `2 x 243 x sin(45 deg) = 344` px wide, otherwise the lower
corners of the fan are cut off; the examples use 352. Any height and width work as long as this geometry
holds. The four examples in `examples/input` are fan-view label maps for ImageCAS patients 1-4, generated from the
STACOM segmentations.

## Output

Per input, in `--out_dir`:

| file | content |
|---|---|
| `<name>.npy` | simulated image, float32 in [0, 1], same grid as the input, zero outside the fan |
| `<name>.png` | the same image as 8-bit grayscale |
| `<name>_label.png` | the input label, colored |
| `<name>_compare.png` | label and simulated image side by side |

`examples/output` holds these files for the four example inputs. The seed is fixed (0), so the same
arguments and package versions give the same image.

## How it works

1. Random scatterers are placed in every fan pixel (Poisson count from `--density`, per mm^2).
2. Each scatterer gets a Gaussian reflectivity scaled by a hand-chosen per-class amplitude
   (`CLASS_AMP`): myocardium 1.0, blood pools 0.03, unlabelled tissue 0.35, coronary 0.3.
   A 3-pixel wall of amplitude 1.0 is added around the blood pools.
3. PyMUST `getparam('P4-2v')` phased array (2.72 MHz, 0.5 dB/cm/MHz attenuation), 21 diverging waves tilted
   from -40 to +40 degrees: `txdelay`, `simus` (RF), `tgc`, `rf2iq`, `dasmtx` delay-and-sum on the pixel grid.
4. Beamformed I/Q is summed coherently over the transmits, then `bmode` with a 40 dB dynamic range,
   min-max normalized to [0, 1] and masked outside the fan.

## Limitations

- The reflectivities, wall model and the 2.7 MHz P4-2v probe are heuristics, not derived from CT or
  calibrated against real TEE. A 5 MHz TEE probe would give finer resolution.
- It works on one 2D slice, so there is no 3D structure or elevation effects.

## Citations

- Bernardino G., Garcia D., "PyMUST: an open-Source Python Library for the Simulation and Analysis of
  Ultrasound", IEEE UFFC-JS 2024, doi:10.1109/uffc-js60046.2024.10793881
- Garcia D., "SIMUS: an open-source simulator for medical ultrasound imaging. Part I: theory & examples",
  Computer Methods and Programs in Biomedicine 218 (2022) 106726, doi:10.1016/j.cmpb.2022.106726
- Garcia D., "Make the most of MUST, an open-source MATLAB UltraSound Toolbox", IEEE IUS 2021,
  doi:10.1109/IUS52206.2021.9593605
- Perrot V. et al., "So you think you can DAS? A viewpoint on delay-and-sum beamforming",
  Ultrasonics 111 (2021) 106309, doi:10.1016/j.ultras.2020.106309

The example labels come from the STACOM LAA label maps on ImageCAS: Hansen et al., "A Public Cardiac CT
Dataset Featuring the Left Atrial Appendage", STACOM 2025, and the ImageCAS dataset. Check their licenses
before redistributing.

## License

MIT, see `LICENSE`. PyMUST is LGPL-2.1.
