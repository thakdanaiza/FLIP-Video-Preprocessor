# FLIP Video Preprocessor

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22939929.svg)](https://doi.org/10.5281/zenodo.22939929)

Minimal release of the original paired-view video preprocessing code used in the experiment.

## Requirements

- Python 3.12
- FFmpeg
- Packages in `requirements.txt`

## Create or update a crop profile

```bash
python3 video_crop_calibrator_fixed.py input.mp4 \
  --out profiles/crop_calibration_space_13.json \
  --crf 0 --preset veryslow
```

Controls: `s` saves the profile, `m` matches seam brightness, `v` exports a test video, and `q` exits.

## Process videos

Camera setup 13:

```bash
python3 batch_flip_videos.py \
  --src /path/to/input_13 --dst /path/to/output_13 \
  --config profiles/crop_calibration_space_13.json \
  --crf 0 --preset veryslow --skip-existing
```

Camera setup 24:

```bash
python3 batch_flip_videos.py \
  --src /path/to/input_24 --dst /path/to/output_24 \
  --config profiles/crop_calibration_space_24.json \
  --crf 0 --preset veryslow --skip-existing
```

Both profiles and published commands fix the encoder settings at H.264 CRF 0 and preset `veryslow`. The original crop geometry and seam-light settings are unchanged.

## Citation

Use GitHub's **Cite this repository** function or cite the archived release: https://doi.org/10.5281/zenodo.22939929
