# Environment Notes

This public package separates lightweight feature/analysis code from heavy model environments.

## Basic Feature And Analysis Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-basic.txt
```

The basic environment supports gait feature extraction from existing MMPose JSON files, speech feature extraction from existing WhisperX JSON files, and experiment analysis.

## WhisperX

WhisperX depends on PyTorch and is usually easiest to run on a GPU server.

```bash
pip install -r requirements-whisperx.txt
```

If CUDA is available, install the PyTorch build matching your server before installing WhisperX.

## Paper Experiment Reproduction

The paper-level model benchmark and SHAP figures require scikit-learn, pandas, and SHAP:

```bash
pip install -r requirements-paper.txt
```

`shap` is only needed when `paper_experiment.run_shap = yes`.

## MMPose

MMPose installation is version-sensitive. Install it according to the official MMPose documentation for your CUDA, PyTorch, MMEngine, MMDetection, and MMPose versions.

This package does not hard-code one MMPose command. Instead, set `mmpose_command_template` in `config.ini`.

Known project environment from previous runs:

- Ubuntu 20.04.6 LTS
- conda environment: `openmmlab`
- `torch 2.4.1`
- `mmengine 0.10.7`
- `mmpose 1.3.2`
- `mmdet 3.2.0`

Known model choices tested in this project:

- `--pose2d human --bbox-thr 0.3`
- `td-hm_hrnet` with `bbox_thr=0.3`
