# Example Folder Layout

`synthetic_features.csv` is generated entirely from random numbers by
`adualnet_toolkit.analysis.make_demo`. All IDs start with SYNTHETIC. It has no
study participants and can be used to test upload/preflight functionality.
The command `python adualnet.py demo` generates this artificial input locally.

The public code expects this layout by default:

```text
paper_public_code/
  config.ini
  data/
    gait_videos/
      CASE_001.mp4
    speech_audio/
      task8/
        CASE_001_8.wav
      task9/
        CASE_001_9.wav
    demo.xlsx
    gait_features_check_previous_format.xlsx
    paper_modeling_dataset.xlsx
  results/
    gait_keypoints/json_flat/
    gait_features/
      gait_features_check_previous_format.xlsx
    speech_asr/task8/
    speech_asr/task9/
    speech_features/
    experiments/
    paper_experiments/
```

Study data are intentionally not included in the public package.
