# Data Templates

Use these templates when preparing private data for the public workflow. Keep the
column names unchanged.

| Template | Use |
| --- | --- |
| `paper_modeling_dataset_template.xlsx/.csv` | Minimal paper reproduction. Fill this one table, set `paper_experiment.input_table`, then run `--step paper-experiments`. |
| `demo_template.xlsx/.csv` | Demographic/label table for the full gait-speech joint workflow. |
| `gait_features_check_previous_template.xlsx/.csv` | `check_previous`-format gait feature table if you already have gait metrics. |

The public repository should include only these blank templates, never real
participant videos, audio, transcripts, demographics, or feature tables.
