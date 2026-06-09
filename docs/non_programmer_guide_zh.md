# 非计算机专业用户快速指南

这个文件夹把论文中的 gait-speech 工作流整理成“改配置、放数据、点运行”的模式。

## 1. 准备文件夹

把公开代码文件夹复制到服务器或本机后，保持默认结构：

```text
paper_public_code/
  data/
  results/
  config.example.ini
  run_pipeline.py
```

不要把受试者隐私数据上传到公开仓库。公开视频、音频、人口学表格请只放在自己本地或服务器的 `data/` 文件夹。

## 2. 编辑配置

复制一份配置文件：

```bash
cp config.example.ini config.ini
```

然后打开 `config.ini`，主要改这些位置：

| 配置项 | 说明 |
| --- | --- |
| `gait.video_dir` | 步态视频所在文件夹。 |
| `gait.keypoint_json_dir` | MMPose 输出 JSON 所在文件夹。 |
| `gait.run_mmpose` | 是否运行 MMPose。已有 JSON 时填 `no`。 |
| `gait.mmpose_command_template` | 你服务器上的 MMPose 命令或 batch 脚本。 |
| `speech.task8_audio_dir` | 单任务语音 TASK08 wav 文件夹。 |
| `speech.task9_audio_dir` | 双任务语音 TASK09 wav 文件夹。 |
| `speech.run_whisperx` | 是否运行 WhisperX。已有 ASR JSON 时填 `no`。 |
| `experiment.demo_xlsx` | 人口学、label、MoCA/MMSE 表。 |
| `experiment.gait_xlsx` | check_previous 格式的 gait feature 表。 |

## 3. 常用运行方式

先看命令是否正确，不真正运行重任务：

```bash
python run_pipeline.py --config config.ini --step validate
python run_pipeline.py --config config.ini --dry-run
```

只从已有 MMPose JSON 提取 gait features：

```bash
python run_pipeline.py --config config.ini --step gait-features
```

只从已有 WhisperX JSON 提取 speech features：

```bash
python run_pipeline.py --config config.ini --step speech-features
```

复现 gait-speech 实验分析：

```bash
python run_pipeline.py --config config.ini --step experiments
```

复现论文里的模型比较、AdaBoost/RF/SVM(RBF)/DT × all features/Nested-selected 表格、confusion matrix、ROC、calibration curve、混杂因素校正、bootstrap 95% CI、SHAP 图和回归 Table 2 风格结果：

```bash
python run_pipeline.py --config config.ini --step paper-experiments
```

完整流程：

```bash
python run_pipeline.py --config config.ini --step all
```

## 4. 输出结果在哪里

默认输出在 `results/`：

| 文件夹 | 内容 |
| --- | --- |
| `results/gait_keypoints/` | MMPose 识别结果。 |
| `results/gait_features/` | gait feature JSON、check_previous 格式 xlsx 和 QC 文件。 |
| `results/speech_asr/` | WhisperX 转写 JSON。 |
| `results/speech_features/` | Serial-7 speech feature JSON。 |
| `results/experiments/` | joint dataset、分类结果、回归结果、summary。 |
| `results/validation_report.json` | 路径、列名、依赖、输出目录检查结果。 |
| `results/step_status.json` | 每一步的 success/skipped/failed 状态、命令和错误尾部。 |
| `results/paper_experiments/classification/` | 论文展示用分类表、confusion matrix、ROC 和 calibration curve。 |
| `results/paper_experiments/classification/nested_feature_selection/` | 每一折只在 training set 内重新选特征的无泄漏 feature-selection 结果。 |
| `results/paper_experiments/classification/confounder_adjustment/` | age-adjusted、education-adjusted、age+sex+education adjusted 模型和 AUROC 稳定性报告。 |
| `results/paper_experiments/regression/` | 回归 Table 2 风格结果和图，和分类结果分开。 |


## 5. 重要提醒

MMPose 和 WhisperX 都依赖 GPU 环境，最容易出错的是 CUDA、PyTorch、MMPose、MMDetection 版本不匹配。如果识别阶段已经由服务器完成，公开代码用户可以直接跳过识别，只用 JSON 做特征提取和复现实验。

Nested-selected features 是根据 nested feature selection 的训练折高频结果形成的固定主文特征表。严格无泄漏的“边选特征边评估”性能仍应优先报告 `classification/nested_feature_selection/` 下的结果，并阅读 `docs/feature_selection_protocol.md`。

默认设置 `promote_nested_selection = yes` 时，`classification/model_performance_table.xlsx`、ROC、calibration 和 confusion matrix 会使用 nested held-out predictions，而不是固定特征表的普通 train-test split 结果。固定特征表结果会保留为 `fixed_panel_*` 文件，方便审计。

如果只想复现论文模型，不想重新跑 MMPose/WhisperX，请直接使用 `data/templates/paper_modeling_dataset_template.xlsx`，并阅读 `docs/minimal_paper_reproduction_zh.md`。
