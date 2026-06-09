# 最小复现模式：只用一个 paper_modeling_dataset.xlsx

这个模式适合只想复现论文模型结果、暂时不重新跑 MMPose/WhisperX 的用户。

## 1. 准备一个表格

从模板复制：

```bash
cp data/templates/paper_modeling_dataset_template.xlsx data/paper_modeling_dataset.xlsx
```

打开 `data/paper_modeling_dataset.xlsx`，在 `age-edu` sheet 中填入每个受试者一行数据。列名不要改。

必须保留的关键列：

| 列名 | 含义 |
| --- | --- |
| `ID` | 受试者或 case ID。 |
| `label` | 二分类标签，通常 `0` = 对照/非损害，`1` = 损害/case。 |
| `age` | 年龄。 |
| `edu` | 教育年限或教育水平数值。 |
| `MoCA` | MoCA 分数，用于回归结果；没有可留空并关闭回归。 |
| `MMSE` | MMSE 分数，保留以便后续分析。 |
| `*_DT` | 双任务 speech 特征。 |
| gait metrics | gaitfeature.py 或其他流程得到的 gait 特征。 |

## 2. 修改配置

复制配置：

```bash
cp config.example.ini config.ini
```

确认 `config.ini` 中这几项：

```ini
[paper_experiment]
input_table = data/paper_modeling_dataset.xlsx
sheet_name = age-edu
feature_set = nestedselected
run_nested_feature_selection = yes
promote_nested_selection = yes
```

如果没有 MoCA 分数：

```ini
run_regression = no
```

如果还没有安装 SHAP：

```ini
run_shap = no
```

## 3. 先检查，不跑模型

```bash
python run_pipeline.py --config config.ini --step validate
```

检查报告会写到：

```text
results/validation_report.json
```

每一步的状态会写到：

```text
results/step_status.json
```

如果 `validate` 报告缺列，请回到模板补齐列名。列可以暂时为空，但列名不能删。

## 4. 跑论文模型复现

```bash
python run_pipeline.py --config config.ini --step paper-experiments
```

主要输出：

| 输出 | 含义 |
| --- | --- |
| `results/paper_experiments/classification/model_performance_table.xlsx` | 主分类性能表。默认来自 nested held-out predictions。 |
| `results/paper_experiments/classification/main_output_source.json` | 说明主分类表/图来自哪一路结果。 |
| `results/paper_experiments/classification/nested_feature_selection/` | 严格训练折内特征选择结果。 |
| `results/paper_experiments/classification/confounder_adjustment/` | 年龄、教育、性别混杂因素校正。 |
| `results/paper_experiments/regression/` | MoCA/MMSE 等回归结果，和分类分开。 |

## 5. 推荐报告方式

分类主结果优先报告：

```text
classification/model_performance_table.xlsx
```

并在方法中说明：

```text
Feature selection was repeated inside each outer training fold, and the held-out
fold was used only for final evaluation.
```

如果要说明固定特征面板，请同时阅读：

```text
docs/feature_selection_protocol.md
```
