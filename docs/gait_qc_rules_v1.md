# Gait QC Rules v1

日期: 2026-03-27  
适用范围: `pred_2025`、`pred_2026` 合并后的步态特征表，以及后续同结构 gait feature 输出  
目标: 在正式统计分析前，自动筛除识别失败、步态事件切分失败、有效步数不足或数值逻辑异常的样本

## 1. 背景

本版 QC 规则不是凭经验拍阈值，而是根据以下对照推出来的:

1. 当前自动清洗池:
   - 原始合并 `804` 行
   - 去重后 `708` 行
2. 人工整理参考集:
   - `gait_data_old&new.xlsx` 中 `new-330`
   - 唯一 ID `415`
3. 关键发现:
   - 在原先“未被自动判坏，但人工删除”的 `195` 个样本中:
     - `step_time_variability = 0` 的比例是 `100%`
     - `stride_time_variability = 0` 的比例是 `100%`
   - 把这两个规则加入后，自动 `analysis_ready` 集合与 `new-330` 完全一致:
     - 自动保留 `415`
     - 与 `new-330` 交集 `415`
     - 无额外保留，也无额外漏删

结论:
- “时间变异性被写成 0” 在当前流程里不应被当作正常结果
- 它应被解释为:
  - 有效步数不足
  - 步态事件切分失败
  - 序列长度不足仍被强行输出特征

## 2. QC 总原则

QC 分两层:

1. 硬性失败 `qc_fail`
   - 样本直接排除，不进入正式分析
2. 软性复核 `qc_review`
   - 样本先保留到复核池，由人工或二级模型决定

当前 v1 先实现稳定的硬性失败规则。

## 3. 硬性失败规则

任意命中以下任一条件，标记为 `qc_fail`:

### 3.1 时间变异性为 0

- `step_time_variability == 0`
- `stride_time_variability == 0`

解释:
- 在本项目数据中，这两个条件几乎完全对应人工删除样本
- 这不是“步态特别稳定”，而更可能是事件数不足或计算退化成常数

### 3.2 时间特征整块缺失

以下 4 个字段同时缺失:
- `average_step_time`
- `average_stride_time`
- `step_time_variability`
- `stride_time_variability`

解释:
- 说明时间维度没有成功提取
- 即使空间特征存在，也不应继续进入正式分析

### 3.3 节律明显异常

- `cadence <= 20`
- `cadence > 200`

解释:
- 低于 20 多数不是慢步态，而是有效步数几乎没有被检测到
- 高于 200 对当前人群和场景不合理

### 3.4 速度明显异常

- `walking_speed <= 0`
- `walking_speed > 2.5`

解释:
- 小于等于 0 一定不合理
- 大于 2.5 在当前数据场景下高度可疑

### 3.5 空间长度异常

- `average_step_length > 2000`
- `average_stride_length > 4000`

解释:
- 这类值通常来自关键点抖动、尺度失败或事件配对错误

### 3.6 时间尺度异常

- `average_step_time > 5`
- `average_stride_time > 10`

解释:
- 在当前步行任务里，这类值通常不是正常步态，而是分段或事件识别失败

### 3.7 相位异常

- `stance_times > 30`
- `swing_times > 30`

解释:
- 当前单位和任务长度下，这类结果不合理

## 4. 软性复核规则

以下指标不建议单独作为硬性淘汰，但建议进入二级 QC 打分或人工复核:

### 4.1 节律-速度组合异常

如果同时满足:
- `walking_speed` 偏高
- `cadence` 偏高
- 步态变异接近 0 或非常低

解释:
- 这类样本“看起来运动表现很好”，但更像事件检测把序列过度简化了

### 4.2 相位闭合不一致

建议计算:
- `stance_swing_cycle_error = |stance_times + swing_times - average_stride_time|`

若明显偏大，提示:
- 相位切分不稳
- 周期定义不一致

### 4.3 相对变异比

建议计算:
- `step_var_ratio = step_time_variability / average_step_time`
- `stride_var_ratio = stride_time_variability / average_stride_time`
- `step_len_var_ratio = step_length_variability / average_step_length`
- `stride_len_var_ratio = stride_length_variability / average_stride_length`

解释:
- 绝对值不极端，但相对比值异常时，常提示序列稳定性差或步数不足

## 5. 上游流程要求

为了避免下游再靠 QC 补救，建议在特征提取端直接修改:

### 5.1 不允许“假 0”

如果有效步数不足，以下字段不能写成 `0`:
- `step_time_variability`
- `stride_time_variability`

应改为:
- `NA`
- 或 `insufficient_steps`
- 或整条样本标记 `qc_fail`

### 5.2 增加上游元数据

后续每条视频建议同时输出:
- `valid_step_count`
- `valid_stride_count`
- `left_right_switch_count`
- `usable_frame_ratio`
- `keypoint_missing_ratio`
- `event_detection_success_flag`

这样 QC 可以更早发现“不是 gait 差，而是提取失败”。

### 5.3 有效步数门槛

建议新增规则:
- 若 `valid_step_count < 最低门槛`，直接不计算 variability

当前文档不写死门槛数值，因为需要结合事件定义和视频长度再定。

## 6. 当前正式清洗逻辑

当前脚本:
- `clean_merged_gait_pred_pool.py` (historical private cleaning script; not included in this public package)

当前正式输出:
- `merged_gait_pred_2025_2026_cleaned.csv` (private derived output; not included)
- `merged_gait_pred_2025_2026_analysis_ready_matched_demo.csv` (private derived output; not included)

当前清洗摘要:
- `merged_gait_pred_2025_2026_cleaning_summary.json` (private derived output; not included)

## 7. v1 结论

当前这套 v1 规则的实际效果是:

1. 自动清洗结果与人工 `new-330` 完全对齐
2. 最大新增价值规则是:
   - `step_time_variability == 0`
   - `stride_time_variability == 0`
3. 说明当前流程的主要误差来源不是少数极端异常值，而是:
   - 有效步数不足仍强行输出特征
   - 时间序列退化后被误写为合法数值 0

## 8. 建议的后续版本

### v2
- 在输出层补 `valid_step_count / valid_stride_count`
- 把 variability 的“0”改成 `NA`

### v3
- 基于 `new-330` 训练一个软性 QC 打分器
- 不只输出 `qc_fail`，还输出 `qc_score`

### v4
- 在 pose 端就做序列级质量控制
- 减少后面 feature 层的被动筛除
