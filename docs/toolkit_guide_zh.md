# ADualNet 本地工具使用说明

## 先跑通，再准备自己的数据

建议 Python 3.11/3.12。按 README 创建环境并安装 requirements-toolkit.txt，
然后运行 `python adualnet.py serve`，用浏览器打开终端给出的本机地址。
默认进入“从一个视频开始”，先点击“体验人工核查演示”。它使用纯合成视频和
预设的演示骨架/回答，无需 GPU 或识别模型；请填写去标识编号、核查者、评分规则，
逐条选择回答正确性，核对轨迹和时窗，最后勾选两项确认并导出。
真实录像的 Ubuntu/本机设置与核查操作见 [单视频说明](video_workflow_zh.md)。

要体验模型统计报告，切换“模型演示 → 运行合成演示”，通常数分钟完成。
该示例同样不需要研究数据，分数仅验证软件，不是论文性能。
关闭网页不取消正在运行的识别或分析；重开页面可恢复本浏览器最近一次视频核查。

## 准备特征表

从 `data/templates/toolkit_features_template.csv` 开始；列说明见同目录的
`toolkit_feature_dictionary.csv`。CSV 与 XLSX 均可，一位参与者一行。

- `ID` 唯一且非空；同人多次录制不能当成不同参与者直接输入。
- `label` 必须每行有明确的 0 或 1，且两组均有样本；依据自己的研究判定标准填写。
- `age`、`sex`、`edu` 为人口学信息；sex 用数字编码并保留编码说明，edu 用年数。
- 15 个 gait 字段和 8 个 speech 字段来自同一 TUG-DT 条件。单任务数据另设研究协议。
- 允许局部缺失，在每个训练折中以中位数填补；整列缺失、缺少候选列或非数值会被提示。
- 填写测量版本、单位、任务起止、正确性规则。系统不从诊断选择提取阈值。

在“我的特征表”上传表格，填写工作表名称（例如 info），点击“检查输入”。
通过后填写测量版本和标签含义，再开始分析。所有输入留在本机。
输出路径为 local_workspace/runs/ 中独立的运行目录；新分析不会覆盖已有运行。

浏览器默认 5 折 × 10 次重复，四种模型、五类特征域组合和三个人口学子组合。
特征排序、缺失填补、缩放和拟合都在训练折内完成。检验对象是该队列中的
cognitive impairment status 分类表现；本工具不提供已验证的临床诊断器。

## 读结果

报告包含性能、配对 ΔAUROC、校准和人口学信息。XLSX 汇总相同结果。
95% 区间采用参与者级条件 bootstrap，保留同一人的所有重复预测。
它不完整涵盖模型重新训练的不确定性，也不保证外部人群性能；详见统计协议。
新工具按完整 OOF repeat 计算指标再平均，与旧稿件按 fold 平均存在统计量差异。
不能直接把新输出替换为论文原值。

逐人预测使用临时匿名编号，未导出原 ID 对照；但仍属参与者级本地研究数据。
不要把 local_workspace 整个上传 GitHub。HTML/XLSX 属汇总结果，分享前检查
你填写的研究说明是否适合公开。

## 视频识别与关键点

MMPose/WhisperX 应在各自兼容环境安装，CPU 特征分析环境不承担 GPU 依赖安装。
复制 config.example.ini 为 config.ini，配置本地输入、输出和 MMPose 命令模板。
记录实际解析的 pose/detector 配置、权重、阈值及版本；不要仅凭 human 别名
断言跨环境模型相同。可参考显式 bbox-thr 0.3 的示例，但必须依据自己的方案固定参数。

```bash
python adualnet.py pipeline --config config.ini --step gait-recognition --dry-run
python adualnet.py pipeline --config config.ini --step speech-recognition --dry-run
```

确认命令和路径后去掉 `--dry-run`。这些调用复用已有的识别包装脚本。
上述是保留的高级入口。新版单视频页面通过独立的 recognition.local.json
调用模型；已做 Ubuntu GPU 上公开图片短片的真实推理检查，不能由此推断临床准确性。

后续 gait-features / speech-features 是保留的提取入口。历史 gait 提取器中的
group 参数影响阈值，不能按待预测受试者的诊断切换它；后续提取器也未被证明
能再生论文的全部原始测量。旧 speech 自动处理含预期序列、计数和时间修正，
不应用于替代人工核查后的真实错误。使用之前阅读 manuscript_feature_provenance.md。
这版工具不把旧提取输出自动合并后直接送入分类器。

## 人工核查语音回答后提取

采用一套单独命名的新注释定义 `reviewed_response_onset_counts_v1`。
先检查 WhisperX 转录、参与者/检查者说话人、真实回答及 onset 时间；按研究协议
由人工填写每个回答的 is_correct（0/1）。这不决定具体正确性规则，研究者必须记录
是逐次减七、固定序列比对，以及如何处理重复/自我纠正。

任务表用 `data/templates/reviewed_speech_tasks.csv`：ID, task_start_s, task_end_s。
回答表用 `data/templates/reviewed_speech_responses.csv`：ID, response_onset_s,
response_text, is_correct。时间使用同一录制时间轴，单位秒。一个任务可没有回答行。
若无回答，ACC 与首答延迟记缺失，计数为 0。任务边界 [start,end)，前20秒边界
[start,min(start+20,end))。不得填入检查者提示。

```bash
python adualnet.py reviewed-speech --tasks tasks.csv --responses responses.csv --out reviewed_speech.csv
```

该命令不改写回答，不加入时间偏移，不设最低计数，不改变人工正确性判断。
输出八个 speech 特征及 protocol.json。它是透明的新工具定义，未认证与历史稿件
注释规则等价；请保留版本并做自己的质量检查。步态测量仍需核对跟踪对象、事件、
帧率和空间标定，不能仅通过重命名将像素量改成米。

## 历史复现与可选模块

已有 `run_pipeline.py`、论文建模、SHAP、回归、统计脚本保留。
历史论文表的已知测量和 CI 局限已在 provenance 和统计协议中记录。
公开小工具的默认演示不调用历史参与者表、不训练 PD 模型，也不加载冻结模型。
若研究需要 SHAP 或回归，先确认其输入/筛选范围，再依据旧入口显式运行。

## 出错和分享代码

依赖报错先确认激活了安装环境。输入报错先看 preflight.json 和页面列名提示。
失败运行保留 FAILED_STEP_README.txt，使用新目录重试。长时运算可查看运行目录旁
的日志；关页面可继续计算。当前服务器仅本机访问，适合个人研究电脑。

`python scripts/build_public_toolkit.py` 创建只含代码、说明、空模板和合成示例的 ZIP；
本地数据、运行结果、模型文件与 Git 历史不进入包。正式开源前由代码权利人选定
LICENSE、补齐论文引用和权重许可说明，再发布；工具不自动推送 GitHub。
