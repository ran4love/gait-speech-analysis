# 从一个视频到人工确认的研究特征

版本：ADualNet 0.3.0。目标是让研究者体验同类 TUG-DT 范式，不是提供临床诊断器，
也不是证明原始录像能自动再生论文的全部特征和性能。

页面右上角可用 **English / 中文** 切换显示，初次默认英文，并在本浏览器记住选择。
切换不重载页面、不清空尚未保存的编辑、不改变核查确认或计算结果。转录原文、评分
规则、备注、ID、原始日志与导出数据保留原文；历史汇总报告保持其原有语言。

新版为深蓝导航与白色操作区。左侧保留视频、模型演示、特征表和高级入口；顶部流程
状态跟随当前视频任务，不代表算法准确性或机构认证。“前往人工核查”仅在已有核查
结果可供审阅后启用。原始处理日志默认折叠，可按需展开。

## 普通使用者：六步

1. 按 README 安装 CPU 工具环境及 FFmpeg/ffprobe，运行 `python adualnet.py serve`。
2. 打开终端显示的本机地址，先点“体验人工核查演示”，无需安装识别模型。
3. 真实使用先点“检查识别环境”，选择含同步语音的视频，确认研究授权及远程传输/保留说明，再开始识别。
4. 回看骨架：选择参与者轨迹，分别确定语音任务与步态分析的起止秒。多人、遮挡及转身必须重点检查。
5. 核对转录：一个词未必等于一次回答。排除检查者提示，增删或拆并回答行，修正起始时间和文字，逐条标记正确/错误，并记录评分规则。
6. 填写本地去标识编号和核查者代码，勾选步态与语音两项确认，再点“确认并生成特征”。下载 XLSX、CSV、QC 与 provenance。

任何编辑都会清除页面确认勾选；未确认、未逐条评分或任务时窗无效时不能导出。
可以先“保存核查草稿”，每次保存创建独立修订。旧导出不会因新草稿被改写。
完整转录文字仅作为审计注释；统计计数使用下方逐条回答表，而非自由文本。
一次任务无回答是允许的：计数为0，ACC和首答延迟缺失，不能伪造最低回答数。

## 环境管理员：推荐 Ubuntu 识别，Mac 审阅

Mac/client 环境只需工具依赖和 FFmpeg/ffprobe。Ubuntu 分别使用已兼容的
MMPose 和 faster-whisper/WhisperX 环境，模型不必装在同一个 Python 中。
识别适配器兼容 Python 3.8；faster-whisper 自身需要其支持的 Python 版本。

```bash
cp recognition.ssh.example.json recognition.local.json
# 修改本地配置；不要写入密码
python adualnet.py serve --recognition-config recognition.local.json
```

需要修改的字段：

- `execution.ssh_argv`：自己的 SSH host/alias；用密钥或 SSH agent，先在终端确认主机身份并能无交互登录。
- `execution.remote_root`：服务器上新任务的专用目录，默认用户缓存下相对路径；需要读写权限。
- `mmpose.python`、`whisper.python`：服务器两个环境的绝对 Python 路径，不用 `~`。
- `mmpose.pose2d`：支持 COCO-17 的配置路径/别名；`human` 是可解析别名，不保证跨版本模型一致。
- `pose2d_weights`、`det_model`、`det_weights`：建议固定配置与本地权重路径；指定det_weights必须同时指定det_model，否则MMPose默认分支可能忽略该权重，工具会提前报错。`bbox_thr` 默认为0.3，仅为检测框筛选阈值，不是关键点阈值。
- `whisper.backend/model/language/device/compute_type`：例如 faster-whisper / large-v2 / zh / cuda / float16；可用本地模型路径以固定版本。
- `analysis_fps`：默认30；会重编码出统一 CFR 审阅视频，并保留源视频和探测信息，不能与源帧编号混用。

本机识别则复制 recognition.example.json，安装兼容的 MMPose/MMCV/PyTorch 组合，
并按 requirements-video.txt 安装可选语音与视频依赖。依赖存在检查不保证 CUDA/权重
可用，第一次必须做无敏感资料的真实推理 smoke test。模型首次下载可能访问模型仓库。
本工具的 Whisper 适配器不传入预期减七序列，不自动纠错或替研究者判断正确性。

## 实际处理与输出

```text
同步视频 -> 本机 FFmpeg：方向归一化、CFR review.mp4、单声道16kHz音频
         -> 本机或 SSH：MMPose 全帧/全人物 JSON + Whisper 原始转写
         -> 本机浏览器：人物/关键点/时窗/回答/正确性人工修订
         -> 两项人工确认 -> 单独版本的特征 XLSX/CSV、QC、溯源
```

两种识别均使用 review.mp4 时间轴；音频从该视频提取，禁止把独立文件的时钟直接拼接。
IoU 人物关联只是启发式，可碎裂或换人。页面支持当前帧轨迹覆盖、COCO-17 坐标修订、
排除/恢复当前帧。修订只写审阅层，原始输出不改。排除帧仍保留时间槽，不能删除后压缩时长。
输入上限512MB、默认600秒；不适合在此轻量服务中批量上传长录像。

默认目录 `local_workspace/video_jobs/<job_id>/`：

- `source.*`、`review.mp4`、`audio.wav`：原始视频、归一化视频、识别音频。
- `media.json`：原始/归一化探测元数据、源文件散列、分析帧率与时间轴。
- `pose_raw.json`、`transcript_raw.json`：原始识别；`poses.json` 增加审阅用启发式轨迹编号。
- `recognition_preflight.json`、`recognition_config.json`、`run.log`：环境、实际配置及进度；失败保留 FAILED_STEP_README.txt。
- `reviews/review_NNNN.json`：带核查者、时窗、修订和评分规则的不可覆盖记录。
- `features/revision_NNNN/`：reviewed_features.xlsx/csv、feature_values.csv、gait_qc.json、provenance.json及人工注释CSV。
- `remote_receipt.json`：SSH传输/远程副本说明。模型适配器还记录解析配置、版本；显式本地姿态权重记录SHA256。

CLI也可准备识别；要从网页审阅，请优先使用网页上传以创建可发现的任务编号：

```bash
python adualnet.py video --input sample.mp4 --out local_workspace/video_case_v1 \
  --recognition-config recognition.local.json
```

## 必须理解的测量限制

复用现有 `gaitfeature.calculate_gait_parameters` 的踝部事件算法，不改写旧代码和结果。
双踝原始有效覆盖需>=80%，最长连续无效段不能超过10帧；这只是技术检查，非临床认证。
MMPose/SimCC输出分数不是概率，可略高于1，原值保留；算法仍使用既有分数阈值。
人工修改的点在审阅层作为已核查点参与计算，原始分数不变。

- 像素位移受视角、透视和镜头畸变影响。未做相机/平面标定，不输出米制步长。
- walking_speed只有填写所选时窗的实测路线长度后才计算，是该路线长度/整个选定时窗时长；含停顿或转身时不等同于直线行走速度。
- 启发式踝部事件不是经验证的足接触。step length/time=stride/2为算法定义；step与stride asymmetry也非独立结果，不能作为物理一致性的独立证据。
- stance_times、swing_times、double_support_times保持缺失并标注原因，禁止用诊断特异比例或常数伪造。
- 名称沿用既有候选列，但本测量版本和论文历史表不自动等价；短片测试不验证生理准确度或重复性。
- 单视频导出不自动接分类模型。人口学和结局也不由视频推断；完整队列建模需独立协议、合格测量及人工整理，不以中位数填补掩盖整域不适用。

## 隐私、保留与发布

这是单研究者本机工具：仅监听127.0.0.1，有会话令牌和同源检查，不能公开转发端口。
不是多用户服务、合规存储系统或自动说话人身份认证。视频、音频、转写和逐人特征都敏感。
SSH模式将归一化视频、音频、模型设置与适配器发送到你配置的服务器，两端均保留副本，
不自动删除；研究者需按机构政策处置。失败中间件同样应保留/清理于机构控制下。
公开源码构建器只打包允许清单，不包含本地配置、工作区、模型或Git历史。
不要上传整个工作目录、截图中的真实参与者资料或原始日志。许可证未定前只称发布候选，
正式GitHub发布另行确认许可证和第三方权重使用条款。
