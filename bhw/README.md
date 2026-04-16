# AI4701 Lab4: 视频螺丝计数分析

本项目开发了一个工业视觉场景下的**多类别零件动态计数系统**，通过深度学习检测+多目标跟踪的融合方案实现视频实时螺丝自动分类与精准计数。系统采用**YOLO 实例分割**进行逐帧检测，**ByteTrack** 维持跨帧轨迹连续性，配合**光流运动估计**与**时序去重算法**消除重复计数，最终生成结构化计数结果和可视化输出。

<!-- 当前仓库中：
- `core/video_tracker.py` 是单视频推理与计数的核心逻辑模块。
- `run.py` 是按作业要求封装好的**一键运行入口**，通过导入 `core.video_tracker` 实现批量执行。
- `tools/extract_frames.py` 用于视频抽帧与数据集准备。
- `tools/train.py` 用于 YOLO 模型训练。 -->

## 0. 依赖说明

为便于评阅，仓库已包含 `ultralytics_main` 目录，无需额外从 GitHub clone。所有依赖由 `requirements.txt` 指定。

## 1. 项目结构

核心文件如下：

```text
bhw/
├── run.py                    # 助教一键运行入口
├── README.md
├── HOMEWORK_VIDEO_SCREW_COUNTING.md
├── SLIDES_OUTLINE.md         # 汇报 PPT 大纲说明
├── 消融实验指南.md             # 消融实验文档
├── core/
│   ├── tracker.py                   # 短脚本入口（单视频调试推荐）
│   ├── video_tracker.py             # 🔴 主编排模块
│   ├── video_tracker_motion.py      # 🟠 光流与运动估计模块
│   ├── video_tracker_visualization.py # 🟡 可视化渲染模块
│   │   ├─ 掩膜叠加（类别色彩编码）
│   │   ├─ 边界框与追踪 ID 标签
│   │   └─ 计数面板统计渲染
│   └── video_tracker_debug.py       # 🟢 日志与汇总模块
│       ├─ 帧级检测统计
│       └─ summary.txt 导出
├── tools/
│   ├── extract_frames.py     # 视频按间隔抽帧脚本
│   └── train.py              # YOLO 训练脚本
├── configs/
│   └── screw.yaml            # 数据集配置文件
├── dataset/                  # 训练与验证数据集 (YOLO 格式)
├── test_videos/              # 用于测试和演示的视频目录
├── ultralytics_main/         # 随作业一起打包的本地 Ultralytics 源码
├── mask_folder/              # run.py 自动生成的叠加结果掩码图
└── weights/
    └── best.pt               # 训练好的模型权重
```

## 2. 环境配置

建议使用 Python 3.10 的 Anaconda 环境。

```bash
conda create -n cv python=3.10 -y
conda activate cv
pip install -r requirements.txt
```

说明：
- `requirements.txt` 中已包含 `torch>=2.0.0` 依赖项（支持 GPU 硬件加速），以及 `-e ./ultralytics_main`。
- 如果没有可用 GPU，可以在运行命令时显式加上 `--device cpu`。

## 3. 作业要求的一键运行

根据作业说明，助教应当可以直接运行：

```bash
python run.py --data_dir /path/to/test_videos_folder --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder/
```

如需指定权重：

```bash
python run.py --data_dir ./test_videos --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder --weights /path/to/best.pt
```

## 4. `run.py` 的工作流程

`run.py` 会完成以下步骤：

1. 遍历 `--data_dir` 下的所有视频文件。
2. 对每个视频调用 `core.video_tracker.process_video` 完成逐帧检测、跟踪和去重计数。
3. 从该视频保存的可视化帧中选取最接近中间位置的一张，复制为 `{video_name}_mask.png`。
4. 读取 `summary.txt` 中的分类计数结果。
5. 汇总所有视频结果并保存为 `result.npy`。
6. 记录总运行时间到 `time.txt`。

## 5. 输出说明

按作业要求，`run.py` 输出以下内容：

- `result.npy`
  - 键为视频名（不带后缀）
  - 值为长度为 5 的列表，对应 5 类螺丝计数

- `time.txt`
  - 记录处理所有视频的总时间（秒）

- `mask_folder/{video_name}_mask.png`
  - 每个视频输出一张叠加图
  - 图中包含分割掩膜、检测框、跟踪标注和计数信息

## 6. 单视频调试

如果需要单独测试某一个视频，可以直接运行：

```bash
python core/tracker.py --source ./test_videos/IMG_2376.MOV --weights ./weights/best.pt 
```

常用参数：
- `--save-every 10`：每隔多少帧保存一张可视化图
- `--tracker bytetrack.yaml`：使用 ByteTrack 跟踪配置
- `--imgsz 1024`：推理分辨率

## 7. 团队分工与工时分配

本项目为**团队合作项目**，分工如下：

### 分工方案

| 成员 | 主要职责 | 代码贡献 | 实验贡献 |
|------|---------|---------|---------|
| **刘东隅（架构与追踪）** | 系统整体设计、追踪+推理模块代码、第一段视频标注 | video_tracker.py 主体逻辑、参数设计 |
| **李青雅（数据与可视化）** | 第二、三段视频标注、YOLO 模型训练优化、可视化实现 | video_tracker_visualization.py、YOLO 微调脚本 |
| **林嘉豪（集成与调优）** | 项目整合、端到端流程测试、参数调优 | run.py、测试框架、集成验证 |

