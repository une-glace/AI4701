# AI4701 Lab4: 视频螺丝计数分析

本项目当前采用 **YOLO 实例分割 + ByteTrack 跟踪 + 时序去重计数** 的方案。

当前仓库中：
- `video_track_count.py` 是单视频推理与计数主脚本。
- `run.py` 是按作业要求封装好的**一键运行入口**。
- `extract_frames.py` 用于视频抽帧与数据集准备。
- `train.py` 用于 YOLO 模型训练。

## 0. 获取 Ultralytics 源码

本项目依赖 Ultralytics 库。请在运行项目前，从 GitHub 克隆仓库并重命名为 `ultralytics_main`：

```bash
git clone https://github.com/ultralytics/ultralytics.git ultralytics_main
```

## 1. 项目结构

核心文件如下：

```text
bhw/
  run.py                    # 助教一键运行入口
  video_track_count.py      # 单视频检测、跟踪、计数主逻辑
  extract_frames.py         # 视频按间隔抽帧脚本
  train.py                  # YOLO 训练脚本
  README.md
  HOMEWORK_VIDEO_SCREW_COUNTING.md
  ultralytics_main/         # 克隆下来的 Ultralytics 源码 (需要手动获取)
  weights/
    best.pt                 # 训练好的模型权重
```

## 2. 环境配置

建议使用 Python 3.10 的 Anaconda 环境。

```bash
conda create -n cv python=3.10 -y
conda activate cv
pip install -r requirements.txt
```

说明：
- `requirements.txt` 中已经包含 `-e ./ultralytics_main`，安装时会自动以可编辑模式使用仓库里的本地 Ultralytics 源码。
- 如果没有可用 GPU，可以在运行时显式加 `--device cpu`。

## 3. 作业要求的一键运行

根据作业说明，助教应当可以直接运行：

```bash
python run.py --data_dir /path/to/test_videos_folder --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder/
```

如果你的权重不在默认位置，也可以额外指定：

```bash
python run.py --data_dir ./test_videos --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder --weights ./weights/best.pt
```

## 4. `run.py` 的工作流程

`run.py` 会完成以下步骤：

1. 遍历 `--data_dir` 下的所有视频文件。
2. 对每个视频调用 `video_track_count.py` 完成逐帧检测、跟踪和去重计数。
3. 从该视频保存的可视化帧中选取最接近中间位置的一张，复制为 `{video_name}_mask.png`。
4. 读取 `summary.txt` 中的分类计数结果。
5. 汇总所有视频结果并保存为 `result.npy`。
6. 记录总运行时间到 `time.txt`。

## 5. 输出说明

按作业要求，`run.py` 输出以下内容：

- `result.npy`
  - 通过 `numpy.load(..., allow_pickle=True).item()` 读取后是一个 `dict`
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
python video_track_count.py --source ./test_videos/IMG_2376.MOV --weights ./weights/best.pt 
```

常用参数：
- `--save-every 10`：每隔多少帧保存一张可视化图
- `--tracker bytetrack.yaml`：使用 ByteTrack 跟踪配置
- `--imgsz 1024`：推理分辨率

## 7. 当前实现说明

- 当前仓库主要包含**推理与计数**流程，以及我们新增加的**数据处理（抽帧）和训练脚本**。
- `video_track_count.py` 是核心计数逻辑。
- `run.py` 是为了满足作业要求而增加的批量封装入口。
- `extract_frames.py` 和 `train.py` 分别用于自定义数据集生成与 YOLO 模型训练。
- 请务必先通过 `git clone https://github.com/ultralytics/ultralytics.git ultralytics_main` 获取核心依赖源码后再执行环境配置。
