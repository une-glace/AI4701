# AI4701 Lab4: 视频螺丝计数分析

本仓库实现了基于 **YOLOv8-seg** (实例分割) 结合 **ByteTrack** (多目标跟踪) 的视频螺丝分类与计数系统。本方案能够高精度地解决传送带动态场景下由于螺丝遮挡、形变引起的重复计数问题，并能完美输出作业要求的掩膜 (Mask) 图像，具有极高的运行效率。

---

## 🛠️ 1. 环境配置

建议在 Anaconda 虚拟环境中运行：

```bash
# 创建并激活环境 (如果尚未创建)
conda create -n cv python=3.10
conda activate cv

# 安装核心依赖
pip install ultralytics opencv-python numpy tqdm
```

---

## 🚀 2. 训练流程 (Training Pipeline)

训练过程分为数据清洗、数据预处理和模型训练三个步骤。

### 2.1 (可选) 数据清洗
原始标注数据中可能包含由于位于画面边缘而截断的不完整螺丝。为了提高模型训练质量，我们提供了一个清洗脚本来过滤这些“残缺”数据。

```bash
python main.py
```
> **注意**：运行此脚本后，它会读取 `dataset/labels` 中的标注，并在 `dataset/filtered_labels` 中生成清洗后的标注。

### 2.2 数据预处理
原始数据集的类别 ID 为 `1~6`，而 YOLO 格式强制要求类别 ID 从 `0` 开始。此外，我们需要将数据集划分为训练集 (train) 和验证集 (val)。

```bash
python prepare_data.py
```
> **注意**：
> - 如果你跳过了 2.1 步，请确保 `prepare_data.py` 第12行为：`src_labels = Path(src_dir) / 'labels'`
> - 如果你执行了 2.1 步，请将该行改为：`src_labels = Path(src_dir) / 'filtered_labels'`
> 
> 该脚本运行后将生成符合 YOLO 标准的 `dataset_yolo` 文件夹。

### 2.3 启动模型训练
使用 Ultralytics 提供的命令行工具启动 YOLOv8-seg 的训练。我们使用的是 `yolov8n-seg.pt` (Nano) 模型以保证最快的推理速度。

```bash
yolo task=segment mode=train data=screw_data.yaml model=yolov8n-seg.pt epochs=50 imgsz=640 batch=16
```
训练完成后，最佳模型权重将保存在 `runs/segment/train/weights/best.pt`。

---

## 🎯 3. 推理与计数评估 (Inference & Counting)

作业要求的所有评估逻辑都封装在 `run.py` 中。该脚本将自动遍历目标文件夹中的所有视频，调用 YOLOv8-seg 获取实例掩膜，并结合内置的 ByteTrack 算法对每颗螺丝分配唯一 ID，最终通过集合 (Set) 实现去重计数。

### 3.1 一键运行测试

请确保你已经训练出了模型权重 (或将其放在了指定路径下)，然后执行：

```bash
python run.py --data_dir /path/to/test_videos_folder --output_path ./result.npy --output_time_path ./time.txt --mask_output_path ./mask_folder/ --weights runs/segment/train/weights/best.pt
```

### 3.2 脚本输出说明
按照作业要求，`run.py` 会产生以下输出：
1. **`result.npy`**: 包含每个视频中 5 种螺丝统计数量的 Python 字典文件。
2. **`time.txt`**: 记录处理所有视频所耗费的总时间（单位：秒）。
3. **`mask_folder/`**: 包含每个视频中间帧的定性分析掩膜图像（例如 `video_01_mask.png`），清晰标注了模型识别出的螺丝位置和边界。
