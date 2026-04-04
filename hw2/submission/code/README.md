# 螺丝计数作业说明

本项目使用基于 YOLO 的目标检测方案，对俯视图中的 5 类螺丝进行检测与计数。最终输出的类别顺序固定为：

- `Type_1`
- `Type_2`
- `Type_3`
- `Type_4`
- `Type_5`

## 1. 环境搭建

建议使用 `conda` 创建独立环境，环境名固定为 `cv`。

### 1.1 创建 conda 环境

```bash
conda create -n cv python=3.10 -y
conda activate cv
```

如果你的终端还不能直接使用 `conda activate`，可以先执行一次：

```bash
conda init powershell
```

然后重新打开 PowerShell 再激活环境。

### 1.2 安装依赖

先进入当前 `code` 目录，再安装依赖：

```bash
cd submission/code
pip install --upgrade pip
pip install -r requirements.txt
```

当前项目依赖的主要库包括：

- `numpy`
- `opencv-python`
- `pillow`
- `torch`
- `torchvision`
- `scikit-learn`
- `tqdm`
- `ultralytics`

### 1.3 环境说明

- 推荐 Python 版本：`3.10`
- 开发时主要在 Windows PowerShell 下测试
- 训练阶段建议使用支持 CUDA 的 NVIDIA GPU
- 推理阶段可以使用 CPU，但速度会更慢

如果 `torch` 或 `ultralytics` 安装失败，通常是本机 CUDA、Python 版本或网络源的问题。作业运行的最低要求是保证 `python run.py ...` 可以正常执行。

## 2. 提交目录要求

本目录应直接作为 `submission.zip` 中的 `code/` 目录提交，结构示例如下：

```text
submission.zip/
  code/
    run.py
    requirements.txt
    README.md
    src/
    scripts/
    models/
```

## 3. 模型权重位置

默认情况下，推理脚本会读取下面这个相对路径的 YOLO 权重：

```text
models/best.pt
```

如果你的权重文件不在这个位置，需要在运行时通过 `--checkpoint_path` 显式指定。

## 4. 作业运行方式

作业要求的主入口为：

```bash
python run.py --data_dir /path/to/test_images --output_path ./result.npy --output_time_path ./time.txt
```

更完整的示例命令如下：

```bash
python run.py \
  --data_dir /path/to/test_images \
  --output_path ./result.npy \
  --output_time_path ./time.txt \
  --checkpoint_path ./models/best.pt \
  --device auto \
  --imgsz 1280 \
  --conf 0.25 \
  --iou 0.7 \
  --debug_dir ./debug_vis
```

参数说明：

- `--data_dir`：测试图片所在文件夹
- `--output_path`：输出的 `result.npy` 路径
- `--output_time_path`：输出的总耗时文本文件路径
- `--checkpoint_path`：YOLO 权重路径，默认是 `models/best.pt`
- `--device`：推理设备，可选 `auto`、`cpu`、`cuda` 或 `0`
- `--imgsz`：推理时输入图像尺寸
- `--conf`：置信度阈值
- `--iou`：NMS 的 IoU 阈值
- `--debug_dir`：可选，用于保存可视化检测结果

## 5. 输出格式

- `result.npy`：使用 `numpy.load(..., allow_pickle=True).item()` 读取后，应为一个 Python 字典
- 字典的 `key`：图片文件名去掉后缀后的名称
- 字典的 `value`：长度为 5 的列表，顺序固定为 `[Type_1, Type_2, Type_3, Type_4, Type_5]`
- `time.txt`：记录处理全部测试图片所需的总时间，单位为秒

输出示例：

```python
{
    "img_001": [2, 0, 1, 4, 0],
    "img_002": [1, 1, 0, 5, 2]
}
```

## 6. 训练流程参考

如果需要从头训练自己的 YOLO 模型，可以参考下面的流程。

### 6.1 准备数据集

数据集目录应满足以下结构：

```text
dataset/
  images/
  labels/
  classes.txt
```

其中：

- `images/` 存放图片
- `labels/` 存放与图片同名的 YOLO 标注文件
- `classes.txt` 存放类别名，每行一个类别

### 6.2 划分训练集和验证集

```bash
python scripts/prepare_yolo_split.py --dataset_dir ./dataset --val_count 2
```

执行后会自动生成：

- `dataset/images/train`
- `dataset/images/val`
- `dataset/labels/train`
- `dataset/labels/val`
- `dataset/dataset.yaml`

### 6.3 训练 YOLO

```bash
python scripts/train_yolo.py \
  --data ./dataset/dataset.yaml \
  --model yolo11s.pt \
  --epochs 150 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

训练输出通常保存在：

```text
models/yolo_runs/<run_name>/
```

最佳权重一般位于：

```text
models/yolo_runs/<run_name>/weights/best.pt
```

### 6.4 放置最终权重

为了让作业入口直接运行，建议将最优权重复制为：

```text
models/best.pt
```

这样就可以直接执行：

```bash
python run.py --data_dir /path/to/test_images --output_path ./result.npy --output_time_path ./time.txt
```

## 7. 项目结构

```text
code/
  run.py
  README.md
  requirements.txt
  src/
    screw_counting/
      __init__.py
      yolo_pipeline.py
      pipeline.py
      detection.py
      dataset.py
      classifier.py
  scripts/
    prepare_crops.py
    prepare_yolo_split.py
    train_yolo.py
    predict_yolo.py
  dataset/
  models/
  data/
```

## 8. 补充说明

- 当前最终提交使用的是检测式方案，不是先裁剪再分类的方案
- `run.py` 会通过统计各类别检测框数量来生成最终计数结果
- 如果 `models/best.pt` 不存在，程序会直接报错并停止运行
- 仓库里还保留了一些早期探索代码，但主提交流程以 `run.py + YOLO 权重` 为准
