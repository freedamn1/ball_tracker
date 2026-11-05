# Ball Tracker

篮球检测与视频标注，默认使用 RF-DETR，可切换 YOLO26。支持视频、摄像头及图片序列，输出 MP4 和 CSV。默认只显示当帧检测；添加 `--kalman` 启用单篮球 Kalman 跟踪。

## 安装

需要 Python 3.10+：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

模型权重不随代码提交。将 RF-DETR 权重放到 `weights/rf-detr-basketball.pth`，将已有 YOLO26 微调权重放到 `weights/yolo26s_ball_fine_tuning.pt`，或用 `--weights` 指定路径。
RF-DETR 权重需保留类别名称信息，程序自动选择 `ball`，不混入 `ball-in-basket`。

## 检测

将视频或图片序列放入 `dataset/`：

```bash
# 默认 RF-DETR，自动选择 CUDA / CPU
python main.py --source dataset/basketball.mp4 --conf 0.5

# 图片序列；优先读取目录内或父目录 seqinfo.ini 的帧率，否则为 30 FPS
python main.py --source-type 1 --source dataset/sequence --conf 0.5

# 可选 YOLO26；使用官方权重时，不存在则自动下载
python main.py --detector yolo26 --weights weights/yolo26s.pt --source dataset/basketball.mp4

# 可选跟踪
python main.py --source dataset/basketball.mp4 --conf 0.5 --kalman
```

默认输出 `outputs/tracked.mp4` 和 `outputs/tracked.csv`，同名文件会覆盖。可用 `--output` 修改视频路径；CSV 与视频同名。
`--show` 开启预览，按 q 退出；`--device cpu` 强制 CPU，`--device cuda:0` 指定 GPU。
`--conf` 默认 0.25；`--rf-short-side` 默认 768，仅用于 RF-DETR，最长边限制 1333，不放大输入；`--imgsz` 默认 1280，仅用于 YOLO26。

CSV 帧号从 0 开始，坐标为原图像素。默认每个检测候选一行，未检出记录 `no_detection`。
启用 Kalman 后每帧记录跟踪状态：`detected`、`predicted` 或 `lost`。横向偏移相对画面中心，正值向右。跟踪不会驱动硬件。

## YOLO26 微调

`tune/` 仅保留独立训练入口。自行准备 YOLO 格式数据集，`data.yaml` 使用 `names: {0: ball}`，标签类别为 0，包含训练和验证划分：

```bash
python tune/train.py --data dataset/basketball/data.yaml --weights weights/yolo26s.pt --device 0
```

训练结果写入 `tune/runs/`，最佳权重复制到 `weights/yolo26s_ball_fine_tuning.pt`。数据、权重、训练日志和可视化结果均不提交。

## 文件

- `main.py`：读取输入、选择检测器、标注与输出。
- `config.py`：参数及共用检测结果类型。
- `rf_detr_detector.py`、`yolo26_detector.py`：两种检测器。
- `kalman_tracker.py`：可选跟踪，默认不启用。
- `tune/train.py`：YOLO26 单类别微调。
- `dataset/`、`outputs/`、`weights/`：本地数据、输出和权重目录。
