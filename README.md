# Ball Tracker

篮球检测与视频标注，默认使用 RF-DETR，可切换 YOLO26。仅支持手机 RTSP 输入，独立线程采集和检测，双窗口预览，输出 MP4 和 CSV。默认只显示当帧检测；添加 `--kalman` 启用单篮球 Kalman 跟踪。

## 安装

需要 Python 3.10+ 和 Git LFS。先安装 Git LFS，再克隆项目以自动获取两个模型权重：

```bash
git lfs install
git clone https://github.com/freedamn1/ball_tracker.git
cd ball_tracker
```

如果已经克隆项目，运行 `git lfs install` 和 `git lfs pull` 获取权重。然后安装 Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

以下两个权重由 Git LFS 管理，克隆或执行 `git lfs pull` 后保存到：

- `weights/rf-detr-basketball.pth`：默认 RF-DETR 篮球模型。
- `weights/yolo26s_ball_fine_tuning.pt`：YOLO26 篮球微调模型。

也可用 `--weights` 指定其他本地权重路径。模型文件应为真实权重，不能使用未下载的 LFS 文本指针。
RF-DETR 权重需保留类别名称信息，程序自动选择 `ball`，不混入 `ball-in-basket`。

## 检测

手机和电脑接入同一局域网，在手机 App 中开启 RTSP 服务，使用其提供的完整地址：

```bash
# 默认 RF-DETR，不主动跳帧
python main.py --rtsp-url "rtsp://192.168.1.100:8554/live" --conf 0.5

# 每 5 帧选取一次预测
python main.py --rtsp-url "rtsp://192.168.1.100:8554/live" --frame-step 5

# YOLO26，可选 Kalman 跟踪
python main.py --rtsp-url "rtsp://192.168.1.100:8554/live" --detector yolo26 --kalman
```

也可通过环境变量 RTSP_URL 提供地址。仅保留 RTSP 输入，移除 --source、--source-type、--show。
RTSP 固定使用 TCP、5 秒连接/读取超时、连续失败最多重试 3 次、间隔 1 秒。
先收到 RTSP 画面，再加载模型；加载和推理期间输入窗口持续刷新。
默认同时开启原始输入和预测结果窗口，按 Q / Esc 或关闭任一窗口退出。

--frame-step 0 表示所有帧均可送检；5 表示选取第 5、10、15……帧。
采集和推理各使用独立线程，待预测任务仅保留最新一帧；检测跟不上时会丢弃旧任务，即使设置为 0 也不保证每帧均被检测。
预测窗口显示最近完成预测的对应画面，不把旧检测框画到新输入帧上。
每次预测记录开始、结束时间与检测器调用耗时（含预处理、推理、结果转换），日志和预测窗口显示延迟。
每 10 秒及退出时汇报累计连接时长，重连等待不计入；故障在读取失败时判定。
退出时窗口立即关闭，后台等待当前读取/模型操作结束后释放资源。

默认输出 outputs/tracked.mp4 和 outputs/tracked.csv，同名文件会覆盖，--output 可修改路径。
MP4 仅保存完成预测的帧，以固定 30 FPS 编码，属于预测结果序列，播放速度不代表现场实时速度。
CSV 的 frame 为从 0 开始的实际接收帧序号，time_s 为相对首次收到画面的单调时钟时间；跳过的帧不写入。
inference_start_s、inference_end_s 使用相同时间原点，inference_ms 为两者之差。
每个检测候选一行，未检出记录 no_detection；Kalman 使用相邻预测帧接收时间差更新，missing 按预测次数计算，重连或输入尺寸改变后重置跟踪。
输入尺寸变化时输出视频缩放至首次预测尺寸，CSV 坐标仍为当帧原图像素。
--device cpu 强制 CPU；--device cuda:0 指定 GPU。--conf 默认 0.25；--rf-short-side 默认 768；--imgsz 默认 1280。

## YOLO26 微调

`tune/` 仅保留独立训练入口。自行准备 YOLO 格式数据集，`data.yaml` 使用 `names: {0: ball}`，标签类别为 0，包含训练和验证划分：

```bash
python tune/train.py --data dataset/basketball/data.yaml --weights weights/yolo26s.pt --device 0
```

训练结果写入 `tune/runs/`，最佳权重复制到 `weights/yolo26s_ball_fine_tuning.pt`。数据、训练日志、可视化结果和其他权重均忽略；上述两个模型通过 Git LFS 提交。

## 文件

- `main.py`：RTSP 启动与双窗口显示。
- `rtsp_input.py`：独立采集和最新帧交换。
- `inference_worker.py`：独立推理、标注及结果输出。
- `config.py`：参数及共用检测结果类型。
- `rf_detr_detector.py`、`yolo26_detector.py`：两种检测器。
- `kalman_tracker.py`：可选跟踪，默认不启用。
- `tune/train.py`：YOLO26 单类别微调。
- `dataset/`、`outputs/`、`weights/`：本地数据、输出和权重目录。
