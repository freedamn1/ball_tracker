"""从现有 YOLO26 权重微调 ball 单类别检测模型。"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="已准备好的 ball 单类别 YOLO data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/yolo26s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0", help="0 / 0,1 / cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name", default="yolo26s_ball")
    args = parser.parse_args()
    if (min(args.epochs, args.imgsz, args.batch) <= 0
            or args.workers < 0 or args.patience < 0):
        parser.error("epochs、imgsz、batch 必须为正；workers、patience 不能为负")
    if not args.data.is_file():
        parser.error(f"数据文件不存在：{args.data}")
    if not args.weights.is_file():
        parser.error(f"初始权重不存在：{args.weights}")
    return args


def main():
    args = parse_args()
    # 与预测入口一致：先加载同一套 cuDNN，再导入 Ultralytics/PyTorch。
    # 无需手动设置环境变量，或创建额外的配置文件。
    import ctypes
    import sys
    import sysconfig

    cudnn_libraries = []
    if sys.platform.startswith("linux") and args.device != "cpu":
        for location in dict.fromkeys(sysconfig.get_path(key) for key in ("purelib", "platlib")):
            library_dir = Path(location) / "nvidia" / "cudnn" / "lib"
            graph = library_dir / "libcudnn_graph.so.9"
            if not graph.is_file():
                continue
            # adv/cnn 依赖 ops；必须先加载本环境的 ops，否则会抢先引用系统旧版。
            dependencies = [graph, library_dir / "libcudnn_ops.so.9"]
            libraries = dependencies + sorted(p for p in library_dir.glob("libcudnn*.so.9")
                                              if p not in dependencies)
            for library in libraries:
                try:
                    cudnn_libraries.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))
                except OSError as exc:
                    raise RuntimeError(f"无法加载虚拟环境 cuDNN 库 {library}：{exc}") from exc
            for handle, library in zip(cudnn_libraries, libraries):
                checks = {"libcudnn_graph.so.9": "cudnnGraphVersionCheck",
                          "libcudnn_ops.so.9": "cudnnOpsVersionCheck",
                          "libcudnn_cnn.so.9": "cudnnCnnVersionCheck",
                          "libcudnn_adv.so.9": "cudnnAdvVersionCheck"}
                if library.name in checks:
                    check = getattr(handle, checks[library.name])
                    check.argtypes, check.restype = [], ctypes.c_int
                    status = check()
                    if status != 0:
                        raise RuntimeError(f"cuDNN 子库版本检查失败：{library.name}，状态码 {status}")
            print(f"已加载当前 Python 环境的 cuDNN：{library_dir}", flush=True)
            break
    import yaml
    data = args.data.resolve()
    if data.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("--data 必须是 YOLO data.yaml")
    metadata = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = metadata.get("names")
    if names not in (["ball"], {0: "ball"}):
        raise ValueError("请准备 names: {0: ball} 的单类别数据集及对应标注")
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    if model.task != "detect" or not Path(model.ckpt_path).stem.startswith("yolo26"):
        raise ValueError("请使用 YOLO26 检测预训练权重。")
    # 单类别 data.yaml 让 Ultralytics 将检测头调整为 nc=1，并迁移兼容的预训练参数。
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, patience=args.patience,
        project=str(ROOT / "tune/runs"), name=args.name,
        seed=0, pretrained=True, save=True, plots=True, val=True,
    )
    print(f"训练输出目录：{model.trainer.save_dir}", flush=True)
    import shutil
    destination = ROOT / "weights/yolo26s_ball_fine_tuning.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(model.trainer.save_dir) / "weights/best.pt", destination)
    print(f"最佳权重：{destination}", flush=True)


if __name__ == "__main__":
    main()
