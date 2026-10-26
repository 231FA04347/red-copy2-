import os
import shutil
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, 'hel', 'data.yaml')
RUNS_DIR = os.path.join(BASE_DIR, 'runs', 'helmet')
MODEL_NAME = 'yolov8n.pt'  # nano for speed


def main():
    assert os.path.exists(DATA_YAML), f"data.yaml not found at {DATA_YAML}"

    # Train
    model = YOLO(MODEL_NAME)
    print(f"[train_helmet] Using dataset: {DATA_YAML}")
    results = model.train(
        data=DATA_YAML,
        epochs=1,            # ultra-quick run; increase later for accuracy
        imgsz=416,
        project=os.path.join(BASE_DIR, 'runs', 'helmet'),
        name='train',
        exist_ok=True,
        verbose=True,
        device='cpu',  # force CPU to avoid CUDA issues on some machines
        workers=0,      # Windows-friendly
        cache=True,
        batch=8,
        plots=False,
        fraction=0.25,
    )

    # Copy best or last weights to project root as helmet.pt
    best = os.path.join(RUNS_DIR, 'train', 'weights', 'best.pt')
    last = os.path.join(RUNS_DIR, 'train', 'weights', 'last.pt')
    out = os.path.join(BASE_DIR, 'helmet.pt')
    src = best if os.path.exists(best) else (last if os.path.exists(last) else None)
    if src:
        shutil.copyfile(src, out)
        print(f"[train_helmet] Saved helmet model to {out} (from {os.path.basename(src)})")
    else:
        raise FileNotFoundError(f"Weights not found at {best} or {last}")


if __name__ == '__main__':
    main()
