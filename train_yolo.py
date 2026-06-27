from ultralytics import YOLO
from pathlib import Path

BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "dataset_yolo"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("\n=== Entraînement YOLOv8 — Drainage Thoracique ===\n")

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data      = str(DATA_DIR),
        epochs    = 30,
        imgsz     = 224,
        batch     = 16,
        patience  = 10,
        save      = True,
        project   = str(MODEL_DIR),
        name      = "drainage_cls",
        exist_ok  = True,
        verbose   = True,
    )

    print("\n=== Entraînement terminé ===")
    print(f"Meilleur modèle : {MODEL_DIR}/drainage_cls/weights/best.pt")

    print("\n=== Évaluation sur le test set ===")
    metrics = model.val(data=str(DATA_DIR), split="test")
    print(f"Accuracy top-1 : {metrics.top1:.2%}")
    print(f"Accuracy top-5 : {metrics.top5:.2%}")

if __name__ == "__main__":
    main()