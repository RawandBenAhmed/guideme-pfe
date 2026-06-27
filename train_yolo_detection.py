
"""
train_yolo_detection.py
Entraîne YOLOv8 Detection pour détecter les outils chirurgicaux.
Avec augmentation maximale pour meilleure généralisation.
Lancer : python train_yolo_detection.py
"""

from ultralytics import YOLO
from pathlib import Path

BASE_DIR  = Path(__file__).parent
DATA_YAML = BASE_DIR / "dataset_yolo_detection" / "data.yaml"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print("\n=== Entraînement YOLO Detection — Outils Chirurgicaux ===\n")

    if not DATA_YAML.exists():
        print(f"❌ data.yaml introuvable : {DATA_YAML}")
        print("Lance d'abord : python prepare_yolo_detection.py")
        return

    model = YOLO("yolov8n.pt")

    print(f"Dataset : {DATA_YAML}")
    print(f"Classes : 12 outils chirurgicaux")
    print(f"Modèle  : yolov8n (nano detection)")
    print(f"Epochs  : 100 avec augmentation maximale")
    print()

    results = model.train(
        data     = str(DATA_YAML),
        epochs   = 100,
        imgsz    = 640,
        batch    = 8,
        patience = 20,
        degrees  = 15.0,
        scale    = 0.5,
        fliplr   = 0.5,
        hsv_h    = 0.02,
        hsv_s    = 0.8,
        hsv_v    = 0.5,
        mosaic   = 1.0,
        mixup    = 0.1,
        save     = True,
        project  = str(MODEL_DIR),
        name     = "drainage_detection",
        exist_ok = True,
        verbose  = True,
    )

    print("\n=== Entraînement terminé ===")
    print(f"Meilleur modèle : {MODEL_DIR}/drainage_detection/weights/best.pt")

    print("\n=== Évaluation sur le test set ===")
    metrics = model.val(data=str(DATA_YAML), split="test")
    print(f"mAP50   : {metrics.box.map50:.2%}")
    print(f"mAP50-95: {metrics.box.map:.2%}")


if __name__ == "__main__":
    main()