"""
test_yolo.py
Teste le modèle YOLOv8 sur une image et intègre avec l'agent.
Lancer : python test_yolo.py
"""

from ultralytics import YOLO
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "drainage_cls" / "weights" / "best.pt"

# Mapping classe YOLO → operation_id
CLASS_TO_OPERATION = {
    "Diagnostique":                1,
    "Preparation patient":         2,
    "Desinfection Betadine":       3,
    "Anesthesie locale":           4,
    "Incision cutanee":            5,
    "Dissection sous-cutanee":     6,
    "Insertion du drain":          7,
    "Connexion bocal de drainage": 8,
}


def predict_operation(image_path: str) -> dict:
    """
    Prédit l'étape chirurgicale à partir d'une image.
    Retourne : operation_id, operation_name, confidence
    """
    model   = YOLO(str(MODEL_PATH))
    results = model.predict(image_path, verbose=False)
    result  = results[0]

    # Classe prédite
    top1_idx  = result.probs.top1
    top1_conf = float(result.probs.top1conf)
    class_name = result.names[top1_idx]

    # Top 3 prédictions
    top3 = []
    for i in result.probs.top5[:3]:
        top3.append({
            "class":      result.names[i],
            "confidence": float(result.probs.data[i]),
            "operation":  CLASS_TO_OPERATION.get(result.names[i])
        })

    return {
        "operation_id":   CLASS_TO_OPERATION.get(class_name, 0),
        "operation_name": class_name,
        "confidence":     top1_conf,
        "top3":           top3,
    }


def main():
    print("\n=== TEST YOLO — Drainage Thoracique ===\n")

    # Vérifier le modèle
    if not MODEL_PATH.exists():
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
        print("Lance d'abord : python train_yolo.py")
        return

    print(f"✅ Modèle chargé : {MODEL_PATH}")
    print()

    # Cherche des images de test dans dataset_yolo/test/
    test_dir = BASE_DIR / "dataset_yolo" / "test"
    test_images = []

    for cls_dir in sorted(test_dir.iterdir()):
        if cls_dir.is_dir():
            imgs = list(cls_dir.glob("*.jpg"))[:1]  # 1 image par classe
            test_images.extend(imgs)

    if not test_images:
        print("❌ Aucune image de test trouvée")
        return

    print(f"Test sur {len(test_images)} images (1 par classe)\n")
    print(f"{'Image':<45} {'Prédit':<30} {'Réel':<30} {'Conf':>6} {'OK':>4}")
    print("-" * 120)

    correct = 0
    for img_path in test_images:
        real_class = img_path.parent.name
        result     = predict_operation(str(img_path))
        predicted  = result["operation_name"]
        confidence = result["confidence"]
        ok         = "✅" if predicted == real_class else "❌"
        if predicted == real_class:
            correct += 1
        print(f"{img_path.name:<45} {predicted:<30} {real_class:<30} {confidence:>5.1%} {ok:>4}")

    print("-" * 120)
    print(f"\nAccuracy : {correct}/{len(test_images)} = {correct/len(test_images):.1%}")

    # Test interactif
    print("\n=== TEST INTERACTIF ===")
    print("Entre le chemin d'une image pour la tester")
    print("(ou appuie sur Entrée pour quitter)\n")

    while True:
        img_input = input("Image : ").strip().strip('"')
        if not img_input:
            break

        p = Path(img_input)
        if not p.exists():
            print(f"  ❌ Image introuvable : {p}")
            continue

        result = predict_operation(str(p))
        print(f"\n  Étape détectée : {result['operation_name']}")
        print(f"  ID opération  : {result['operation_id']}")
        print(f"  Confiance     : {result['confidence']:.1%}")
        print(f"  Top 3 :")
        for i, t in enumerate(result["top3"]):
            print(f"    {i+1}. {t['class']:<30} {t['confidence']:.1%}")
        print()


if __name__ == "__main__":
    main()