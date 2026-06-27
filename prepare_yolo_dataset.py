"""
prepare_yolo_dataset.py
Convertit le dataset CSV en format YOLOv8 classification.
Structure : dataset_yolo/train|val|test/<classe>/image.jpg
Lancer : python prepare_yolo_dataset.py
"""

import shutil
import random
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
BASE_DIR   = Path(__file__).parent
CSV_PATH   = BASE_DIR / "dataset_output" / "dataset_drainage_thoracique.csv"
YOLO_DIR   = BASE_DIR / "dataset_yolo"

# Ratio train/val/test
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

# Classes YOLO — ordre fixe important
CLASSES = [
    "Diagnostique",
    "Preparation patient",
    "Desinfection Betadine",
    "Anesthesie locale",
    "Incision cutanee",
    "Dissection sous-cutanee",
    "Insertion du drain",
    "Connexion bocal de drainage",
]

# ============================================================
# PIPELINE
# ============================================================

def main():
    print("\n=== Préparation dataset YOLO ===\n")

    # Charger CSV
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.fillna("")

    # Garder uniquement les lignes avec une image d'étape valide
    df = df[
        (df["operation_image"] != "") &
        (df["source"] != "tools_classification")
    ].reset_index(drop=True)

    print(f"Lignes valides : {len(df)}")
    print(f"Classes : {df['operation_name'].nunique()}")
    print()

    # Créer les dossiers
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            cls_dir = YOLO_DIR / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)

    # Traiter chaque classe
    total_train = total_val = total_test = 0
    random.seed(RANDOM_SEED)

    print(f"{'Classe':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 65)

    for cls in CLASSES:
        rows = df[df["operation_name"] == cls].to_dict(orient="records")

        # Filtrer les images qui existent vraiment
        valid = [r for r in rows if Path(r["operation_image"]).exists()]

        if not valid:
            print(f"{cls:<35} {'AUCUNE IMAGE':>25}")
            continue

        # Mélanger aléatoirement
        random.shuffle(valid)

        # Split
        n       = len(valid)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        n_test  = n - n_train - n_val

        splits = {
            "train": valid[:n_train],
            "val":   valid[n_train:n_train + n_val],
            "test":  valid[n_train + n_val:],
        }

        # Copier les images
        for split_name, split_rows in splits.items():
            dest_dir = YOLO_DIR / split_name / cls
            for i, row in enumerate(split_rows):
                src  = Path(row["operation_image"])
                ext  = src.suffix
                dest = dest_dir / f"{cls.replace(' ', '_')}_{i:05d}{ext}"
                if not dest.exists():
                    shutil.copy2(src, dest)

        total_train += n_train
        total_val   += n_val
        total_test  += n_test
        print(f"{cls:<35} {n_train:>6} {n_val:>6} {n_test:>6} {n:>7}")

    print("-" * 65)
    print(f"{'TOTAL':<35} {total_train:>6} {total_val:>6} {total_test:>6} {total_train+total_val+total_test:>7}")

    # Générer data.yaml
    yaml_content = f"""# YOLOv8 Classification Dataset — Drainage Thoracique
path: {YOLO_DIR.resolve()}
train: train
val: val
test: test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n OK Dataset YOLO créé : {YOLO_DIR.resolve()}")
    print(f" OK data.yaml         : {yaml_path}")
    print(f"\n Structure :")
    print(f"   train : {total_train} images")
    print(f"   val   : {total_val} images")
    print(f"   test  : {total_test} images")
    print(f"\n Prêt pour l'entraînement YOLOv8 !")


if __name__ == "__main__":
    main()