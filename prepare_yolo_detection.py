"""
prepare_yolo_detection.py
Convertit les annotations CVAT (YOLO format) en dataset YOLO Detection.
Les images viennent de data/raw/video1..3/
Les labels viennent de data/raw/detection_export/video1..3_detection/
Lancer : python prepare_yolo_detection.py
"""

import shutil
import random
from pathlib import Path

BASE_DIR      = Path(__file__).parent
EXPORT_DIR    = BASE_DIR / "data" / "raw" / "detection_export"
RAW_DIR       = BASE_DIR / "data" / "raw"
YOLO_DET_DIR  = BASE_DIR / "dataset_yolo_detection"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# Mapping vidéo → dossier source images
VIDEO_MAP = {
    "video1_detection": "video1",
    "video2_detection": "video2",
    "video3_detection": "video3",
}

# Classes (ordre exact de obj.names)
CLASSES = [
    "stethoscope",
    "masque_respiratoire",
    "gants",
    "champ_sterile",
    "compresse_sterile",
    "betadine",
    "seringue",
    "xylocaine",
    "lame_bistouri",
    "ciseau",
    "drain",
    "systeme_drainage",
]


def main():
    print("\n=== Préparation dataset YOLO Detection ===\n")

    # Créer les dossiers
    for split in ["train", "val", "test"]:
        (YOLO_DET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Collecter toutes les paires (image, label)
    all_pairs = []

    for det_folder, video_folder in VIDEO_MAP.items():
        det_dir   = EXPORT_DIR / det_folder / "obj_train_data"
        img_dir   = RAW_DIR / video_folder

        if not det_dir.exists():
            print(f"  [WARN] {det_dir} introuvable, ignoré.")
            continue

        txt_files = list(det_dir.glob("*.txt"))
        print(f"-> {det_folder} : {len(txt_files)} annotations")

        for txt_path in txt_files:
            # Cherche l'image correspondante
            frame_name = txt_path.stem  # ex: frame_00000
            img_path   = None

            for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                candidate = img_dir / (frame_name + ext)
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Vérifie que le txt n'est pas vide
            content = txt_path.read_text().strip()
            if not content:
                continue

            all_pairs.append((img_path, txt_path))

    print(f"\nTotal paires valides : {len(all_pairs)}")

    # Mélanger et diviser
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)

    n       = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": all_pairs[:n_train],
        "val":   all_pairs[n_train:n_train + n_val],
        "test":  all_pairs[n_train + n_val:],
    }

    # Copier les fichiers
    for split_name, pairs in splits.items():
        for i, (img_path, txt_path) in enumerate(pairs):
            ext      = img_path.suffix
            new_name = f"frame_{i:06d}"

            # Image
            dest_img = YOLO_DET_DIR / "images" / split_name / (new_name + ext)
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

            # Label
            dest_txt = YOLO_DET_DIR / "labels" / split_name / (new_name + ".txt")
            if not dest_txt.exists():
                shutil.copy2(txt_path, dest_txt)

        print(f"  {split_name:5} : {len(pairs)} images")

    # Générer data.yaml
    yaml_content = f"""# YOLOv8 Detection Dataset — Drainage Thoracique Tools
path: {YOLO_DET_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = YOLO_DET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n OK Dataset YOLO Detection : {YOLO_DET_DIR.resolve()}")
    print(f" OK data.yaml              : {yaml_path}")
    print(f"\n Structure :")
    print(f"   train : {len(splits['train'])} images")
    print(f"   val   : {len(splits['val'])} images")
    print(f"   test  : {len(splits['test'])} images")
    print(f"\n Prêt pour l'entraînement YOLO Detection !")


if __name__ == "__main__":
    main()