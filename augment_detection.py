"""
augment_detection.py
Augmente le dataset YOLO Detection x3 (blur, zoom, bright).
Copie et ajuste les fichiers .txt pour chaque augmentation.
Lancer : python augment_detection.py
"""

import shutil
import random
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

BASE_DIR = Path(__file__).parent
DET_DIR  = BASE_DIR / "dataset_yolo_detection"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# TRANSFORMATIONS IMAGE
# ============================================================

def aug_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def aug_bright(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.4)


def aug_zoom(img: Image.Image, margin: float = 0.08) -> Image.Image:
    w, h = img.size
    mw = int(w * margin)
    mh = int(h * margin)
    cropped = img.crop((mw, mh, w - mw, h - mh))
    return cropped.resize((w, h), Image.LANCZOS)


# ============================================================
# AJUSTEMENT DES BOUNDING BOXES POUR LE ZOOM
# ============================================================

def adjust_boxes_zoom(txt_content: str, margin: float = 0.08) -> str:
    """
    Recalcule les coordonnées YOLO après crop/zoom.
    Format YOLO : class x_center y_center width height (normalisé 0-1)
    """
    lines = txt_content.strip().splitlines()
    new_lines = []
    crop_size = 1.0 - 2 * margin  # taille de la zone croppée

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            new_lines.append(line)
            continue

        cls = parts[0]
        x, y, w, h = map(float, parts[1:])

        # Recalcule dans le repère croppé
        x_new = (x - margin) / crop_size
        y_new = (y - margin) / crop_size
        w_new = w / crop_size
        h_new = h / crop_size

        # Clamp entre 0 et 1
        x_new = max(0.0, min(1.0, x_new))
        y_new = max(0.0, min(1.0, y_new))
        w_new = max(0.0, min(1.0, w_new))
        h_new = max(0.0, min(1.0, h_new))

        # Ignore si la boîte est hors champ après zoom
        if x_new + w_new/2 > 1.0 or x_new - w_new/2 < 0.0:
            continue
        if y_new + h_new/2 > 1.0 or y_new - h_new/2 < 0.0:
            continue

        new_lines.append(f"{cls} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}")

    return "\n".join(new_lines)


# ============================================================
# AUGMENTATIONS DÉFINIES
# ============================================================

AUGMENTATIONS = [
    ("blur",   aug_blur,   False),  # (suffix, fn_image, recalc_boxes)
    ("bright", aug_bright, False),
    ("zoom",   aug_zoom,   True),
]


# ============================================================
# PIPELINE
# ============================================================

def augment_split(split: str):
    img_dir = DET_DIR / "images" / split
    lbl_dir = DET_DIR / "labels" / split

    if not img_dir.exists():
        print(f"  [SKIP] {split} introuvable")
        return 0

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + \
             list(img_dir.glob("*.png"))

    # Exclure les images déjà augmentées
    images = [p for p in images if not any(
        p.stem.endswith(f"_{s}") for s, _, _ in AUGMENTATIONS
    )]

    print(f"  {split} : {len(images)} images originales")
    new_count = 0

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            continue

        txt_content = lbl_path.read_text(encoding="utf-8").strip()
        if not txt_content:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    [WARN] {img_path.name} : {e}")
            continue

        for suffix, fn_img, recalc in AUGMENTATIONS:
            new_img_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
            new_lbl_name = f"{img_path.stem}_{suffix}.txt"
            new_img_path = img_dir / new_img_name
            new_lbl_path = lbl_dir / new_lbl_name

            if new_img_path.exists():
                continue

            try:
                # Image augmentée
                aug_img = fn_img(img)
                aug_img.save(new_img_path, quality=90)

                # Label ajusté ou copié
                if recalc:
                    new_txt = adjust_boxes_zoom(txt_content)
                else:
                    new_txt = txt_content

                if new_txt.strip():
                    new_lbl_path.write_text(new_txt, encoding="utf-8")
                    new_count += 1
                else:
                    # Pas de boîtes valides après zoom → supprime l'image
                    new_img_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"    [WARN] {new_img_name} : {e}")

    return new_count


def main():
    print("\n=== Augmentation Dataset YOLO Detection ===\n")

    total = 0
    for split in ["train", "val", "test"]:
        n = augment_split(split)
        print(f"    → {n} nouvelles images générées\n")
        total += n

    print(f"{'='*50}")
    print(f"  Total nouvelles images : {total}")
    print(f"{'='*50}")

    # Compte total
    for split in ["train", "val", "test"]:
        img_dir = DET_DIR / "images" / split
        if img_dir.exists():
            count = len(list(img_dir.glob("*.jpg")))
            print(f"  {split:6} : {count} images total")

    print(f"\n Prêt pour réentraînement : python train_yolo_detection.py")


if __name__ == "__main__":
    main()