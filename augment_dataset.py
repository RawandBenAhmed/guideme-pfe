"""
augment_dataset.py
Augmentation x3 du dataset drainage thoracique
Lancer : python augment_dataset.py
"""

import shutil
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

BASE_DIR = Path(__file__).parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "dataset_output"
CSV_PATH = OUT_DIR / "dataset_drainage_thoracique.csv"

# ============================================================
# TRANSFORMATIONS
# ============================================================

def augment_blur(img: Image.Image) -> Image.Image:
    """Version floue."""
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def augment_zoom(img: Image.Image) -> Image.Image:
    """Version zoomée (crop centre 80% puis resize original)."""
    w, h = img.size
    margin_w = int(w * 0.1)
    margin_h = int(h * 0.1)
    cropped = img.crop((margin_w, margin_h, w - margin_w, h - margin_h))
    return cropped.resize((w, h), Image.LANCZOS)


def augment_bright(img: Image.Image) -> Image.Image:
    """Version lumineuse (luminosité x1.5)."""
    return ImageEnhance.Brightness(img).enhance(1.5)


AUGMENTATIONS = [
    ("blur",   augment_blur),
    ("zoom",   augment_zoom),
    ("bright", augment_bright),
]

# ============================================================
# PIPELINE
# ============================================================

def augment_folder(folder: Path, step_key: str, existing_rows: list) -> list:
    """Augmente toutes les images d'un dossier et retourne les nouvelles lignes."""
    new_rows = []
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
             list(folder.glob("*.png")) + list(folder.glob("*.PNG")) + \
             list(folder.glob("*.webp"))

    # Filtrer les images déjà augmentées
    images = [p for p in images if not any(
        p.stem.endswith(f"_{suf}") for suf, _ in AUGMENTATIONS
    )]

    print(f"  {folder.name} : {len(images)} image(s) originale(s)")

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    [WARN] Impossible d'ouvrir {img_path.name} : {e}")
            continue

        # Cherche la ligne correspondante dans le CSV
        ref_rows = [r for r in existing_rows if r.get("frame_id") == img_path.name]
        ref_row  = ref_rows[0] if ref_rows else None

        for suffix, transform in AUGMENTATIONS:
            new_name = f"{img_path.stem}_{suffix}{img_path.suffix}"
            new_path = folder / new_name

            if new_path.exists():
                continue

            try:
                new_img = transform(img)
                new_img.save(new_path, quality=90)
            except Exception as e:
                print(f"    [WARN] Erreur augmentation {new_name} : {e}")
                continue

            if ref_row:
                new_row = ref_row.copy()
                new_row["frame_id"]        = new_name
                new_row["operation_image"] = str(new_path)
                new_rows.append(new_row)

    return new_rows


def main():
    print("\n Augmentation du dataset — Drainage Thoracique")
    print(f" Dossier : {PROC_DIR}")
    print(f" CSV     : {CSV_PATH}\n")

    # Charger le CSV existant
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df = df.fillna("")
    existing_rows = df.to_dict(orient="records")
    print(f" CSV chargé : {len(df)} lignes existantes\n")

    all_new_rows = []

    # Augmenter chaque dossier d'étape
    step_folders = [
        "diagnostique", "preparation_patient", "desinfection_betadine",
        "anesthesie_locale", "incision_cutanee", "dissection_sous_cutanee",
        "insertion_drain", "connexion_bocal"
    ]

    print("=== Étapes ===")
    for step in step_folders:
        folder = PROC_DIR / step
        if not folder.exists():
            print(f"  [WARN] Dossier introuvable : {step}")
            continue
        step_rows = [r for r in existing_rows if r.get("source") != "tools_classification"
                     and Path(str(r.get("operation_image", ""))).parent.name == step]
        new_rows = augment_folder(folder, step, step_rows)
        all_new_rows.extend(new_rows)
        print(f"    → {len(new_rows)} nouvelles images générées")

    # Augmenter le dossier tools
    print("\n=== Outils ===")
    tools_folder = PROC_DIR / "tools"
    if tools_folder.exists():
        tools_rows = [r for r in existing_rows if r.get("source") == "tools_classification"]
        new_tool_rows = augment_folder(tools_folder, "tools", tools_rows)
        all_new_rows.extend(new_tool_rows)
        print(f"    → {len(new_tool_rows)} nouvelles images outils générées")

    # Mettre à jour le CSV
    if all_new_rows:
        df_new  = pd.DataFrame(all_new_rows)
        df_full = pd.concat([df, df_new], ignore_index=True)
        df_full = df_full.sort_values(
            ["operation_id", "source", "frame_id"]
        ).reset_index(drop=True)

        df_full.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        df_full.to_parquet(OUT_DIR / "dataset_drainage_thoracique.parquet", index=False)

        print(f"\n{'='*55}")
        print(f"  Avant augmentation : {len(df)} lignes")
        print(f"  Nouvelles lignes   : {len(all_new_rows)}")
        print(f"  Total final        : {len(df_full)} lignes")
        print(f"{'='*55}")
        print(f"\n OK CSV mis à jour -> {CSV_PATH}")
    else:
        print("\n Aucune nouvelle image générée.")

    print("\n Terminé !")


if __name__ == "__main__":
    main()