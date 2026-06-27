from pathlib import Path
import xml.etree.ElementTree as ET
import shutil

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed_VR"

LABEL_TO_STEP = {
    "diagnostic":               "diagnostique",
    "diagnostique":             "diagnostique",
    "preparation_patient":      "preparation_patient",
    "preparation":              "preparation_patient",
    "desinfection_betadine":    "desinfection_betadine",
    "desinfection":             "desinfection_betadine",
    "anesthesie_locale":        "anesthesie_locale",
    "anesthesie":               "anesthesie_locale",
    "incision_cutanee":         "incision_cutanee",
    "incision":                 "incision_cutanee",
    "dissection_sous_cutanee":  "dissection_sous_cutanee",
    "dissection":               "dissection_sous_cutanee",
    "insertion_drain":          "insertion_drain",
    "insertion":                "insertion_drain",
    "connexion_bocal":          "connexion_bocal",
    "connexion":                "connexion_bocal",
}

# Crée les dossiers
for step in set(LABEL_TO_STEP.values()):
    (OUT_DIR / step).mkdir(parents=True, exist_ok=True)

total = 0
for video in ["video1", "video2", "video3"]:
    xml_path = RAW_DIR / video / "annotations.xml"
    vid_dir  = RAW_DIR / video
    if not xml_path.exists():
        print(f"[SKIP] {video} — pas d annotations.xml")
        continue

    tree = ET.parse(xml_path)
    root = tree.getroot()
    copied = 0

    for img in root.findall(".//image"):
        img_name = img.get("name", "")
        # Cherche le label dans les tags
        label = None
        for tag in img.findall("tag"):
            lbl = tag.get("label", "").strip().lower()
            if lbl in LABEL_TO_STEP:
                label = LABEL_TO_STEP[lbl]
                break

        if not label:
            continue

        # Cherche le fichier image
        img_path = vid_dir / img_name
        if not img_path.exists():
            img_path = vid_dir / Path(img_name).name
        if not img_path.exists():
            continue

        dest = OUT_DIR / label / img_path.name
        if not dest.exists():
            shutil.copy2(img_path, dest)
            copied += 1

    print(f"{video} : {copied} frames copiées")
    total += copied

print(f"\nTotal : {total} frames")
print("\nRésultat par étape :")
for step in sorted(set(LABEL_TO_STEP.values())):
    imgs = list((OUT_DIR / step).glob("*.jpg"))
    print(f"  {step:30} : {len(imgs)} images")