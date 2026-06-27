import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
OUT_DIR  = BASE_DIR / "data" / "processed_VR"

# Mapping étape → vidéo source
STEP_SOURCES = {
    "diagnostique":           "video2",
    "preparation_patient":    "video1",
    "desinfection_betadine":  "video1",
    "anesthesie_locale":      "video3",
    "incision_cutanee":       "video3",
    "dissection_sous_cutanee":"video1",
    "insertion_drain":        "video1",
    "connexion_bocal":        "video1",
}

for step, video in STEP_SOURCES.items():
    src_dir  = RAW_DIR / "processed" if (RAW_DIR / "processed" / step).exists() else BASE_DIR / "data" / "processed"
    step_dir = src_dir / step
    out_step = OUT_DIR / step
    out_step.mkdir(parents=True, exist_ok=True)

    # Copie seulement les frames originales (pas blur/bright/zoom)
    frames = [f for f in step_dir.glob("frame_*.jpg")
              if not any(s in f.stem for s in ["_blur", "_bright", "_zoom"])]

    print(f"{step:30} : {len(frames)} frames → {out_step}")
    for f in frames:
        shutil.copy2(f, out_step / f.name)

print("\nTerminé ✅")