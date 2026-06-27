import shutil
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR  = BASE_DIR / "dataset_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

STEPS_CONFIG = {
 "diagnostique": {
    "index": 1, "label": "Diagnostique",
    "tools": [
        {"id": "T02", "name": "Masque respiratoire", "file": "masque respiratoire.jpg"},
        {"id": "T03", "name": "Gants steriles",      "file": "Gants.webp"},
        {"id": "T01", "name": "Stethoscope",         "file": "stéthoscope.jpg"},
    ],
    "action": "Installe le masque respiratoire au patient, enfile les gants, puis ausculte les deux poumons avec le stethoscope.",
},
    "preparation_patient": {
        "index": 2, "label": "Preparation patient",
        "tools": [
            {"id": "T03", "name": "Gants steriles chirurgicaux", "file": "Gants.webp"},
            {"id": "T04", "name": "Champ sterile fenetré",       "file": "Champs stérile fenêtré.webp"},
        ],
        "action": "Enfile les gants steriles chirurgicaux, palpe le 5e espace intercostal sur la ligne axillaire moyenne, repere le site d insertion et installe le champ sterile.",
    },
    "desinfection_betadine": {
        "index": 3, "label": "Desinfection Betadine",
        "tools": [
            {"id": "T05", "name": "Compresse sterile", "file": "compresse stérile.jpg"},
            {"id": "T06", "name": "Betadine",           "file": "betadine.jpg"},
        ],
        "action": "Imbibe la compresse de Betadine et desinfecte du centre vers la peripherie.",
    },
    "anesthesie_locale": {
        "index": 4, "label": "Anesthesie locale",
        "tools": [
            {"id": "T07", "name": "Seringue",     "file": "seringue.jpg"},
            {"id": "T08", "name": "Xylocaine 1%", "file": "Xylocaïne 1%.jpg"},
        ],
        "action": "Charge la seringue avec Xylocaine 1% et infiltre la peau.",
    },
    "incision_cutanee": {
        "index": 5, "label": "Incision cutanee",
        "tools": [
            {"id": "T09", "name": "Lame de bistouri", "file": "Lame de bistouri.PNG"},
        ],
        "action": "Realise une incision franche d environ 1 cm sur la zone anesthesiee.",
    },
    "dissection_sous_cutanee": {
        "index": 6, "label": "Dissection sous-cutanee",
        "tools": [
            {"id": "T10", "name": "Ciseau", "file": "ciseau.jpg"},
        ],
        "action": "Effectue 3 mouvements de dissection avec le ciseau pour traverser les tissus sous-cutanes et atteindre l espace intercostal.",
    },
    "insertion_drain": {
        "index": 7, "label": "Insertion du drain",
        "tools": [
            {"id": "T11", "name": "Drain thoracique", "file": "drain.jpg"},
        ],
        "action": "Insere le drain dans l espace pleural et oriente-le vers l apex.",
    },
    "connexion_bocal": {
        "index": 8, "label": "Connexion bocal de drainage",
        "tools": [
            {"id": "T12", "name": "Systeme de drainage", "file": "systeme_drainage.jpg"},
        ],
        "action": "Connecte le drain au systeme de drainage et verifie l etancheite.",
    },
}

LABEL_ALIASES = {
    "diagnostic": "diagnostique", "diagnostique": "diagnostique",
    "preparation_patient": "preparation_patient", "preparation": "preparation_patient",
    "desinfection_betadine": "desinfection_betadine", "desinfection": "desinfection_betadine",
    "anesthesie_locale": "anesthesie_locale", "anesthesie": "anesthesie_locale",
    "incision_cutanee": "incision_cutanee", "incision": "incision_cutanee",
    "dissection_sous_cutanee": "dissection_sous_cutanee", "dissection": "dissection_sous_cutanee",
    "insertion_drain": "insertion_drain", "insertion": "insertion_drain",
    "connexion_bocal": "connexion_bocal", "connexion": "connexion_bocal",
    "stethoscope": "diagnostique", "stéthoscope": "diagnostique",
    "masque_respiratoire": "diagnostique", "masque respiratoire": "diagnostique", "masque": "diagnostique",
    "gants": "preparation_patient", "gant": "preparation_patient",
    "champ_sterile_fenetré": "preparation_patient", "champs stérile fenêtré": "preparation_patient",
    "champ stérile fenêtré": "preparation_patient", "champ": "preparation_patient",
    "compresse_sterile": "desinfection_betadine", "compresse stérile": "desinfection_betadine",
    "compresse": "desinfection_betadine", "betadine": "desinfection_betadine", "bétadine": "desinfection_betadine",
    "seringue": "anesthesie_locale", "xylocaine_1pct": "anesthesie_locale",
    "xylocaïne 1%": "anesthesie_locale", "xylocaine": "anesthesie_locale",
    "lame_bistouri": "incision_cutanee", "lame de bistouri": "incision_cutanee",
    "bistouri": "incision_cutanee", "lame": "incision_cutanee",
    "ciseau": "dissection_sous_cutanee", "ciseaux": "dissection_sous_cutanee", "pince": "dissection_sous_cutanee",
    "drain": "insertion_drain", "drain thoracique": "insertion_drain",
    "systeme_drainage": "connexion_bocal", "système de drainage": "connexion_bocal", "bocal": "connexion_bocal",
}


def normalize(raw):
    return raw.strip().lower()


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    for img in root.findall(".//image"):
        name = img.get("name", "")
        for tag in img.findall("tag"):
            lbl = normalize(tag.get("label", ""))
            if lbl:
                records.append({"image_name": name, "label": lbl})
        for attr in img.findall(".//attribute"):
            if attr.get("name", "").lower() == "label" and attr.text:
                records.append({"image_name": name, "label": normalize(attr.text)})
    if not records:
        for item in root.findall(".//item"):
            name = item.get("name", item.get("id", ""))
            lbl  = normalize(item.get("label", ""))
            if lbl:
                records.append({"image_name": name, "label": lbl})
    return records


def find_image(name, folder):
    direct = folder / name
    if direct.exists():
        return direct
    fname = Path(name).name
    for ext in ["", ".jpg", ".jpeg", ".png", ".PNG", ".webp", ".JPG"]:
        c = folder / (fname + ext)
        if c.exists():
            return c
    matches = list(folder.rglob(fname))
    return matches[0] if matches else None


def find_tool_image(file_name, tools_dir):
    dest_tools = PROC_DIR / "tools"
    dest_tools.mkdir(parents=True, exist_ok=True)
    p = tools_dir / file_name
    if p.exists():
        dest = dest_tools / file_name
        if not dest.exists():
            shutil.copy2(p, dest)
        return str(dest)
    for f in tools_dir.iterdir():
        if f.name.lower() == file_name.lower():
            dest = dest_tools / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
            return str(dest)
    return ""


def build_agent_message(cfg):
    names = [t["name"] for t in cfg["tools"]]
    tools = " et ".join(f"[ {n} ]" for n in names)
    return (
        f"Tu es dans l etape {cfg['index']} : {cfg['label']}. "
        f"Prends {tools} et effectue : {cfg['action']}"
    )


def copy_to_processed(src, step_key):
    dest_dir = PROC_DIR / step_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest)


def main():
    all_rows = []
    unknown_labels = set()
    tools_dir = RAW_DIR / "tools_classification"

    sources = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"\n[INFO] {len(sources)} dossier(s) dans data/raw/\n")

    for src_dir in sources:
        source = src_dir.name
        xml_path = src_dir / "annotations.xml"

        if not xml_path.exists():
            print(f"  [WARN] Pas de annotations.xml dans {source}, ignore.")
            continue

        records = parse_cvat_xml(xml_path)
        print(f"-> {source} : {len(records)} annotation(s)")

        for rec in records:
            label = rec["label"]
            step_key = LABEL_ALIASES.get(label)

            if step_key is None:
                unknown_labels.add(label)
                continue

            cfg = STEPS_CONFIG[step_key]

            if source == "tools_classification":
                frame_dest = find_tool_image(rec["image_name"], src_dir)
            else:
                frame_src = find_image(rec["image_name"], src_dir)
                frame_dest = copy_to_processed(frame_src, step_key) if frame_src else ""

            tool_ids    = [t["id"]   for t in cfg["tools"]]
            tool_names  = [t["name"] for t in cfg["tools"]]
            tool_images = [find_tool_image(t["file"], tools_dir) for t in cfg["tools"]]

            all_rows.append({
                "operation_id":    cfg["index"],
                "operation_name":  cfg["label"],
                "operation_image": frame_dest,
                "source":          source,
                "frame_id":        rec["image_name"],
                "tool_ids":        ", ".join(tool_ids),
                "tool_names":      ", ".join(tool_names),
                "tool_images":     ", ".join(tool_images),
                "tool_count":      len(cfg["tools"]),
                "agent_message":   build_agent_message(cfg),
            })

    if unknown_labels:
        print(f"\n[WARN] Labels non reconnus :")
        for lbl in sorted(unknown_labels):
            print(f"   {lbl}")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["operation_id", "source", "frame_id"]).reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"  Dataset : {len(df)} lignes | {df['operation_name'].nunique()} operations")
    print(f"{'='*55}")

    summary = (
        df.groupby(["operation_id", "operation_name"])
        .agg(frames=("frame_id", "count"), sources=("source", "nunique"))
        .reset_index()
    )
    print(summary.to_string(index=False))

    nb_tools   = len(df[df["source"] == "tools_classification"])
    nb_missing = len(df[
        (df["operation_image"].isna() | (df["operation_image"] == "")) &
        (df["source"] != "tools_classification")
    ])

    print(f"\n OK {nb_tools} lignes outils")
    if nb_missing > 0:
        print(f" INFO {nb_missing} frame(s) sans image sur le disque")
    else:
        print(f" OK Toutes les frames sont trouvees.")

    csv_path = OUT_DIR / "dataset_drainage_thoracique.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n OK CSV     -> {csv_path}")

    parquet_path = OUT_DIR / "dataset_drainage_thoracique.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f" OK Parquet -> {parquet_path}")

    summary.to_csv(OUT_DIR / "stats_operations.csv", index=False, encoding="utf-8-sig")
    print(f" OK Stats   -> {OUT_DIR / 'stats_operations.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df.groupby(["operation_id", "operation_name"])["frame_id"].count()
    ylabels = [f"{i}. {n}" for i, n in zip(
        counts.index.get_level_values(0),
        counts.index.get_level_values(1),
    )]
    axes[0].barh(ylabels, counts.values, color="#4E8FD4")
    axes[0].set_title("Frames par operation")
    axes[0].set_xlabel("Nombre de frames")
    axes[0].invert_yaxis()

    src = df["source"].value_counts().sort_index()
    axes[1].bar(src.index, src.values, color="#7C62C9")
    axes[1].set_title("Frames par source")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig_path = OUT_DIR / "distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f" OK Graph   -> {fig_path}")
    print(f"\n Termine -- resultats dans : {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()