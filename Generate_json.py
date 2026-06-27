"""
generate_json.py
Génère les 3 fichiers JSON depuis le dataset CSV
Lancer : python generate_json.py
"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
OUT_DIR  = BASE_DIR / "dataset_output"
JSON_DIR = BASE_DIR / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT_DIR / "dataset_drainage_thoracique.csv", encoding="utf-8-sig")

# ============================================================
# CONFIG COMPLÈTE
# ============================================================

TOOLS_CONFIG = {
    "T01": {"tool_name": "Stethoscope",          "file": "stéthoscope.jpg",               "description": "Used to auscultate the patient and detect abnormal breath sounds",                          "used_in_operations": [1]},
    "T02": {"tool_name": "Masque respiratoire",   "file": "masque respiratoire.jpg",        "description": "Respiratory mask placed on the patient before any procedure",                                "used_in_operations": [1]},
    "T03": {"tool_name": "Gants steriles",        "file": "Gants.webp",                     "description": "Sterile gloves worn by the surgeon to maintain sterility",                                   "used_in_operations": [2]},
    "T04": {"tool_name": "Champ sterile fenetré", "file": "Champs stérile fenêtré.webp",    "description": "Sterile drape with a window placed around the operative field",                             "used_in_operations": [2]},
    "T05": {"tool_name": "Compresse sterile",     "file": "compresse stérile.jpg",          "description": "Sterile compress soaked in Betadine for skin disinfection",                                 "used_in_operations": [3]},
    "T06": {"tool_name": "Betadine",              "file": "betadine.jpg",                   "description": "Antiseptic solution used for skin disinfection before incision",                             "used_in_operations": [3]},
    "T07": {"tool_name": "Seringue",              "file": "seringue.jpg",                   "description": "Syringe used to inject local anesthetic into the skin and subcutaneous tissue",              "used_in_operations": [4]},
    "T08": {"tool_name": "Xylocaine 1%",          "file": "Xylocaïne 1%.jpg",               "description": "Local anesthetic injected to numb the operative area",                                      "used_in_operations": [4]},
    "T09": {"tool_name": "Lame de bistouri",      "file": "Lame de bistouri.PNG",           "description": "Scalpel blade used to make a sharp skin incision",                                          "used_in_operations": [5]},
    "T10": {"tool_name": "Ciseau",                "file": "ciseau.jpg",                     "description": "Scissors used to dissect subcutaneous tissue down to the intercostal space",                "used_in_operations": [6]},
    "T11": {"tool_name": "Drain thoracique",      "file": "drain.jpg",                      "description": "Thoracic drain inserted into the pleural space to evacuate air or fluid",                   "used_in_operations": [7]},
    "T12": {"tool_name": "Systeme de drainage",   "file": "systeme_drainage.jpg",           "description": "Hermetic drainage system connected to the drain to collect pleural fluid",                  "used_in_operations": [8]},
}

OPS_CONFIG = {
    1: {"phase": "diagnostic",    "phase_label": "Phase 1 - Diagnostic",    "next_operation": 2,    "completion_criteria": "user_confirmation", "common_errors": ["Forgetting to check that the mask is properly fitted", "Not auscultating both lungs", "Starting the procedure without the mask in place"]},
    2: {"phase": "preparation",   "phase_label": "Phase 2 - Préparation",   "next_operation": 3,    "completion_criteria": "user_confirmation", "common_errors": ["Touching a non-sterile surface after putting on gloves", "Not centering the sterile drape correctly", "Putting on gloves before installing the drape"]},
    3: {"phase": "preparation",   "phase_label": "Phase 2 - Préparation",   "next_operation": 4,    "completion_criteria": "user_confirmation", "common_errors": ["Moving from periphery to center instead of center to periphery", "Using the same compress twice", "Not waiting for Betadine to dry before proceeding"]},
    4: {"phase": "preparation",   "phase_label": "Phase 2 - Préparation",   "next_operation": 5,    "completion_criteria": "user_confirmation", "common_errors": ["Not aspirating before injecting - risk of intravascular injection", "Injecting too fast causing pain", "Forgetting to load the syringe with Xylocaine", "Not waiting for anesthesia to take effect"]},
    5: {"phase": "intervention",  "phase_label": "Phase 3 - Intervention",  "next_operation": 6,    "completion_criteria": "user_confirmation", "common_errors": ["Making the incision too small - less than 1 cm", "Incision not perpendicular to the skin", "Incising before anesthesia is confirmed"]},
    6: {"phase": "intervention",  "phase_label": "Phase 3 - Intervention",  "next_operation": 7,    "completion_criteria": "user_confirmation", "common_errors": ["Dissecting along the lower rib edge - risk of injuring the vasculo-nervous bundle", "Using excessive force", "Not dissecting deep enough before inserting the drain"]},
    7: {"phase": "intervention",  "phase_label": "Phase 3 - Intervention",  "next_operation": 8,    "completion_criteria": "user_confirmation", "common_errors": ["Not orienting the drain toward the apex", "Forcing the drain when resistance is felt", "Inserting the drain too deep"]},
    8: {"phase": "intervention",  "phase_label": "Phase 3 - Intervention",  "next_operation": None, "completion_criteria": "user_confirmation", "common_errors": ["Not checking the hermetic seal of the connection", "Not verifying bubbling in the bowl after connection"]},
}

MSGS_CONFIG = {
    1: {
        "intro":       "You are in Phase 1 - Diagnostic. Step 1 : Diagnostique. This is the first step of the thoracic drainage procedure.",
        "tool_prompt": "Take the Stethoscope and the Masque respiratoire.",
        "action":      "Auscultate the patient with the stethoscope and verify that the respiratory mask is properly in place.",
        "wrong_tool":  "This is not the correct tool. You need the Stethoscope and the Masque respiratoire.",
        "completion":  "Step 1 completed. Moving to step 2 : Préparation patient.",
        "alert":       "Make sure the mask is properly fitted before starting auscultation."
    },
    2: {
        "intro":       "You are in Phase 2 - Préparation. Step 2 : Préparation patient.",
        "tool_prompt": "Take the Gants steriles and the Champ sterile fenetré.",
        "action":      "Put on the sterile gloves and install the sterile fenestrated drape around the operative field.",
        "wrong_tool":  "This is not the correct tool. You need the Gants steriles and the Champ sterile fenetré.",
        "completion":  "Step 2 completed. Moving to step 3 : Désinfection Bétadine.",
        "alert":       "Do not touch any non-sterile surface after putting on gloves."
    },
    3: {
        "intro":       "You are in Phase 2 - Préparation. Step 3 : Désinfection Bétadine.",
        "tool_prompt": "Take the Compresse sterile and the Betadine.",
        "action":      "Soak the sterile compress with Betadine and disinfect from center to periphery using circular movements.",
        "wrong_tool":  "This is not the correct tool. You need the Compresse sterile and the Betadine.",
        "completion":  "Step 3 completed. Moving to step 4 : Anesthésie locale.",
        "alert":       "Always move from center to periphery. Never go back over an already disinfected area."
    },
    4: {
        "intro":       "You are in Phase 2 - Préparation. Step 4 : Anesthésie locale.",
        "tool_prompt": "Take the Seringue and the Xylocaine 1%.",
        "action":      "Load the syringe with Xylocaine 1% and infiltrate locally the skin and subcutaneous tissue.",
        "wrong_tool":  "This is not the correct tool. You need the Seringue and the Xylocaine 1%.",
        "completion":  "Step 4 completed. Moving to step 5 : Incision cutanée.",
        "alert":       "Always aspirate before injecting to avoid intravascular injection. Wait for anesthesia to take effect."
    },
    5: {
        "intro":       "You are in Phase 3 - Intervention. Step 5 : Incision cutanée.",
        "tool_prompt": "Take the Lame de bistouri.",
        "action":      "Make a sharp skin incision of about 1 cm with the scalpel on the anesthetized area.",
        "wrong_tool":  "This is not the correct tool. You need the Lame de bistouri.",
        "completion":  "Step 5 completed. Moving to step 6 : Dissection sous-cutanée.",
        "alert":       "The incision must be about 1 cm, sharp and perpendicular to the skin."
    },
    6: {
        "intro":       "You are in Phase 3 - Intervention. Step 6 : Dissection sous-cutanée.",
        "tool_prompt": "Take the Ciseau.",
        "action":      "Use the scissors to gently dissect the subcutaneous tissue down to the intercostal space.",
        "wrong_tool":  "This is not the correct tool. You need the Ciseau.",
        "completion":  "Step 6 completed. Moving to step 7 : Insertion du drain.",
        "alert":       "Dissect along the upper edge of the lower rib to avoid the vasculo-nervous bundle."
    },
    7: {
        "intro":       "You are in Phase 3 - Intervention. Step 7 : Insertion du drain.",
        "tool_prompt": "Take the Drain thoracique.",
        "action":      "Insert the thoracic drain into the pleural space following the dissected path. Orient it toward the apex.",
        "wrong_tool":  "This is not the correct tool. You need the Drain thoracique.",
        "completion":  "Step 7 completed. Moving to step 8 : Connexion bocal de drainage.",
        "alert":       "Orient the drain toward the apex. Stop immediately if you feel resistance."
    },
    8: {
        "intro":       "You are in Phase 3 - Intervention. Step 8 : Connexion bocal de drainage. This is the final step.",
        "tool_prompt": "Take the Systeme de drainage.",
        "action":      "Connect the drain to the hermetic drainage system and verify the sealing of the entire assembly.",
        "wrong_tool":  "This is not the correct tool. You need the Systeme de drainage.",
        "completion":  "Step 8 completed. The thoracic drainage procedure is now complete. Well done!",
        "alert":       "Check bubbling in the bowl after connection to confirm correct drain positioning."
    },
}

# ============================================================
# 1. tools.json
# ============================================================
tools_list = []
for tid, cfg in TOOLS_CONFIG.items():
    tools_list.append({
        "tool_id":           tid,
        "tool_name":         cfg["tool_name"],
        "tool_image":        f"data/processed/tools/{cfg['file']}",
        "description":       cfg["description"],
        "used_in_operations": cfg["used_in_operations"],
    })

path = JSON_DIR / "tools.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"tools": tools_list}, f, ensure_ascii=False, indent=2)
print(f"OK tools.json      -> {path}  ({len(tools_list)} tools)")

# ============================================================
# 2. operations.json
# ============================================================
ops_list = []
seen_ops = set()

for _, row in df.iterrows():
    oid = int(row["operation_id"])
    if oid not in seen_ops:
        seen_ops.add(oid)
        cfg      = OPS_CONFIG[oid]
        tool_ids = [x.strip() for x in str(row["tool_ids"]).split(",")]
        ops_list.append({
            "operation_id":       oid,
            "operation_name":     row["operation_name"],
            "phase":              cfg["phase"],
            "phase_label":        cfg["phase_label"],
            "frames_folder":      f"data/processed/{df[df['operation_id']==oid].iloc[0].get('source', '')}",
            "tool_ids":           tool_ids,
            "tool_count":         int(row["tool_count"]),
            "tool_names":         row["tool_names"],
            "next_operation":     cfg["next_operation"],
            "completion_criteria": cfg["completion_criteria"],
            "common_errors":      cfg["common_errors"],
        })

ops_list.sort(key=lambda x: x["operation_id"])
path = JSON_DIR / "operations.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"operations": ops_list}, f, ensure_ascii=False, indent=2)
print(f"OK operations.json -> {path}  ({len(ops_list)} operations)")

# ============================================================
# 3. messages.json
# ============================================================
msgs_list = []
seen_msgs = set()

for _, row in df.iterrows():
    oid = int(row["operation_id"])
    if oid not in seen_msgs:
        seen_msgs.add(oid)
        tool_ids = [x.strip() for x in str(row["tool_ids"]).split(",")]
        msgs_list.append({
            "operation_id":   oid,
            "operation_name": row["operation_name"],
            "phase":          OPS_CONFIG[oid]["phase"],
            "tool_ids":       tool_ids,
            "tool_names":     row["tool_names"],
            "messages":       MSGS_CONFIG[oid],
        })

msgs_list.sort(key=lambda x: x["operation_id"])
path = JSON_DIR / "messages.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump({"messages": msgs_list}, f, ensure_ascii=False, indent=2)
print(f"OK messages.json   -> {path}  ({len(msgs_list)} messages)")

print(f"\nDone -- JSON files in : {JSON_DIR.resolve()}")