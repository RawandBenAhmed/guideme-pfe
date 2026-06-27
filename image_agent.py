"""
image_agent.py
Agent GuideME avec input image.
YOLO Classification → identifie l'étape
YOLO Detection      → identifie les outils
Agent + LLM         → guide l'étudiant
Lancer : python image_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from agent.knowledge_base  import KnowledgeBase
from agent.state_manager   import StateManager
from agent.intent_detector import detect_intent, get_intent_label
from agent.context_builder import ContextBuilder
from agent.llm_connector   import LLMConnector
from agent.logger          import Logger

# ============================================================
# CHEMINS MODÈLES
# ============================================================
BASE_DIR       = Path(__file__).parent
CLS_MODEL_PATH = BASE_DIR / "models" / "drainage_cls"      / "weights" / "best.pt"
DET_MODEL_PATH = BASE_DIR / "models" / "drainage_detection" / "weights" / "best.pt"

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

# Outils attendus par étape
EXPECTED_TOOLS = {
    1: ["stethoscope", "masque_respiratoire", "gants"],
    2: ["gants", "champ_sterile"],
    3: ["compresse_sterile", "betadine"],
    4: ["seringue", "xylocaine"],
    5: ["lame_bistouri"],
    6: ["ciseau"],
    7: ["drain"],
    8: ["systeme_drainage"],
}

# Outils non encore détectables (manque de données)
UNDETECTABLE_TOOLS = ["lame_bistouri", "ciseau", "drain", "systeme_drainage"]


# ============================================================
# YOLO PREDICTOR
# ============================================================

class YOLOPredictor:
    def __init__(self):
        print("[YOLO] Chargement des modèles...")
        self.cls_model = YOLO(str(CLS_MODEL_PATH))
        self.det_model = YOLO(str(DET_MODEL_PATH))
        print("[YOLO] Modèles chargés ✅")

    def predict_operation(self, image_path: str) -> dict:
        """Identifie l'étape chirurgicale depuis une image."""
        results    = self.cls_model.predict(image_path, verbose=False)
        result     = results[0]
        top1_idx   = result.probs.top1
        top1_conf  = float(result.probs.top1conf)
        class_name = result.names[top1_idx]
        op_id      = CLASS_TO_OPERATION.get(class_name, 0)
        return {
            "operation_id":   op_id,
            "operation_name": class_name,
            "confidence":     top1_conf,
        }

    def detect_tools(self, image_path: str, conf: float = 0.3) -> list:
        """Détecte les outils visibles dans une image."""
        results  = self.det_model.predict(image_path, conf=conf, verbose=False)
        detected = []
        for r in results:
            for box in r.boxes:
                tool_name = r.names[int(box.cls)]
                tool_conf = float(box.conf)
                detected.append({
                    "tool":       tool_name,
                    "confidence": tool_conf,
                })
        return detected

    def analyze(self, image_path: str) -> dict:
        """Analyse complète d'une image."""
        op      = self.predict_operation(image_path)
        tools   = self.detect_tools(image_path)
        op_id   = op["operation_id"]

        # Outils attendus à cette étape
        expected = EXPECTED_TOOLS.get(op_id, [])

        # Outils détectés
        detected_names = [t["tool"] for t in tools]

        # Outils corrects présents
        correct = [t for t in detected_names if t in expected]

        # Mauvais outils présents
        wrong = [t for t in detected_names if t not in expected]

        # Outils manquants (attendus mais non détectés)
        missing = [t for t in expected
                   if t not in detected_names and t not in UNDETECTABLE_TOOLS]

        return {
            "operation_id":   op_id,
            "operation_name": op["operation_name"],
            "op_confidence":  op["confidence"],
            "detected_tools": tools,
            "correct_tools":  correct,
            "wrong_tools":    wrong,
            "missing_tools":  missing,
            "expected_tools": expected,
        }


# ============================================================
# AFFICHAGE
# ============================================================

def print_header():
    print("\n" + "=" * 60)
    print("   GuideME — Agent Vision Drainage Thoracique")
    print("=" * 60)
    print("  Commandes : chemin/image.jpg | 'texte' | 'quitter'")
    print("=" * 60 + "\n")

def print_analysis(analysis: dict):
    print(f"\n{'─'*60}")
    print(f"  📷 ANALYSE IMAGE")
    print(f"{'─'*60}")
    print(f"  Étape détectée : {analysis['operation_name']} ({analysis['op_confidence']:.1%})")
    if analysis["detected_tools"]:
        print(f"  Outils détectés :")
        for t in analysis["detected_tools"]:
            status = "✅" if t["tool"] in analysis["correct_tools"] else "❌"
            print(f"    {status} {t['tool']} ({t['confidence']:.1%})")
    if analysis["wrong_tools"]:
        print(f"  ⚠️  Mauvais outils : {', '.join(analysis['wrong_tools'])}")
    if analysis["missing_tools"]:
        print(f"  ❓ Outils non détectés : {', '.join(analysis['missing_tools'])}")
    print(f"{'─'*60}\n")

def print_agent(response: str):
    print(f"\n🤖 GuideME : {response}\n")

def print_step_banner(op: dict):
    print(f"\n{'═'*60}")
    print(f"  {op['phase_label']}")
    print(f"  Étape {op['operation_id']} — {op['operation_name']}")
    print(f"  Outils : {op['tool_names']}")
    print(f"{'═'*60}\n")


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def build_image_prompt(analysis: dict, kb: KnowledgeBase, sm: StateManager) -> str:
    """Construit le prompt basé sur l'analyse de l'image."""
    from agent.prompt import SYSTEM_PROMPT

    op_id    = analysis["operation_id"]
    ctx      = kb.get_context(sm.current())
    expected = analysis["expected_tools"]
    detected = [t["tool"] for t in analysis["detected_tools"]]
    wrong    = analysis["wrong_tools"]
    missing  = analysis["missing_tools"]

    # Cas 1 — mauvais outil détecté
    if wrong:
        wrong_str    = ", ".join(wrong)
        expected_str = ", ".join(expected)
        return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape    : {ctx['operation_id']} — {ctx['operation_name']}
Outils attendus : {expected_str}

--- ANALYSE IMAGE ---
Mauvais outil détecté : {wrong_str}

--- INSTRUCTION ---
Dis à l'étudiant qu'il a pris le mauvais outil.
Explique quel outil il doit prendre et pourquoi. Maximum 2 phrases.
"""

    # Cas 2 — bonne étape, bons outils
    if correct := analysis["correct_tools"]:
        correct_str = ", ".join(correct)
        return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape   : {ctx['operation_id']} — {ctx['operation_name']}
Action  : {ctx['message_action']}
Alerte  : {ctx['message_alert']}

--- ANALYSE IMAGE ---
Outils corrects détectés : {correct_str}

--- INSTRUCTION ---
Confirme que l'étudiant a le bon outil et guide-le pour effectuer l'action.
Maximum 2 phrases, ton encourageant.
"""

    # Cas 3 — étape détectée mais aucun outil visible
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape   : {ctx['operation_id']} — {ctx['operation_name']}
Outils attendus : {', '.join(expected)}

--- ANALYSE IMAGE ---
Aucun outil détecté dans l'image.

--- INSTRUCTION ---
Dis à l'étudiant de prendre les bons outils pour cette étape.
Maximum 2 phrases.
"""


def run():
    print_header()

    # Initialisation
    print("Chargement de GuideME...")
    kb    = KnowledgeBase()
    sm    = StateManager()
    llm   = LLMConnector()
    yolo  = YOLOPredictor()
    lg    = Logger(sm.session_id)
    cb    = ContextBuilder(kb, sm)

    if not llm.is_available():
        print("❌ Ollama non disponible. Lance : ollama serve")
        sys.exit(1)

    print(f"\n✅ GuideME prêt\n")

    sm.start()
    lg.log_step_start(sm.current(), kb.get_operation(sm.current())["operation_name"])

    # Intro étape 1
    op = kb.get_context(sm.current())
    print_step_banner(op)
    intro = llm.call(cb.build_intro(sm.current()))
    print_agent(intro)

    # Boucle principale
    while not sm.is_finished:
        try:
            user_input = input("👤 Toi (image ou texte) : ").strip().strip('"')
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrompue.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quitter", "quit", "q"]:
            print("\nSession terminée.")
            break

        # Vérifie si c'est un chemin d'image
        img_path = Path(user_input)
        if img_path.exists() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            # Analyse de l'image
            print("\n⏳ Analyse de l'image...")
            analysis = yolo.analyze(str(img_path))
            print_analysis(analysis)

            # Vérification étape
            detected_op = analysis["operation_id"]
            current_op  = sm.current()

            if detected_op != current_op and detected_op != 0:
                print(f"  ℹ️  Image : étape {detected_op} | Attendu : étape {current_op}")

            # Construction prompt basé sur l'analyse
            prompt   = build_image_prompt(analysis, kb, sm)
            response = llm.call(prompt)
            print_agent(response)
            lg.log_agent_response(response, sm.current())

            # Enregistre erreur si mauvais outil
            if analysis["wrong_tools"]:
                sm.add_error()
                lg.log_error(sm.current(), f"Mauvais outil : {analysis['wrong_tools']}")

        else:
            # Input texte — pipeline normal
            intent   = detect_intent(user_input)
            lg.log_user_message(user_input, intent)

            print("\n⏳ GuideME réfléchit...")
            prompt   = cb.build(intent, user_input)
            response = llm.call(prompt)
            print_agent(response)
            lg.log_agent_response(response, sm.current())

            # Gestion transitions
            if intent == "next":
                next_id  = kb.get_next_operation_id(sm.current())
                duration = sm.get_step_elapsed()
                errors   = sm.get_errors()
                lg.log_step_complete(sm.current(), duration, errors)
                sm.next(next_id)

                if sm.is_finished:
                    lg.log_session_end(sm.get_progress())
                    print("\n🎉 Procédure complète ! Bravo !\n")
                    break

                lg.log_step_start(sm.current(), kb.get_operation(sm.current())["operation_name"])
                op = kb.get_context(sm.current())
                print_step_banner(op)
                intro = llm.call(cb.build_intro(sm.current()))
                print_agent(intro)

            elif intent == "error":
                sm.add_error()
                lg.log_error(sm.current(), user_input)

            elif intent == "restart":
                sm.reset()
                sm.start()
                op = kb.get_context(sm.current())
                print_step_banner(op)
                intro = llm.call(cb.build_intro(sm.current()))
                print_agent(intro)

    print(f"\n{'-'*60}")
    print(lg.summary())
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    run()