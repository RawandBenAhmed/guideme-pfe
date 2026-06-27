"""
main.py
Point d'entrée de l'agent GuideME.
Mode CLI — l'étudiant tape des messages textuels.
Lancer : python main.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from agent.knowledge_base   import KnowledgeBase
from agent.state_manager    import StateManager
from agent.intent_detector  import detect_intent, get_intent_label
from agent.context_builder  import ContextBuilder
from agent.llm_connector    import LLMConnector
from agent.logger           import Logger


# ============================================================
# AFFICHAGE
# ============================================================

def print_separator():
    print("\n" + "─" * 60)

def print_header():
    print("\n" + "=" * 60)
    print("        GuideME — Assistant Drainage Thoracique")
    print("=" * 60)
    print("  Tape ta question ou commande.")
    print("  Commandes : 'suivant' | 'recommencer' | 'quitter'")
    print("=" * 60 + "\n")

def print_step_banner(op: dict):
    print(f"\n{'═'*60}")
    print(f"  {op['phase_label']}")
    print(f"  Étape {op['operation_id']} — {op['operation_name']}")
    print(f"  Outils : {op['tool_names']}")
    print(f"  Durée estimée : {op.get('estimated_duration_seconds', 30)}s")
    print(f"{'═'*60}\n")

def print_agent(response: str):
    print(f"\n🤖 GuideME : {response}\n")

def print_progress(progress: dict):
    bar_len  = 20
    filled   = int(bar_len * progress["percent"] / 100)
    bar      = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  [{bar}] {progress['percent']}% — Étape {progress['done']}/{progress['total_operations']}\n")


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def run():
    print_header()

    # --- Initialisation ---
    print("Chargement de GuideME...")
    kb  = KnowledgeBase()
    sm  = StateManager()
    llm = LLMConnector()
    lg  = Logger(sm.session_id)
    cb  = ContextBuilder(kb, sm)

    # Vérification Ollama
    if not llm.is_available():
        print("\n❌ Ollama n'est pas disponible. Lance : ollama serve")
        sys.exit(1)

    print(f"\n✅ GuideME prêt — Agent : {kb.get_agent_name()}\n")

    # --- Démarrage session ---
    sm.start()
    lg.log_step_start(sm.current(), kb.get_operation(sm.current())["operation_name"])

    # Intro de la première étape
    op      = kb.get_context(sm.current())
    print_step_banner(op)
    prompt  = cb.build_intro(sm.current())
    response = llm.call(prompt)
    print_agent(response)
    lg.log_agent_response(response, sm.current())

    # --- Boucle de dialogue ---
    while not sm.is_finished:
        try:
            user_input = input("👤 Toi : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession interrompue.")
            break

        if not user_input:
            continue

        # Commande spéciale : quitter
        if user_input.lower() in ["quitter", "quit", "exit", "q"]:
            print("\nSession terminée. À bientôt !")
            break

        # Détection intention
        intent = detect_intent(user_input)
        lg.log_user_message(user_input, intent)

        # Construction du prompt
        prompt = cb.build(intent, user_input)

        # Appel LLM
        print("\n⏳ GuideME réfléchit...")
        response = llm.call(prompt)
        print_agent(response)
        lg.log_agent_response(response, sm.current())

        # Gestion des transitions d'état
        if intent == "next":
            next_id = kb.get_next_operation_id(sm.current())
            duration = sm.get_step_elapsed()
            errors   = sm.get_errors()
            lg.log_step_complete(sm.current(), duration, errors)
            sm.next(next_id)

            if sm.is_finished:
                lg.log_session_end(sm.get_progress())
                print_progress(sm.get_progress())
                print("\n🎉 Procédure complète ! Bravo !\n")
                break

            # Nouvelle étape
            lg.log_step_start(sm.current(), kb.get_operation(sm.current())["operation_name"])
            op = kb.get_context(sm.current())
            print_step_banner(op)
            intro_prompt = cb.build_intro(sm.current())
            intro_response = llm.call(intro_prompt)
            print_agent(intro_response)
            lg.log_agent_response(intro_response, sm.current())

        elif intent == "error":
            nb = sm.add_error()
            lg.log_error(sm.current(), user_input)
            if sm.too_many_errors():
                print(f"\n⚠️  {nb} erreurs à cette étape — fais attention !\n")

        elif intent == "restart":
            print("\n🔄 Redémarrage de la procédure...\n")
            sm.reset()
            sm.start()
            lg.log_step_start(sm.current(), kb.get_operation(sm.current())["operation_name"])
            op = kb.get_context(sm.current())
            print_step_banner(op)
            intro_prompt  = cb.build_intro(sm.current())
            intro_response = llm.call(intro_prompt)
            print_agent(intro_response)

        # Affiche progression
        if intent in ["next", "progress"]:
            print_progress(sm.get_progress())

    # Résumé final
    print_separator()
    print(lg.summary())
    print_separator()


# ============================================================
# LANCEMENT
# ============================================================

if __name__ == "__main__":
    run()