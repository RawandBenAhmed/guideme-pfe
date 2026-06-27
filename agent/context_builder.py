"""
agent/context_builder.py
Assemble le contexte complet pour le LLM.
Choisit le bon template de prompt selon l'intention détectée.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.intent_detector import (
    INTENT_NEXT, INTENT_TOOL, INTENT_ACTION, INTENT_WHY,
    INTENT_ERROR, INTENT_WRONG_TOOL, INTENT_RESTART,
    INTENT_PROGRESS, INTENT_ALERT, INTENT_GENERAL
)
from agent.prompt import (
    prompt_intro, prompt_tool_question, prompt_action_question,
    prompt_why_question, prompt_wrong_tool, prompt_error,
    prompt_completion, prompt_final_completion, prompt_general_question
)


class ContextBuilder:
    """
    Reçoit l'intention, l'état et la knowledge base
    et retourne le prompt final prêt à envoyer au LLM.
    """

    def __init__(self, knowledge_base, state_manager):
        self.kb = knowledge_base
        self.sm = state_manager
        print("[CTX] ContextBuilder initialisé")

    # ============================================================
    # MÉTHODE PRINCIPALE
    # ============================================================

    def build(self, intent: str, user_message: str = "", extra: dict = None) -> str:
        """
        Construit le prompt final selon l'intention.

        Args:
            intent       : intention détectée par intent_detector
            user_message : message original de l'étudiant
            extra        : données supplémentaires (ex: outil utilisé)

        Returns:
            str : prompt complet prêt pour le LLM
        """
        extra = extra or {}
        op_id = self.sm.current()

        # Vérifie que l'opération est valide
        if not self.kb.is_valid_operation(op_id):
            return self._prompt_error_system(op_id)

        ctx = self.kb.get_context(op_id)

        # ---- Routing selon l'intention ----

        if intent == INTENT_NEXT:
            return self._build_next(ctx)

        elif intent == INTENT_TOOL:
            return prompt_tool_question(ctx, user_message)

        elif intent == INTENT_ACTION:
            return prompt_action_question(ctx, user_message)

        elif intent == INTENT_WHY:
            return prompt_why_question(ctx, user_message)

        elif intent == INTENT_ERROR:
            nb_errors = self.sm.get_errors()
            error_desc = user_message or "L'étudiant a signalé une erreur."
            return prompt_error(ctx, error_desc, nb_errors)

        elif intent == INTENT_WRONG_TOOL:
            tool_used = extra.get("tool_used", "")
            return prompt_wrong_tool(ctx, tool_used)

        elif intent == INTENT_RESTART:
            return self._build_restart()

        elif intent == INTENT_PROGRESS:
            return self._build_progress(ctx)

        elif intent == INTENT_ALERT:
            return self._build_alert(ctx)

        else:
            # INTENT_GENERAL ou non reconnu
            return prompt_general_question(ctx, user_message)

    # ============================================================
    # BUILDERS SPÉCIFIQUES
    # ============================================================

    def _build_next(self, ctx: dict) -> str:
        """Prompt pour passer à l'étape suivante."""
        next_id = ctx["next_operation"]

        if next_id is None:
            # Dernière étape — procédure terminée
            return prompt_final_completion()

        next_ctx = self.kb.get_context(next_id)
        return prompt_completion(ctx, next_ctx)

    def _build_restart(self) -> str:
        """Prompt pour recommencer la procédure."""
        from agent.prompt import SYSTEM_PROMPT
        return f"""{SYSTEM_PROMPT}

--- INSTRUCTION ---
L'étudiant veut recommencer la procédure depuis le début.
Dis-lui que la procédure va recommencer à l'étape 1 — Diagnostique.
Maximum 2 phrases, ton encourageant.
"""

    def _build_progress(self, ctx: dict) -> str:
        """Prompt pour afficher la progression."""
        from agent.prompt import SYSTEM_PROMPT
        progress = self.sm.get_progress()
        errors   = sum(progress["errors"].values())
        elapsed  = progress["elapsed_seconds"]

        return f"""{SYSTEM_PROMPT}

--- PROGRESSION ACTUELLE ---
Étape actuelle     : {progress['current_operation']} — {ctx['operation_name']}
Étapes terminées   : {progress['done']} / {progress['total_operations']}
Pourcentage        : {progress['percent']}%
Temps écoulé       : {elapsed} secondes
Erreurs au total   : {errors}

--- INSTRUCTION ---
Résume la progression de l'étudiant de façon encourageante.
Maximum 2 phrases.
"""

    def _build_alert(self, ctx: dict) -> str:
        """Prompt pour afficher l'alerte de sécurité de l'étape."""
        from agent.prompt import SYSTEM_PROMPT
        return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape   : {ctx['operation_id']} — {ctx['operation_name']}
Alerte  : {ctx['message_alert']}
Erreurs fréquentes :
{chr(10).join(f"  - {e}" for e in ctx['common_errors'])}

--- INSTRUCTION ---
Rappelle à l'étudiant l'alerte de sécurité importante de cette étape.
Maximum 2 phrases, sois précis sur le risque médical.
"""

    def _prompt_error_system(self, op_id: int) -> str:
        """Prompt d'erreur système si l'opération est invalide."""
        from agent.prompt import SYSTEM_PROMPT
        return f"""{SYSTEM_PROMPT}

--- ERREUR SYSTÈME ---
L'opération {op_id} n'existe pas dans la base de données.

--- INSTRUCTION ---
Dis à l'étudiant qu'il y a un problème technique et qu'il faut recommencer.
"""

    # ============================================================
    # INTRO D'ÉTAPE
    # ============================================================

    def build_intro(self, operation_id: int = None) -> str:
        """
        Construit le prompt d'introduction pour une étape.
        Utilisé au démarrage ou après un passage à l'étape suivante.
        """
        op_id = operation_id or self.sm.current()
        ctx   = self.kb.get_context(op_id)
        return prompt_intro(ctx)


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from agent.knowledge_base  import KnowledgeBase
    from agent.state_manager   import StateManager
    from agent.intent_detector import detect_intent

    print("\n=== TEST CONTEXT BUILDER ===\n")

    kb = KnowledgeBase()
    sm = StateManager()
    sm.start()
    cb = ContextBuilder(kb, sm)

    tests = [
        ("quel outil je prends ?",               {}),
        ("pourquoi je dois aspirer ?",            {}),
        ("j'ai fait une erreur",                  {}),
        ("ce n'est pas le bon outil",             {"tool_used": "bistouri"}),
        ("où en suis-je ?",                       {}),
        ("c'est quoi le risque ici ?",            {}),
        ("bonjour comment tu vas",                {}),
    ]

    for msg, extra in tests:
        intent = detect_intent(msg)
        prompt = cb.build(intent, msg, extra)
        print(f"Message  : '{msg}'")
        print(f"Intention: {intent}")
        print(f"Prompt   : {prompt[:120].strip()}...")
        print("-" * 60)

    # Test intro
    print("\n--- Intro étape 1 ---")
    intro = cb.build_intro(1)
    print(intro[:200] + "...")