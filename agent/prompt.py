"""
agent/prompt.py
Prompt engineering pour GuideME.
Ton : amical, court (1-2 phrases), pédagogique en cas d'erreur.
"""


# ============================================================
# PROMPT SYSTÈME — identité de l'agent
# ============================================================

SYSTEM_PROMPT = """Tu es GuideME, un assistant médical pédagogique spécialisé UNIQUEMENT dans la procédure de drainage thoracique en simulation VR.

RÈGLES ABSOLUES :
- Réponds UNIQUEMENT en français
- Utilise UNIQUEMENT les informations du contexte fourni — ne jamais inventer
- Tutoie toujours l'étudiant
- Maximum 2 phrases par réponse
- Sois direct, précis et encourageant
- En cas d'erreur, explique brièvement POURQUOI c'est dangereux médicalement
- Si la question ne concerne pas le drainage thoracique ou la procédure en cours,
  réponds exactement : "Je suis spécialisé uniquement dans la procédure de drainage
  thoracique. Pose-moi une question sur l'étape en cours."
- Ne réponds JAMAIS à des questions hors contexte médical ou hors procédure
"""

# ============================================================
# TEMPLATES DE PROMPT — un par intention
# ============================================================

def prompt_intro(ctx: dict) -> str:
    """Quand une nouvelle étape commence."""
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Phase          : {ctx['phase_label']}
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Outils requis  : {ctx['tool_names']}
Action         : {ctx['message_action']}
Durée estimée  : {ctx['estimated_duration']} secondes

--- INSTRUCTION ---
Présente cette étape à l'étudiant de façon courte et amicale.
Dis-lui quelle étape il commence et ce qu'il doit faire.
"""


def prompt_tool_question(ctx: dict, user_question: str) -> str:
    """Quand l'étudiant demande quel outil prendre."""
    tools_detail = "\n".join(
        f"  - {t['tool_name']} : {t['description']}"
        for t in ctx['tools_detail']
    )
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Outils requis  : {ctx['tool_names']}
Détail outils  :
{tools_detail}

--- QUESTION DE L'ÉTUDIANT ---
{user_question}

--- INSTRUCTION ---
Indique-lui quel(s) outil(s) prendre et pourquoi en 1-2 phrases.
"""


def prompt_action_question(ctx: dict, user_question: str) -> str:
    """Quand l'étudiant demande quoi faire."""
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Action à faire : {ctx['message_action']}
Alerte         : {ctx['message_alert']}

--- QUESTION DE L'ÉTUDIANT ---
{user_question}

--- INSTRUCTION ---
Explique-lui l'action à effectuer en 1-2 phrases claires.
"""


def prompt_why_question(ctx: dict, user_question: str) -> str:
    """Quand l'étudiant demande pourquoi."""
    errors = "\n".join(f"  - {e}" for e in ctx['common_errors'])
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Action         : {ctx['message_action']}
Alerte         : {ctx['message_alert']}
Erreurs fréquentes :
{errors}

--- QUESTION DE L'ÉTUDIANT ---
{user_question}

--- INSTRUCTION ---
Réponds à sa question en expliquant le raisonnement médical en 1-2 phrases.
"""


def prompt_wrong_tool(ctx: dict, tool_used: str = "") -> str:
    """Quand l'étudiant prend le mauvais outil."""
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Outils requis  : {ctx['tool_names']}
Outil utilisé  : {tool_used if tool_used else 'outil incorrect'}

--- INSTRUCTION ---
Dis-lui que ce n'est pas le bon outil, explique brièvement pourquoi
le bon outil est nécessaire à cette étape. Maximum 2 phrases.
"""


def prompt_error(ctx: dict, error_description: str, nb_errors: int) -> str:
    """Quand l'étudiant fait une erreur."""
    errors = "\n".join(f"  - {e}" for e in ctx['common_errors'])
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Erreurs fréquentes à cette étape :
{errors}
Alerte médicale : {ctx['message_alert']}

--- ERREUR DE L'ÉTUDIANT ---
{error_description}
Nombre d'erreurs à cette étape : {nb_errors}

--- INSTRUCTION ---
Explique-lui son erreur de façon pédagogique — dis-lui ce qu'il a mal fait
et pourquoi c'est important médicalement. Maximum 2 phrases.
{'Encourage-le car il a fait plusieurs erreurs.' if nb_errors >= 2 else ''}
"""


def prompt_completion(ctx: dict, next_ctx: dict = None) -> str:
    """Quand l'étudiant termine une étape."""
    next_info = ""
    if next_ctx:
        next_info = f"Prochaine étape : {next_ctx['operation_id']} — {next_ctx['operation_name']}"

    return f"""{SYSTEM_PROMPT}

--- ÉTAPE TERMINÉE ---
Étape terminée : {ctx['operation_id']} — {ctx['operation_name']}
{next_info}

--- INSTRUCTION ---
Félicite-le brièvement pour avoir terminé cette étape
et annonce-lui la prochaine si elle existe. Maximum 2 phrases.
"""


def prompt_final_completion() -> str:
    """Quand l'étudiant termine toute la procédure."""
    return f"""{SYSTEM_PROMPT}

--- PROCÉDURE TERMINÉE ---
L'étudiant vient de terminer les 8 étapes du drainage thoracique.

--- INSTRUCTION ---
Félicite-le chaleureusement pour avoir complété toute la procédure
de drainage thoracique. Maximum 2 phrases.
"""


def prompt_general_question(ctx: dict, user_question: str) -> str:
    """Pour toute autre question générale."""
    return f"""{SYSTEM_PROMPT}

--- ÉTAPE ACTUELLE ---
Étape          : {ctx['operation_id']} — {ctx['operation_name']}
Phase          : {ctx['phase_label']}
Outils requis  : {ctx['tool_names']}
Action         : {ctx['message_action']}

--- QUESTION DE L'ÉTUDIANT ---
{user_question}

--- INSTRUCTION ---
Réponds à sa question en restant dans le contexte de l'étape actuelle.
Maximum 2 phrases.
"""


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    # Contexte de test
    ctx_test = {
        "operation_id":       4,
        "operation_name":     "Anesthésie locale",
        "phase_label":        "Phase 2 - Préparation",
        "tool_names":         "Seringue, Xylocaïne 1%",
        "tool_count":         2,
        "estimated_duration": 30,
        "message_action":     "Charge la seringue avec la Xylocaïne 1% et infiltre localement la peau.",
        "message_alert":      "Aspire toujours avant d'injecter pour éviter une injection intravasculaire.",
        "message_completion": "Étape 4 terminée — tu vas passer à l'étape 5 : Incision cutanée.",
        "common_errors": [
            "Ne pas aspirer avant d'injecter — risque d'injection intravasculaire",
            "Injecter trop vite et provoquer une douleur",
            "Oublier de charger la seringue avec la Xylocaïne",
        ],
        "tools_detail": [
            {"tool_name": "Seringue",     "description": "Seringue pour injecter l'anesthésique"},
            {"tool_name": "Xylocaïne 1%", "description": "Anesthésique local"},
        ],
    }

    print("=== TEST PROMPTS ===\n")

    print("--- INTRO ---")
    print(prompt_intro(ctx_test))

    print("--- QUESTION OUTIL ---")
    print(prompt_tool_question(ctx_test, "quel outil je prends ?"))

    print("--- QUESTION POURQUOI ---")
    print(prompt_why_question(ctx_test, "pourquoi je dois aspirer avant d'injecter ?"))

    print("--- ERREUR ---")
    print(prompt_error(ctx_test, "l'étudiant a injecté sans aspirer", 2))

    print("--- MAUVAIS OUTIL ---")
    print(prompt_wrong_tool(ctx_test, "bistouri"))

    print("--- COMPLETION ---")
    print(prompt_completion(ctx_test))