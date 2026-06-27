"""
agent/intent_detector.py
Détecte l'intention de l'utilisateur à partir de son message.
"""

import re

INTENT_NEXT       = "next"
INTENT_TOOL       = "tool"
INTENT_ACTION     = "action"
INTENT_WHY        = "why"
INTENT_ERROR      = "error"
INTENT_WRONG_TOOL = "wrong_tool"
INTENT_RESTART    = "restart"
INTENT_PROGRESS   = "progress"
INTENT_ALERT      = "alert"
INTENT_GENERAL    = "general"

# ✅ FIX : séparation claire entre keywords "exacts" (phrases entières)
# et keywords "mots isolés" (noms d'instruments) qui nécessitent
# une vérification par mot entier pour éviter les faux positifs.
# Ex: "lame" ne doit pas matcher "reclamé", "bocal" pas "local", etc.

KEYWORDS_PHRASE = {
    INTENT_WRONG_TOOL: [
        "mauvais outil", "pas le bon outil", "wrong tool",
        "ce n'est pas le bon", "pas correct", "autre outil",
        "ce n est pas le bon", "pas le bon"
    ],
    INTENT_RESTART: [
        "recommencer", "restart", "reset", "reprendre",
        "repartir", "refaire", "depuis le début", "depuis le debut",
        "repartir à zéro", "tout recommencer"
    ],
    INTENT_NEXT: [
        "suivant", "next", "continuer", "suite", "j'ai fini",
        "c'est fait", "terminé", "done", "passer",
        "étape suivante", "j ai fini", "fini", "passons",
        "on passe", "avancer", "valider", "j'ai terminé",
        "j ai terminé", "termine", "c est fait"
    ],
    INTENT_WHY: [
        "pourquoi", "why", "raison", "je comprends pas",
        "c'est quoi", "c est quoi", "utilité", "objectif",
        "je ne comprends pas", "expliquer pourquoi"
    ],
    INTENT_ERROR: [
        "erreur", "j ai fait une erreur", "j'ai fait faux",
        "pas bon", "incorrect", "je me suis trompé",
        "j ai raté", "j'ai raté", "j'ai fait une erreur"
    ],
    INTENT_TOOL: [
        # Demandes directes
        "donne moi", "donne-moi", "apporte moi", "apporte-moi",
        "je veux", "je voudrais", "montre moi", "montre-moi",
        "besoin de", "il me faut",
        # Questions sur les outils
        "quel outil", "quoi prendre", "je prends quoi",
        "avec quoi", "instruments", "matériel",
        "quel instrument", "qu est ce que je prends",
        "outils nécessaires", "outils requis",
        "quels outils", "quel outil",
    ],
    INTENT_ACTION: [
        "quoi faire", "comment faire", "que dois-je",
        "je fais quoi", "comment procéder", "que faire",
        "comment je fais", "quelle action", "comment je",
        "je dois faire", "que je fais", "quesque je fait",
        "quesque je dois", "je suis dans quelle", "quelle etape",
        "quelle étape", "dans quelle étape",
    ],
    INTENT_PROGRESS: [
        "progression", "où en suis", "combien",
        "reste", "bilan", "résumé", "avancement",
        "étape actuelle", "où j en suis"
    ],
    INTENT_ALERT: [
        "alerte", "danger", "risque", "avertissement",
        "prudence", "warning", "conseil", "attention"
    ],
}

# ✅ FIX : noms d'instruments détectés par MOT ENTIER uniquement
# pour éviter "lame" dans "reclamé", "mal" dans "malaise", etc.
TOOL_NAME_KEYWORDS = [
    "stéthoscope", "stethoscope", "masque", "gants", "gant",
    "compresse", "bétadine", "betadine", "seringue",
    "xylocaine", "xylocaïne", "bistouri", "lame",
    "ciseau", "ciseaux", "drain", "bocal",
    "champ stérile", "champ sterile",
]

# ✅ FIX : mots courts qui causaient des faux positifs — retirés de KEYWORDS_PHRASE
# "faux", "mal", "raté" → maintenant vérifiés comme mots entiers uniquement
ERROR_WORD_KEYWORDS = ["faux", "mal", "raté", "rate"]

HORS_SUJET = [
    "bonjour", "bonsoir", "salut", "hello", "hi",
    "comment tu vas", "ca va", "ça va", "comment vas tu",
    "merci", "au revoir", "bye", "ciao", "bonne journée"
]


def _word_in_msg(word: str, msg: str) -> bool:
    """Vérifie si 'word' apparaît comme mot entier dans msg."""
    pattern = r'(?<![a-zàâäéèêëîïôùûüç])' + re.escape(word) + r'(?![a-zàâäéèêëîïôùûüç])'
    return bool(re.search(pattern, msg, re.IGNORECASE))


def detect_intent(user_message: str) -> str:
    msg = user_message.strip().lower()

    # Salutations courtes
    if any(h in msg for h in HORS_SUJET) and len(msg) < 30:
        return INTENT_GENERAL

    # Message très court
    if len(msg) <= 3 and msg in ["ok", "go", "oui", "yes", "si"]:
        return INTENT_NEXT

    # Cherche les phrases dans l'ordre de priorité
    for intent, keywords in KEYWORDS_PHRASE.items():
        for kw in keywords:
            if kw in msg:
                return intent

    # ✅ FIX : noms d'instruments — vérification mot entier uniquement
    for tool_kw in TOOL_NAME_KEYWORDS:
        if _word_in_msg(tool_kw, msg):
            return INTENT_TOOL

    # ✅ FIX : mots d'erreur courts — vérification mot entier uniquement
    for err_kw in ERROR_WORD_KEYWORDS:
        if _word_in_msg(err_kw, msg):
            return INTENT_ERROR

    return INTENT_GENERAL


INTENT_LABELS = {
    INTENT_NEXT:       "Passer à l'étape suivante",
    INTENT_TOOL:       "Demande d'outil",
    INTENT_ACTION:     "Demande d'action",
    INTENT_WHY:        "Demande d'explication",
    INTENT_ERROR:      "Signalement d'erreur",
    INTENT_WRONG_TOOL: "Mauvais outil",
    INTENT_RESTART:    "Recommencer",
    INTENT_PROGRESS:   "Progression",
    INTENT_ALERT:      "Demande d'alerte",
    INTENT_GENERAL:    "Question générale",
}

def get_intent_label(intent: str) -> str:
    return INTENT_LABELS.get(intent, "Inconnu")


if __name__ == "__main__":
    tests = [
        ("donne moi le masque",              "tool"),
        ("donne moi les gants",              "tool"),
        ("donnez-moi les gants",             "tool"),
        ("quel outil je dois utiliser",      "tool"),
        ("quels sont les outils nécessaires","tool"),
        ("je suis dans quelle étape",        "action"),
        ("pourquoi on utilise le ciseau",    "why"),
        ("j'ai fini",                        "next"),
        ("quelle est la recette de la pizza","general"),
        ("quoi faire maintenant",            "action"),
        ("de numoire légaux",                "general"),   # ✅ ne doit plus matcher tool
        ("j'ai fait une erreur",             "error"),
        ("c'est mal fait",                   "error"),
        ("j'ai raté",                        "error"),
    ]
    print("=== TEST INTENT DETECTOR ===\n")
    ok = 0
    for msg, expected in tests:
        result = detect_intent(msg)
        status = "✅" if result == expected else "❌"
        if result == expected: ok += 1
        print(f"{status} {msg:<45} → {result} (attendu: {expected})")
    print(f"\n{ok}/{len(tests)} correct")