"""
compare_models.py
Compare Qwen 2.5 7B vs Meditron 7B sur des questions médicales
de drainage thoracique.
Lancer : python compare_models.py
"""

import json
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = {
    "Qwen 2.5 7B":  "qwen2.5:7b-instruct",
    "Meditron 7B":  "meditron",
}

SYSTEM_PROMPT = """Tu es GuideME, un assistant pédagogique spécialisé
dans la procédure de drainage thoracique.
Réponds en français, maximum 2 phrases, ton amical et pédagogique."""

# 10 questions de test couvrant les 8 étapes
TEST_QUESTIONS = [
    {
        "id": 1,
        "etape": "Diagnostique",
        "question": "Tu es dans l'étape 1 : Diagnostique. Prends le masque respiratoire, les gants et le stéthoscope. Que dois-je faire maintenant ?",
        "mots_cles": ["masque", "gants", "stéthoscope", "ausculter", "poumon"],
    },
    {
        "id": 2,
        "etape": "Préparation patient",
        "question": "Tu es dans l'étape 2 : Préparation patient. Prends les gants stériles et le champ stérile. Comment je procède ?",
        "mots_cles": ["gants", "champ", "stérile", "intercostal", "axillaire"],
    },
    {
        "id": 3,
        "etape": "Désinfection Bétadine",
        "question": "Tu es dans l'étape 3 : Désinfection Bétadine. Prends la compresse et la Bétadine. Dans quel sens je désinfecte ?",
        "mots_cles": ["centre", "périphérie", "compresse", "bétadine", "désinfect"],
    },
    {
        "id": 4,
        "etape": "Anesthésie locale",
        "question": "Tu es dans l'étape 4 : Anesthésie locale. Prends la seringue et la Xylocaïne. Pourquoi je dois aspirer avant d'injecter ?",
        "mots_cles": ["intravasculaire", "aspirer", "xylocaïne", "injection", "sécurité"],
    },
    {
        "id": 5,
        "etape": "Incision cutanée",
        "question": "Tu es dans l'étape 5 : Incision cutanée. Prends la lame de bistouri. Quelle taille doit avoir l'incision ?",
        "mots_cles": ["1 cm", "incision", "franche", "bistouri", "peau"],
    },
    {
        "id": 6,
        "etape": "Dissection sous-cutanée",
        "question": "Tu es dans l'étape 6 : Dissection sous-cutanée. Prends le ciseau. Combien de mouvements de dissection ?",
        "mots_cles": ["3", "dissection", "ciseau", "intercostal", "supérieur"],
    },
    {
        "id": 7,
        "etape": "Insertion du drain",
        "question": "Tu es dans l'étape 7 : Insertion du drain. Prends le drain thoracique. Vers où je dois orienter le drain ?",
        "mots_cles": ["apex", "drain", "pleural", "orienter", "insérer"],
    },
    {
        "id": 8,
        "etape": "Connexion bocal",
        "question": "Tu es dans l'étape 8 : Connexion bocal. Prends le système de drainage. Comment je vérifie que ça fonctionne ?",
        "mots_cles": ["bullage", "bocal", "étanchéité", "connexion", "drainage"],
    },
    {
        "id": 9,
        "etape": "Mauvais outil",
        "question": "Tu es dans l'étape 3 : Désinfection Bétadine. L'étudiant a pris la seringue — c'est le mauvais outil. Que lui dire ?",
        "mots_cles": ["compresse", "bétadine", "mauvais", "désinfection", "prendre"],
    },
    {
        "id": 10,
        "etape": "Question hors sujet",
        "question": "Quelle est la capitale de la France ?",
        "mots_cles": ["drainage", "spécialisé", "thoracique", "procédure"],
    },
]


def call_model(model_id: str, question: str) -> tuple:
    """Appelle le modèle et retourne (réponse, temps)."""
    payload = {
        "model":  model_id,
        "prompt": f"{SYSTEM_PROMPT}\n\nQuestion : {question}",
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 100, "num_ctx": 512}
    }
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result   = json.loads(resp.read().decode("utf-8"))
            response = result.get("response", "").strip()
            elapsed  = round(time.time() - start, 1)
            return response, elapsed
    except Exception as e:
        return f"ERREUR : {e}", 0.0


def score_response(response: str, mots_cles: list) -> int:
    """Score 0-10 : nombre de mots-clés présents dans la réponse."""
    r = response.lower()
    found = sum(1 for kw in mots_cles if kw.lower() in r)
    return round((found / len(mots_cles)) * 10)


def count_sentences(text: str) -> int:
    """Compte le nombre de phrases."""
    return len([s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()])


def main():
    print("\n" + "="*70)
    print("  COMPARAISON MODÈLES LLM — Drainage Thoracique")
    print("="*70)

    results = {name: {"scores": [], "times": [], "phrases": [], "responses": []}
               for name in MODELS}

    for q in TEST_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Q{q['id']} — Étape : {q['etape']}")
        print(f"Question : {q['question'][:80]}...")
        print()

        for model_name, model_id in MODELS.items():
            print(f"  [{model_name}] en cours...")
            response, elapsed = call_model(model_id, q["question"])
            score   = score_response(response, q["mots_cles"])
            phrases = count_sentences(response)

            results[model_name]["scores"].append(score)
            results[model_name]["times"].append(elapsed)
            results[model_name]["phrases"].append(phrases)
            results[model_name]["responses"].append(response)

            print(f"  Réponse : {response[:100]}...")
            print(f"  Score   : {score}/10 | Temps : {elapsed}s | Phrases : {phrases}")

    # ── RÉSUMÉ FINAL ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RÉSULTATS FINAUX")
    print("="*70)
    print(f"\n{'Critère':<30} {'Qwen 2.5 7B':>15} {'Meditron 7B':>15}")
    print("─"*62)

    for model_name in MODELS:
        r = results[model_name]
        avg_score   = round(sum(r["scores"])  / len(r["scores"]), 2)
        avg_time    = round(sum(r["times"])   / len(r["times"]),  1)
        avg_phrases = round(sum(r["phrases"]) / len(r["phrases"]), 1)
        results[model_name]["avg_score"]   = avg_score
        results[model_name]["avg_time"]    = avg_time
        results[model_name]["avg_phrases"] = avg_phrases

    print(f"{'Score pertinence (0-10)':<30} {results['Qwen 2.5 7B']['avg_score']:>15} {results['Meditron 7B']['avg_score']:>15}")
    print(f"{'Temps moyen (s)':<30} {results['Qwen 2.5 7B']['avg_time']:>15} {results['Meditron 7B']['avg_time']:>15}")
    print(f"{'Phrases moyennes':<30} {results['Qwen 2.5 7B']['avg_phrases']:>15} {results['Meditron 7B']['avg_phrases']:>15}")

    # Meilleur modèle
    best = max(MODELS, key=lambda m: results[m]["avg_score"])
    print(f"\n✅ Meilleur modèle : {best}")

    # Sauvegarde JSON
    out_path = Path("results_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n Résultats sauvegardés : {out_path}")


if __name__ == "__main__":
    main()