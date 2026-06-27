"""
benchmark_llm.py
Compare Qwen 2.5 7B (Ollama) vs Meditron 7B (Ollama) vs LLaMA 3.3 70B (Groq)
sur 10 questions médicales de drainage thoracique.
Lancer : python benchmark_llm.py
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

# ─── CONFIG ───────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = config.GROQ_API_KEY

MODELS = [
    {"name": "Qwen 2.5 7B",    "id": "qwen2.5:7b-instruct", "provider": "ollama"},
    {"name": "Meditron 7B",    "id": "meditron",             "provider": "ollama"},
    {"name": "LLaMA 3.3 70B",  "id": "llama-3.3-70b-versatile", "provider": "groq"},
]

SYSTEM_PROMPT = """Tu es GuideME, assistant pédagogique spécialisé UNIQUEMENT
dans la procédure de drainage thoracique.
Réponds en français, maximum 2 phrases, ton amical et pédagogique.
Utilise UNIQUEMENT les informations fournies dans la question."""

TEST_QUESTIONS = [
    {
        "id": 1, "etape": "Diagnostique",
        "question": "Tu es dans l'étape 1 : Diagnostique. Outils : masque respiratoire, gants, stéthoscope. Action : installer le masque au patient, enfiler les gants, ausculter les deux poumons. Que dois-je faire maintenant ?",
        "mots_cles": ["masque", "gants", "stéthoscope", "ausculter", "poumon"],
    },
    {
        "id": 2, "etape": "Préparation patient",
        "question": "Tu es dans l'étape 2 : Préparation patient. Outils : gants stériles, champ stérile fenêtré. Action : enfiler les gants, repérer le 5e espace intercostal, installer le champ stérile. Comment je procède ?",
        "mots_cles": ["gants", "champ", "stérile", "intercostal", "repérer"],
    },
    {
        "id": 3, "etape": "Désinfection Bétadine",
        "question": "Tu es dans l'étape 3 : Désinfection Bétadine. Outils : compresse stérile, Bétadine. Action : imbiber la compresse et désinfecter du centre vers la périphérie. Dans quel sens je désinfecte ?",
        "mots_cles": ["centre", "périphérie", "compresse", "bétadine", "désinfecter"],
    },
    {
        "id": 4, "etape": "Anesthésie locale",
        "question": "Tu es dans l'étape 4 : Anesthésie locale. Outils : seringue, Xylocaïne 1%. Action : charger la seringue et infiltrer la peau. Pourquoi aspirer avant d'injecter ?",
        "mots_cles": ["aspirer", "xylocaïne", "injection", "vasculaire", "sécurité"],
    },
    {
        "id": 5, "etape": "Incision cutanée",
        "question": "Tu es dans l'étape 5 : Incision cutanée. Outil : lame de bistouri. Action : réaliser une incision franche d'environ 1 cm. Quelle taille doit avoir l'incision ?",
        "mots_cles": ["1 cm", "incision", "franche", "bistouri", "peau"],
    },
    {
        "id": 6, "etape": "Dissection sous-cutanée",
        "question": "Tu es dans l'étape 6 : Dissection sous-cutanée. Outil : ciseau. Action : effectuer 3 mouvements de dissection pour atteindre l'espace intercostal. Combien de mouvements ?",
        "mots_cles": ["3", "dissection", "ciseau", "intercostal", "mouvements"],
    },
    {
        "id": 7, "etape": "Insertion du drain",
        "question": "Tu es dans l'étape 7 : Insertion du drain. Outil : drain thoracique. Action : insérer le drain et orienter vers l'apex. Vers où orienter le drain ?",
        "mots_cles": ["apex", "drain", "pleural", "orienter", "insérer"],
    },
    {
        "id": 8, "etape": "Connexion bocal",
        "question": "Tu es dans l'étape 8 : Connexion bocal. Outil : système de drainage. Action : connecter le drain et vérifier l'étanchéité. Comment vérifier que ça fonctionne ?",
        "mots_cles": ["bullage", "bocal", "étanchéité", "connexion", "vérifier"],
    },
    {
        "id": 9, "etape": "Mauvais outil",
        "question": "Tu es dans l'étape 3 : Désinfection Bétadine. Outils corrects : compresse stérile, Bétadine. L'étudiant a pris la seringue — c'est le mauvais outil. Que lui dire ?",
        "mots_cles": ["compresse", "bétadine", "mauvais", "désinfection", "prendre"],
    },
    {
        "id": 10, "etape": "Hors sujet",
        "question": "Quelle est la capitale de la France ?",
        "mots_cles": ["drainage", "spécialisé", "thoracique", "procédure"],
    },
]


def call_ollama(model_id, prompt):
    payload = {
        "model": model_id, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.3, "num_predict": 100, "num_ctx": 512}
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(OLLAMA_URL, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            r = json.loads(resp.read().decode())
            return r.get("response", "").strip(), round(time.time()-start, 1)
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def call_groq(model_id, prompt):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 150,
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(GROQ_URL, data=data,
                                  headers={
                                      "Content-Type": "application/json",
                                      "Authorization": f"Bearer {GROQ_API_KEY}",
                                      "User-Agent": "Mozilla/5.0",
                                  }, method="POST")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"].strip(), round(time.time()-start, 1)
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def call_model(model, question):
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion : {question}"
    if model["provider"] == "ollama":
        return call_ollama(model["id"], prompt)
    else:
        return call_groq(model["id"], prompt)


def score_response(response, mots_cles):
    r = response.lower()
    found = sum(1 for kw in mots_cles if kw.lower() in r)
    return round((found / len(mots_cles)) * 10)


def count_sentences(text):
    return len([s for s in text.replace("!",".").replace("?",".").split(".") if s.strip()])


def is_french(text):
    fr_words = ["je", "tu", "il", "nous", "vous", "les", "des", "une", "dans", "pour", "avec", "sur"]
    tl = text.lower()
    return sum(1 for w in fr_words if w in tl) >= 2


def main():
    print("\n" + "="*72)
    print("  BENCHMARK LLM — Drainage Thoracique")
    print("  Modèles : Qwen 2.5 7B | Meditron 7B | LLaMA 3.3 70B (Groq)")
    print("="*72)

    results = {m["name"]: {"scores":[], "times":[], "phrases":[], "french":[]}
               for m in MODELS}

    for q in TEST_QUESTIONS:
        print(f"\n{'─'*72}")
        print(f"Q{q['id']} — {q['etape']}")
        print(f"Question : {q['question'][:70]}...")

        for model in MODELS:
            response, elapsed = call_model(model, q["question"])
            score   = score_response(response, q["mots_cles"])
            phrases = count_sentences(response)
            french  = is_french(response)

            results[model["name"]]["scores"].append(score)
            results[model["name"]]["times"].append(elapsed)
            results[model["name"]]["phrases"].append(phrases)
            results[model["name"]]["french"].append(french)

            print(f"\n  [{model['name']}]")
            print(f"  Réponse : {response[:90]}...")
            print(f"  Score:{score}/10 | Temps:{elapsed}s | Phrases:{phrases} | Français:{'✅' if french else '❌'}")

    # ─── RÉSUMÉ ───────────────────────────────────────────────
    print("\n\n" + "="*72)
    print("  RÉSULTATS FINAUX — BENCHMARK")
    print("="*72)
    print(f"\n{'Critère':<28} {'Qwen 2.5 7B':>14} {'Meditron 7B':>14} {'LLaMA 3.3 70B':>14}")
    print("─" * 72)

    summary = {}
    for m in MODELS:
        r = results[m["name"]]
        summary[m["name"]] = {
            "score":   round(sum(r["scores"])  / len(r["scores"]),  2),
            "time":    round(sum(r["times"])   / len(r["times"]),   1),
            "phrases": round(sum(r["phrases"]) / len(r["phrases"]), 1),
            "french":  round(sum(r["french"])  / len(r["french"]) * 100),
        }

    names = [m["name"] for m in MODELS]
    print(f"{'Score pertinence (0-10)':<28} " +
          " ".join(f"{summary[n]['score']:>14}" for n in names))
    print(f"{'Temps moyen (s)':<28} " +
          " ".join(f"{summary[n]['time']:>14}" for n in names))
    print(f"{'Phrases moyennes':<28} " +
          " ".join(f"{summary[n]['phrases']:>14}" for n in names))
    print(f"{'Réponses en français (%)':<28} " +
          " ".join(f"{summary[n]['french']:>13}%" for n in names))

    best = max(names, key=lambda n: summary[n]["score"])
    print(f"\n✅ Meilleur modèle : {best}")
    print(f"   Score : {summary[best]['score']}/10")
    print(f"   Temps : {summary[best]['time']}s")

    # Sauvegarde
    out = Path("benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Résultats sauvegardés : {out}")


if __name__ == "__main__":
    main()