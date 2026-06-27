"""
benchmark_llm_v2.py
Évaluation complète : BLEU, ROUGE, cohérence médicale,
format, latence — sur 3 modèles LLM
"""

import json, time, urllib.request, urllib.error
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    METRICS_OK = True
except:
    print("[WARN] pip install nltk rouge-score")
    METRICS_OK = False

OLLAMA_URL = "http://localhost:11434/api/generate"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"

MODELS = [
    {"name": "Qwen 2.5 7B",   "id": "qwen2.5:7b-instruct",      "provider": "ollama"},
    {"name": "Meditron 7B",   "id": "meditron",                  "provider": "ollama"},
    {"name": "LLaMA 3.3 70B", "id": "llama-3.3-70b-versatile",  "provider": "groq"},
]

SYSTEM = """Tu es GuideME, assistant pédagogique spécialisé UNIQUEMENT
dans la procédure de drainage thoracique.
Réponds en français, maximum 2 phrases."""

# Questions + réponses de référence idéales
TEST_CASES = [
    {
        "id": 1, "etape": "Diagnostique",
        "question": "Étape 1 : Diagnostique. Outils : masque respiratoire, gants, stéthoscope. Que faire ?",
        "reference": "Installe le masque respiratoire au patient, enfile les gants stériles et ausculte les deux poumons avec le stéthoscope pour détecter tout épanchement.",
        "mots_cles_medicaux": ["masque", "gants", "stéthoscope", "ausculter", "poumon"],
        "nb_phrases_attendu": 2,
    },
    {
        "id": 2, "etape": "Désinfection Bétadine",
        "question": "Étape 3 : Désinfection Bétadine. Outils : compresse, bétadine. Dans quel sens désinfecter ?",
        "reference": "Imprègne la compresse stérile de bétadine et désinfecte en effectuant des mouvements circulaires du centre vers la périphérie.",
        "mots_cles_medicaux": ["centre", "périphérie", "compresse", "bétadine", "circulaire"],
        "nb_phrases_attendu": 2,
    },
    {
        "id": 3, "etape": "Anesthésie locale",
        "question": "Étape 4 : Anesthésie locale. Outils : seringue, xylocaïne. Pourquoi aspirer avant d'injecter ?",
        "reference": "Il faut aspirer avant d'injecter pour vérifier que l'aiguille n'est pas dans un vaisseau sanguin et éviter une injection intravasculaire accidentelle de xylocaïne.",
        "mots_cles_medicaux": ["aspirer", "vaisseau", "intravasculaire", "xylocaïne", "aiguille"],
        "nb_phrases_attendu": 2,
    },
    {
        "id": 4, "etape": "Incision cutanée",
        "question": "Étape 5 : Incision cutanée. Outil : bistouri. Quelle taille d'incision ?",
        "reference": "Réalise une incision franche et nette d'environ 1 cm avec la lame de bistouri au niveau du site d'insertion du drain.",
        "mots_cles_medicaux": ["1 cm", "incision", "franche", "bistouri", "drain"],
        "nb_phrases_attendu": 2,
    },
    {
        "id": 5, "etape": "Insertion drain",
        "question": "Étape 7 : Insertion drain. Outil : drain thoracique. Vers où orienter ?",
        "reference": "Insère le drain thoracique dans la cavité pleurale et oriente-le vers l'apex du poumon pour assurer un drainage optimal.",
        "mots_cles_medicaux": ["apex", "drain", "pleurale", "cavité", "orienter"],
        "nb_phrases_attendu": 2,
    },
    {
        "id": 6, "etape": "Hors sujet",
        "question": "Quelle est la capitale de la France ?",
        "reference": "Je suis spécialisé uniquement dans la procédure de drainage thoracique et ne peux pas répondre à cette question.",
        "mots_cles_medicaux": ["drainage", "spécialisé", "thoracique"],
        "nb_phrases_attendu": 1,
    },
]


def call_ollama(model_id, prompt):
    payload = {
        "model": model_id, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.3, "num_predict": 120, "num_ctx": 512}
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            r = json.loads(resp.read().decode())
            return r.get("response", "").strip(), round(time.time()-start, 2)
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def call_groq(model_id, prompt):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3, "max_tokens": 120,
    }
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        GROQ_URL, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.GROQ_API_KEY}",
            "User-Agent": "Mozilla/5.0",
        }, method="POST"
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read().decode())
            return r["choices"][0]["message"]["content"].strip(), round(time.time()-start, 2)
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def compute_bleu(reference, hypothesis):
    if not METRICS_OK:
        return 0.0
    ref_tokens  = reference.lower().split()
    hyp_tokens  = hypothesis.lower().split()
    smoothie    = SmoothingFunction().method4
    return round(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie), 3)


def compute_rouge(reference, hypothesis):
    if not METRICS_OK:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores['rouge1'].fmeasure, 3),
        "rouge2": round(scores['rouge2'].fmeasure, 3),
        "rougeL": round(scores['rougeL'].fmeasure, 3),
    }


def compute_medical_accuracy(response, mots_cles):
    r = response.lower()
    found = sum(1 for kw in mots_cles if kw.lower() in r)
    return round((found / len(mots_cles)) * 10, 1)


def count_sentences(text):
    return len([s for s in text.replace("!",".").replace("?",".").split(".") if s.strip()])


def is_french(text):
    fr = ["je","tu","il","nous","les","des","une","dans","pour","avec","sur","prends","dois"]
    return sum(1 for w in fr if w in text.lower()) >= 2


def format_respected(response, expected_nb):
    nb = count_sentences(response)
    return nb <= expected_nb + 1


def main():
    print("\n" + "="*70)
    print("  BENCHMARK LLM V2 — Métriques complètes")
    print("  BLEU | ROUGE | Précision médicale | Format | Latence")
    print("="*70)

    all_results = {m["name"]: {
        "bleu": [], "rouge1": [], "rouge2": [], "rougeL": [],
        "medical": [], "format": [], "french": [], "latence": []
    } for m in MODELS}

    for tc in TEST_CASES:
        print(f"\n{'─'*70}")
        print(f"Q{tc['id']} — {tc['etape']}")
        print(f"Question  : {tc['question'][:65]}...")
        print(f"Référence : {tc['reference'][:65]}...")

        for model in MODELS:
            prompt   = f"{SYSTEM}\n\nQuestion : {tc['question']}"
            if model["provider"] == "ollama":
                response, latence = call_ollama(model["id"], prompt)
            else:
                response, latence = call_groq(model["id"], prompt)

            bleu    = compute_bleu(tc["reference"], response)
            rouge   = compute_rouge(tc["reference"], response)
            medical = compute_medical_accuracy(response, tc["mots_cles_medicaux"])
            fmt     = format_respected(response, tc["nb_phrases_attendu"])
            french  = is_french(response)

            all_results[model["name"]]["bleu"].append(bleu)
            all_results[model["name"]]["rouge1"].append(rouge["rouge1"])
            all_results[model["name"]]["rouge2"].append(rouge["rouge2"])
            all_results[model["name"]]["rougeL"].append(rouge["rougeL"])
            all_results[model["name"]]["medical"].append(medical)
            all_results[model["name"]]["format"].append(1 if fmt else 0)
            all_results[model["name"]]["french"].append(1 if french else 0)
            all_results[model["name"]]["latence"].append(latence)

            print(f"\n  [{model['name']}]")
            print(f"  Réponse : {response[:80]}...")
            print(f"  BLEU:{bleu} | R1:{rouge['rouge1']} R2:{rouge['rouge2']} RL:{rouge['rougeL']}")
            print(f"  Médical:{medical}/10 | Format:{'✅' if fmt else '❌'} | FR:{'✅' if french else '❌'} | Temps:{latence}s")

    # ─── RÉSUMÉ ───────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  RÉSULTATS FINAUX")
    print("="*70)

    avg = {}
    for name, r in all_results.items():
        avg[name] = {
            "BLEU":    round(sum(r["bleu"])   / len(r["bleu"]),   3),
            "ROUGE-1": round(sum(r["rouge1"]) / len(r["rouge1"]), 3),
            "ROUGE-2": round(sum(r["rouge2"]) / len(r["rouge2"]), 3),
            "ROUGE-L": round(sum(r["rougeL"]) / len(r["rougeL"]), 3),
            "Médical": round(sum(r["medical"])/ len(r["medical"]), 1),
            "Format%": round(sum(r["format"]) / len(r["format"]) * 100),
            "Français%":round(sum(r["french"]) / len(r["french"])* 100),
            "Latence": round(sum(r["latence"])/ len(r["latence"]), 2),
        }

    names = [m["name"] for m in MODELS]
    print(f"\n{'Métrique':<18} {'Qwen 2.5 7B':>14} {'Meditron 7B':>14} {'LLaMA 3.3 70B':>14}")
    print("─" * 62)
    for k in ["BLEU","ROUGE-1","ROUGE-2","ROUGE-L","Médical","Format%","Français%","Latence"]:
        print(f"{k:<18} " + " ".join(f"{avg[n][k]:>14}" for n in names))

    best = max(names, key=lambda n: avg[n]["BLEU"] + avg[n]["Médical"]/10)
    print(f"\n✅ Meilleur modèle global : {best}")

    with open("benchmark_llm_v2_results.json","w",encoding="utf-8") as f:
        json.dump(avg, f, ensure_ascii=False, indent=2)
    print("📄 Résultats sauvegardés : benchmark_llm_v2_results.json")


if __name__ == "__main__":
    main()