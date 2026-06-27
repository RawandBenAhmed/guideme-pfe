"""
app.py - GuideME Drainage Thoracique
"""

import os, sys, base64, uuid, shutil, threading, random, time
from pathlib import Path
from io import BytesIO

from flask import Flask, render_template, request, jsonify, session
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from agent.graph_knowledge import GraphKnowledge
from agent.guideme_graph   import build_agent, init_agent, initial_state
from agent.llm_connector   import LLMConnector
from agent.logger          import Logger

app = Flask(__name__)
app.secret_key = "guideme_secret_2026"

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
VR_DIR     = BASE_DIR / "data" / "processed_VR"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "models" / "drainage_cls" / "weights" / "best.pt"

STEP_MAP = {
    "anesthesie locale":           4,
    "connexion bocal de drainage": 8,
    "desinfection betadine":       3,
    "diagnostique":                1,
    "dissection sous-cutanee":     6,
    "incision cutanee":            5,
    "insertion du drain":          7,
    "preparation patient":         2,
}

STEP_FOLDERS = {
    "diagnostique":1,"preparation_patient":2,
    "desinfection_betadine":3,"anesthesie_locale":4,
    "incision_cutanee":5,"dissection_sous_cutanee":6,
    "insertion_drain":7,"connexion_bocal":8,
}

TOOL_KEYWORDS = {
    "stethoscope":        ["stéthoscope","stethoscope","stetho","auscultation"],
    "masque_respiratoire":["masque","respiratoire"],
    "gants":              ["gants","gant"],
    "champ_sterile":      ["champ","champ stérile"],
    "compresse_sterile":  ["compresse","compresse stérile"],
    "betadine":           ["bétadine","betadine","désinfectant"],
    "seringue":           ["seringue","injection"],
    "xylocaine":          ["xylocaïne","xylocaine","anesthésique"],
    "lame_bistouri":      ["bistouri","lame","scalpel"],
    "ciseau":             ["ciseau","ciseaux"],
    "drain":              ["drain","drain thoracique"],
    "systeme_drainage":   ["système","drainage","bocal"],
}

TOOL_IMAGES = {
    "stethoscope":"stéthoscope.jpg","masque_respiratoire":"masque respiratoire.jpg",
    "gants":"Gants.webp","champ_sterile":"Champs stérile fenêtré.webp",
    "compresse_sterile":"compresse stérile.jpg","betadine":"betadine.jpg",
    "seringue":"seringue.jpg","xylocaine":"Xylocaïne 1%.jpg",
    "lame_bistouri":"Lame de bistouri.PNG","ciseau":"ciseau.jpg",
    "drain":"drain.jpg","systeme_drainage":"systeme_drainage.jpg",
}

HORS_SUJET_KEYWORDS = [
    "pizza","recette","cuisine","football","sport","film","musique",
    "météo","temps","actualité","politique","argent","salaire",
    "capitale","pays","ville","géographie","histoire","guerre",
    "instagram","tiktok","youtube","réseaux","social",
    "blague","humour","joke","jeu","game",
]

WHISPER_PROMPT = (
    "drainage thoracique étape outil masque gants seringue bistouri "
    "drain betadine ciseau anesthésie incision dissection connexion bocal "
    "stéthoscope compresse xylocaïne préparation diagnostique"
)

gk = agent_graph = _llm_direct = None
sessions   = {}
tts_lock   = threading.Lock()
_img_cache = {}


def is_hors_sujet(message: str) -> bool:
    msg_lower = message.lower()
    for kw in HORS_SUJET_KEYWORDS:
        if kw in msg_lower:
            return True
    return False


def get_vr_images(folder):
    if folder not in _img_cache:
        d = VR_DIR / folder
        _img_cache[folder] = sorted(d.glob("*.jpg")) if d.exists() else []
    return _img_cache[folder]


def setup_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        d   = Path(exe).parent
        lnk = d / "ffmpeg.exe"
        if not lnk.exists(): shutil.copy2(exe, lnk)
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH","")
    except: pass


# ✅ FIX VOLUME : utilise Windows SAPI directement via win32com
# pour un volume fort et fiable, sans dépendre de pyttsx3
def speak_text(text):
    if not text:
        return
    def run():
        with tts_lock:
            try:
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")

                # Cherche la voix Hortense (française)
                voices = speaker.GetVoices()
                french_voice = None
                for i in range(voices.Count):
                    v = voices.Item(i)
                    desc = v.GetDescription().lower()
                    if any(k in desc for k in ["hortense", "french", "français", "julie", "fr-fr", "fr "]):
                        french_voice = v
                        break

                if french_voice:
                    speaker.Voice = french_voice

                # ✅ Volume max (0-100) et vitesse normale
                speaker.Volume = 100
                speaker.Rate   = 0   # -10 (lent) à +10 (rapide), 0 = normal

                speaker.Speak(text)

            except Exception as ex:
                print(f"[TTS win32] Erreur : {ex}")
                # Fallback pyttsx3 si win32com pas disponible
                try:
                    import pyttsx3
                    e = pyttsx3.init()
                    voices = e.getProperty('voices')
                    french_voice = None
                    for v in voices:
                        vid   = v.id.lower()
                        vname = v.name.lower()
                        if any(k in vid   for k in ['fr', 'hortense', 'julie', 'fr-fr']):
                            french_voice = v.id; break
                        if any(k in vname for k in ['french', 'français', 'france', 'hortense']):
                            french_voice = v.id; break
                    if french_voice:
                        e.setProperty('voice', french_voice)
                    e.setProperty('rate',   150)
                    e.setProperty('volume', 1.0)
                    e.say(text)
                    e.runAndWait()
                    e.stop()
                except Exception as ex2:
                    print(f"[TTS fallback] Erreur : {ex2}")

    threading.Thread(target=run, daemon=True).start()


def init_models():
    global gk, agent_graph, _llm_direct
    setup_ffmpeg()
    init_agent()
    gk          = GraphKnowledge()
    agent_graph = build_agent()
    _llm_direct = LLMConnector()
    print("[APP] Agent LangGraph prêt ✅")
    print(f"[APP] Modèle YOLO : {'✅ trouvé' if MODEL_PATH.exists() else '❌ absent'}")
    print("[APP] Chargement Whisper...")
    import whisper
    global whisper_model
    whisper_model = whisper.load_model("small")
    print("[APP] Pret ✅")


def get_session():
    sid = session.get("sid")
    if not sid or sid not in sessions:
        sid = str(uuid.uuid4())[:8]
        session["sid"] = sid
        sessions[sid] = {
            "step_id":          1,
            "score":            0,
            "total":            0,
            "current_quiz_step":1,
            "questions_posees": 0,
            "temps_etapes":     [],
            "step_start_time":  time.time(),
            "session_start":    time.time(),
            "lg":               Logger(sid),
        }
    return sessions[sid]


def img_to_b64(path, size=(800,600)):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except: return ""


def tool_img_b64(tool_name):
    tools_dir = BASE_DIR / "data" / "processed" / "tools"
    fname = TOOL_IMAGES.get(tool_name,"")
    if not fname: return ""
    p = tools_dir / fname
    if not p.exists():
        p = BASE_DIR / "data" / "raw" / "tools_classification" / fname
    if not p.exists(): return ""
    return img_to_b64(p, (300,300))


def detect_tools(text):
    tl = text.lower()
    return [t for t,kws in TOOL_KEYWORDS.items() if any(k in tl for k in kws)]


def get_step_tools_with_images(step_id):
    """Retourne les outils d'une étape avec leurs images."""
    tools = gk.get_tools_for_step(step_id)
    result_names  = []
    result_images = []

    for t in tools:
        tool_words = set(t["name"].lower()
                         .replace("é","e").replace("è","e")
                         .replace("â","a").replace("ê","e")
                         .replace("î","i").replace("ô","o")
                         .replace("û","u").replace("ç","c")
                         .split())

        best_key   = ""
        best_score = 0
        for k in TOOL_IMAGES:
            key_words = set(k.replace("_"," ").split())
            score     = len(tool_words & key_words)
            if score > best_score:
                best_score = score
                best_key   = k

        result_names.append(t["name"])
        result_images.append(tool_img_b64(best_key) if best_key else "")

    return result_names, result_images


def run_agent(message, step_id, session_id, score=0, total=0):
    state  = initial_state(message, step_id, session_id, score, total)
    result = agent_graph.invoke(state)
    return result


def calcul_niveau(score, total, questions, temps_etapes):
    precision   = (score / total * 100) if total > 0 else 0
    temps_moyen = sum(temps_etapes) / len(temps_etapes) if temps_etapes else 999
    if precision >= 80 and questions <= 5 and temps_moyen <= 25:
        return {"niveau":"Avancé","emoji":"🏆",
                "message":"Excellent ! Tu maîtrises parfaitement la procédure de drainage thoracique.",
                "color":"#10B981","precision":round(precision,1),
                "temps_moyen":round(temps_moyen,1),"questions":questions}
    elif precision >= 50:
        return {"niveau":"Intermédiaire","emoji":"📈",
                "message":"Bon travail ! Continue à pratiquer pour améliorer ta maîtrise.",
                "color":"#F59E0B","precision":round(precision,1),
                "temps_moyen":round(temps_moyen,1),"questions":questions}
    else:
        return {"niveau":"Débutant","emoji":"📚",
                "message":"Continue à t'entraîner, chaque essai t'aide à progresser !",
                "color":"#EF4444","precision":round(precision,1),
                "temps_moyen":round(temps_moyen,1),"questions":questions}


# ─── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    s      = get_session()
    step   = gk.get_step(s["step_id"])
    snames = {st["id"]: st["name"] for st in gk.get_all_steps()}
    return render_template("index.html",
                           operation=step,
                           step_names=snames,
                           progress={
                               "session_id":        session.get("sid",""),
                               "current_operation": s["step_id"],
                               "total_operations":  8,
                               "done":              s["step_id"] - 1,
                               "errors":            {},
                           })


@app.route("/random_image")
def random_image():
    s      = get_session()
    op_id  = s["current_quiz_step"]
    folder = [k for k,v in STEP_FOLDERS.items() if v == op_id][0]
    images = get_vr_images(folder)
    if not images:
        return jsonify({"error":"Pas d images"}), 404
    img_path = random.choice(images)
    s["current_quiz_step"] = op_id + 1 if op_id < 8 else 1
    step = gk.get_step(op_id)
    return jsonify({
        "image":      img_to_b64(img_path),
        "correct_id": op_id,
        "step_name":  step.get("name",""),
    })


@app.route("/check_step", methods=["POST"])
def check_step():
    s          = get_session()
    data       = request.get_json()
    clicked_id = int(data.get("clicked_id", 0))
    correct_id = int(data.get("correct_id", 0))
    s["total"] += 1

    elapsed = time.time() - s.get("step_start_time", time.time())
    s["temps_etapes"].append(round(elapsed, 1))
    s["step_start_time"] = time.time()

    correct     = clicked_id == correct_id
    step_data   = gk.get_step(correct_id)
    step_name   = step_data.get("name", "")
    tools       = gk.get_tools_for_step(correct_id)
    tool_names  = ", ".join([t["name"] for t in tools])
    step_action = step_data.get("message_action", "")

    if correct:
        s["score"] += 1
        prompt = (
            f"IMPORTANT : Réponds UNIQUEMENT en français. "
            f"Tu es GuideME. "
            f"L'étudiant a correctement identifié l'étape {correct_id} : {step_name}. "
            f"Félicite-le en une seule phrase très courte. "
            f"Juste un encouragement court."
        )
    else:
        prompt = (
            f"IMPORTANT : Réponds UNIQUEMENT en français. "
            f"Tu es GuideME. "
            f"L'étudiant a cliqué sur la mauvaise étape. "
            f"Dis-lui : Tu es dans l'étape {correct_id} : {step_name}. "
            f"Prends les outils : {tool_names}. "
            f"Action : {step_action}. "
            f"Maximum 2 phrases."
        )

    response = _llm_direct.call(prompt)
    speak_text(response)

    finished = (correct_id == 8 and correct)
    niveau   = None
    if finished:
        niveau = calcul_niveau(s["score"], s["total"],
                               s["questions_posees"], s["temps_etapes"])

    return jsonify({
        "correct":      correct,
        "correct_id":   correct_id,
        "correct_name": step_name,
        "message":      response,
        "score":        s["score"],
        "total":        s["total"],
        "tool_names":   [],
        "tool_images":  [],
        "finished":     finished,
        "niveau":       niveau,
    })


@app.route("/timeout_hint", methods=["POST"])
def timeout_hint():
    data       = request.get_json()
    correct_id = int(data.get("correct_id", 1))
    result     = run_agent("Que dois-je faire dans cette étape ?",
                           correct_id, session.get("sid",""))
    response = result.get("response","")
    speak_text(response)
    return jsonify({
        "message":     response,
        "tool_names":  [],
        "tool_images": [],
    })


@app.route("/chat", methods=["POST"])
def chat():
    s               = get_session()
    data            = request.get_json()
    msg             = data.get("message","").strip()
    current_step_id = int(data.get("current_step_id", s["step_id"]))
    if not msg:
        return jsonify({"error":"vide"}), 400

    s["questions_posees"] += 1

    # Hors sujet
    if is_hors_sujet(msg):
        response = "Je suis spécialisé uniquement dans la procédure de drainage thoracique. Pose-moi une question sur l'étape en cours."
        speak_text(response)
        return jsonify({
            "response":        response,
            "intent":          "hors_sujet",
            "tool_names":      [],
            "tool_images":     [],
            "next_step":       None,
            "next_image":      None,
            "next_correct_id": None,
            "finished":        False,
            "niveau":          None,
        })

    result   = run_agent(msg, current_step_id,
                         session.get("sid",""),
                         s["score"], s["total"])
    response = result.get("response","")
    intent   = result.get("intent","")

    next_step  = None
    next_image = None
    if intent == "next" and not result.get("finished"):
        new_id = result.get("step_id", current_step_id)
        s["step_id"]         = new_id
        s["step_start_time"] = time.time()
        next_step_data = gk.get_step(new_id)
        next_step = {"id": new_id, "name": next_step_data.get("name","")}
        folder = [k for k,v in STEP_FOLDERS.items() if v == new_id]
        if folder:
            images = get_vr_images(folder[0])
            if images:
                next_image = img_to_b64(random.choice(images))
                s["current_quiz_step"] = new_id + 1 if new_id < 8 else 1

    niveau = None
    if result.get("finished"):
        niveau = calcul_niveau(s["score"], s["total"],
                               s["questions_posees"], s["temps_etapes"])

    # ── Gestion des outils ──────────────────────────────────
    if intent == "tool":
        all_names, all_imgs = get_step_tools_with_images(current_step_id)

        # Cherche si un outil spécifique est mentionné dans le message
        msg_lower      = msg.lower()
        filtered_names = []
        filtered_imgs  = []

        for i, name in enumerate(all_names):
            name_words = name.lower().replace("é","e").replace("è","e")\
                             .replace("â","a").replace("ê","e")\
                             .replace("î","i").replace("ô","o")\
                             .replace("û","u").replace("ç","c").split()
            if any(w in msg_lower for w in name_words if len(w) > 3):
                filtered_names.append(name)
                filtered_imgs.append(all_imgs[i])

        # Si aucun outil spécifique détecté → tous les outils de l'étape
        if not filtered_names:
            filtered_names = all_names
            filtered_imgs  = all_imgs

        tool_names_list = filtered_names
        tool_imgs       = filtered_imgs
        # Pas de TTS pour les outils — juste les images

    else:
        tool_names_list = []
        tool_imgs       = []
        speak_text(response)

    return jsonify({
        "response":        response,
        "intent":          intent,
        "tool_names":      tool_names_list,
        "tool_images":     tool_imgs,
        "next_step":       next_step,
        "next_image":      next_image,
        "next_correct_id": next_step["id"] if next_step else None,
        "finished":        result.get("finished", False),
        "niveau":          niveau,
    })


@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"response":"Aucune image reçue.",
                        "tool_names":[],"tool_images":[]}), 400

    s            = get_session()
    file         = request.files["image"]
    current_step = int(request.form.get("current_step_id", s["step_id"]))

    tmp_path = UPLOAD_DIR / f"upload_{uuid.uuid4().hex[:8]}.jpg"
    file.save(str(tmp_path))

    try:
        if MODEL_PATH.exists():
            from ultralytics import YOLO
            model  = YOLO(str(MODEL_PATH))
            result = model.predict(str(tmp_path), verbose=False)[0]
            top1   = result.probs.top1
            conf   = float(result.probs.top1conf)
            names  = result.names

            predicted_class = names[top1]
            predicted_norm  = predicted_class.lower().strip()
            step_id = STEP_MAP.get(predicted_norm, 0)

            print(f"[YOLO] Prédit : '{predicted_class}' → étape {step_id} (conf={conf:.2f})")

            if step_id > 0:
                step_data  = gk.get_step(step_id)
                step_name  = step_data.get("name", predicted_class)
                response   = f"Étape {step_id} : {step_name}."
            else:
                response   = "Je n'ai pas pu identifier l'étape. Montre une image plus claire."
        else:
            step_data  = gk.get_step(current_step)
            step_name  = step_data.get("name", "")
            response   = f"Étape {current_step} : {step_name}."

        speak_text(response)

        return jsonify({
            "response":    response,
            "tool_names":  [],
            "tool_images": [],
        })

    except Exception as e:
        print(f"[ANALYZE_IMAGE] Erreur : {e}")
        return jsonify({"response": f"Erreur : {str(e)}",
                        "tool_names":[], "tool_images":[]}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.route("/niveau", methods=["GET"])
def get_niveau():
    s = get_session()
    return jsonify(calcul_niveau(s["score"], s["total"],
                                 s["questions_posees"], s["temps_etapes"]))


@app.route("/transcribe", methods=["POST"])
def transcribe():
    import json as _json, urllib.request, urllib.error, config
    if "audio" not in request.files:
        return jsonify({"text":"","tool_names":[],"tool_images":[]}), 200
    audio_bytes = request.files["audio"].read()
    boundary = "----GuideMEBoundary"
    def field(n,v):
        return (f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="{n}"\r\n\r\n{v}\r\n').encode()
    def ffile(n,fn,ct,d):
        return (f'--{boundary}\r\nContent-Disposition: form-data; '
                f'name="{n}"; filename="{fn}"\r\n'
                f'Content-Type: {ct}\r\n\r\n').encode() + d + b'\r\n'
    body = (ffile("file","audio.webm","audio/webm",audio_bytes) +
            field("model","whisper-large-v3") +
            field("language","fr") +
            field("prompt", WHISPER_PROMPT) +
            field("response_format","json") +
            f'--{boundary}--\r\n'.encode())
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        data=body,
        headers={"Authorization":f"Bearer {config.GROQ_API_KEY}",
                 "Content-Type":f"multipart/form-data; boundary={boundary}",
                 "User-Agent":"Mozilla/5.0"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = _json.loads(resp.read().decode()).get("text","").strip()
    except Exception as e:
        print(f"[WHISPER] {e}"); text = ""
    print(f"[WHISPER] {text}")
    return jsonify({"text":text,"tool_names":[],"tool_images":[]})


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid and sid in sessions: del sessions[sid]
    session.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_models()
    print("\n GuideME LangGraph — http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)