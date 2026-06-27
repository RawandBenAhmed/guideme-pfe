"""
benchmark_whisper.py
Compare Whisper small (local) vs Groq Whisper large-v3 (cloud)
sur des enregistrements audio en français.
Lancer : python benchmark_whisper.py
"""

import json
import time
import urllib.request
import urllib.error
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import config

GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_API_KEY     = config.GROQ_API_KEY

# Phrases de test à lire à voix haute
TEST_PHRASES = [
    "donne moi le stéthoscope",
    "je suis dans l'étape de désinfection bétadine",
    "prends la lame de bistouri",
    "quels outils je dois utiliser pour cette étape",
    "j'ai terminé l'anesthésie locale",
    "donne moi la compresse stérile et la bétadine",
    "je veux passer à l'étape suivante",
    "comment je désinfecte avec la bétadine",
]

DURATION = 5  # secondes d'enregistrement
FS       = 16000


def record_audio(phrase, duration=DURATION):
    print(f"\n  🎤 Lis cette phrase à voix haute : \"{phrase}\"")
    input("     [Appuie sur ENTRÉE pour commencer l'enregistrement]")
    print(f"     ⏺️  Enregistrement pendant {duration} secondes...")
    recording = sd.rec(int(duration * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    print("     ⏹️  Enregistrement terminé")
    return recording


def save_wav(recording):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, FS, recording)
    return tmp.name


def transcribe_whisper_local(wav_path):
    """Transcription avec Whisper small local."""
    try:
        import whisper

        # Setup ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = Path(ffmpeg_exe).parent
            ffmpeg_lnk = ffmpeg_dir / "ffmpeg.exe"
            if not ffmpeg_lnk.exists():
                shutil.copy2(ffmpeg_exe, ffmpeg_lnk)
            os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
        except:
            pass

        model  = whisper.load_model("small")
        start  = time.time()
        result = model.transcribe(wav_path, language="fr", fp16=False)
        elapsed = round(time.time() - start, 2)
        return result["text"].strip(), elapsed
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def transcribe_groq_whisper(wav_path):
    """Transcription avec Groq Whisper large-v3."""
    with open(wav_path, 'rb') as f:
        audio_bytes = f.read()

    boundary = "----GuideMEBoundary"

    def field(name, value):
        return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()

    def file_field(name, fname, ctype, data):
        return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n').encode() + data + b'\r\n'

    body = (
        file_field("file", "audio.wav", "audio/wav", audio_bytes) +
        field("model", "whisper-large-v3") +
        field("language", "fr") +
        field("response_format", "json") +
        f'--{boundary}--\r\n'.encode()
    )

    req = urllib.request.Request(
        GROQ_WHISPER_URL, data=body,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  f"multipart/form-data; boundary={boundary}",
            "User-Agent":    "Mozilla/5.0",
        }, method="POST"
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            r       = json.loads(resp.read().decode())
            elapsed = round(time.time() - start, 2)
            return r.get("text", "").strip(), elapsed
    except Exception as e:
        return f"ERREUR: {e}", 0.0


def similarity_score(ref, hyp):
    """Score de similarité simple basé sur les mots communs."""
    ref_words = set(ref.lower().split())
    hyp_words = set(hyp.lower().split())
    if not ref_words:
        return 0
    common = ref_words & hyp_words
    return round(len(common) / len(ref_words) * 100)


def main():
    print("\n" + "="*65)
    print("  BENCHMARK WHISPER — Reconnaissance Vocale")
    print("  Modèles : Whisper small (local) vs Groq Whisper large-v3")
    print("="*65)
    print("\nTu vas lire 8 phrases médicales.")
    print("Chaque phrase sera transcrite par les 2 modèles.\n")

    results = {
        "Whisper small (local)":    {"scores": [], "times": [], "texts": []},
        "Groq Whisper large-v3":    {"scores": [], "times": [], "texts": []},
    }

    for i, phrase in enumerate(TEST_PHRASES):
        print(f"\n{'─'*65}")
        print(f"Test {i+1}/{len(TEST_PHRASES)}")

        recording = record_audio(phrase)
        wav_path  = save_wav(recording)

        # Whisper local
        print("\n  [Whisper small local] transcription...")
        text_local, t_local = transcribe_whisper_local(wav_path)
        score_local = similarity_score(phrase, text_local)
        results["Whisper small (local)"]["scores"].append(score_local)
        results["Whisper small (local)"]["times"].append(t_local)
        results["Whisper small (local)"]["texts"].append(text_local)
        print(f"  Texte  : {text_local}")
        print(f"  Score  : {score_local}% | Temps : {t_local}s")

        # Groq Whisper
        print("\n  [Groq Whisper large-v3] transcription...")
        text_groq, t_groq = transcribe_groq_whisper(wav_path)
        score_groq = similarity_score(phrase, text_groq)
        results["Groq Whisper large-v3"]["scores"].append(score_groq)
        results["Groq Whisper large-v3"]["times"].append(t_groq)
        results["Groq Whisper large-v3"]["texts"].append(text_groq)
        print(f"  Texte  : {text_groq}")
        print(f"  Score  : {score_groq}% | Temps : {t_groq}s")

        os.unlink(wav_path)

    # ─── RÉSUMÉ ───────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  RÉSULTATS FINAUX — BENCHMARK WHISPER")
    print("="*65)
    print(f"\n{'Critère':<30} {'Whisper small':>16} {'Groq large-v3':>16}")
    print("─" * 65)

    for name, r in results.items():
        r["avg_score"] = round(sum(r["scores"]) / len(r["scores"]), 1)
        r["avg_time"]  = round(sum(r["times"])  / len(r["times"]),  2)

    local = results["Whisper small (local)"]
    groq  = results["Groq Whisper large-v3"]

    print(f"{'Score similarité (%)':<30} {local['avg_score']:>15}% {groq['avg_score']:>15}%")
    print(f"{'Temps moyen (s)':<30} {local['avg_time']:>16} {groq['avg_time']:>16}")

    best = "Groq Whisper large-v3" if groq["avg_score"] >= local["avg_score"] else "Whisper small (local)"
    print(f"\n✅ Meilleur modèle : {best}")

    out = Path("benchmark_whisper_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Résultats sauvegardés : {out}")


if __name__ == "__main__":
    main()