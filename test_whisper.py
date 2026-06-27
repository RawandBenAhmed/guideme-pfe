import imageio_ffmpeg
import os
from pathlib import Path

# Renomme l'exe en ffmpeg.exe pour que Whisper le trouve
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = Path(ffmpeg_exe).parent
ffmpeg_link = ffmpeg_dir / "ffmpeg.exe"

# Crée un lien symbolique ou copie
if not ffmpeg_link.exists():
    import shutil
    shutil.copy2(ffmpeg_exe, ffmpeg_link)
    print(f"ffmpeg copié : {ffmpeg_link}")

os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ["PATH"]
print(f"ffmpeg disponible : {ffmpeg_link}")

import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

print("Parle pendant 5 secondes...")
fs = 16000
duration = 5
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Enregistrement termine")

wav.write("test_audio.wav", fs, recording)
model = whisper.load_model("small")
MEDICAL_PROMPT = """
Procédure médicale de drainage thoracique.
Outils : stéthoscope, masque respiratoire, gants stériles, 
champ stérile fenêtré, compresse stérile, bétadine, 
seringue, xylocaïne, lame de bistouri, ciseau, 
drain thoracique, système de drainage.
Étapes : diagnostique, préparation, désinfection bétadine, 
anesthésie locale, incision cutanée, dissection sous-cutanée, 
insertion du drain, connexion bocal de drainage.
"""

result = model.transcribe(
    "test_audio.wav",
    language   = "fr",
    fp16       = False,
    initial_prompt = MEDICAL_PROMPT
)
text = result["text"]
print(f"Transcription : {text}")