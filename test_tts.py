import pyttsx3

engine = pyttsx3.init()

# Cherche une voix française
voices = engine.getProperty('voices')
for i, voice in enumerate(voices):
    print(f"{i} : {voice.name} — {voice.languages}")

# Essaie la première voix française disponible
for voice in voices:
    if 'fr' in voice.id.lower() or 'french' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        print(f"Voix utilisée : {voice.name}")
        break

engine.setProperty('rate', 150)
engine.say("Tu es dans l'étape 1 : Diagnostique. Prends le stéthoscope et ausculte les deux poumons.")
engine.runAndWait()