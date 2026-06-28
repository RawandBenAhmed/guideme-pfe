# GuideME 🎓

Agent pédagogique IA pour simulation VR de drainage thoracique.  
Projet de Fin d'Études — École Polytechnique de Sousse | Partenaire : Tadreex

## 🎬 Démonstration

[![GuideME Demo](https://img.youtube.com/vi/RFMMB7UhkFg/0.jpg)](https://youtu.be/RFMMB7UhkFg)

## 🧠 Stack technique

| Composant | Technologie |
|-----------|------------|
| Classification gestes | YOLOv8-cls — 99.69% Top-1 |
| LLM | LLaMA 3.3 70B via Groq API |
| Reconnaissance vocale | Whisper large-v3 |
| Orchestration agent | LangGraph (6 nodes) |
| Graphe de connaissances | NetworkX (20 noeuds, 33 arêtes) |
| Interface | Flask |

## 📦 Installation

```bash
pip install -r requirements.txt
python app.py
```

## 👩‍💻 Auteure
**Rawand Ben Ahmed** — Ingénieure Data Science  
École Polytechnique de Sousse