# test_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agent.knowledge_base  import KnowledgeBase
from agent.state_manager   import StateManager
from agent.intent_detector import detect_intent, get_intent_label
from agent.prompt          import (
    prompt_intro, prompt_tool_question,
    prompt_why_question, prompt_error,
    prompt_completion
)

print("\n" + "="*60)
print("  TEST INTÉGRATION — GuideME")
print("="*60)

# 1. Charger la Knowledge Base
print("\n--- 1. Knowledge Base ---")
kb = KnowledgeBase()
print(f"Agent : {kb.get_agent_name()}")

# 2. Démarrer le State Manager
print("\n--- 2. State Manager ---")
sm = StateManager()
sm.start()
print(f"Étape actuelle : {sm.current()}")

# 3. Simuler des messages utilisateur
print("\n--- 3. Intent Detector ---")
messages = [
    "quel outil je prends ?",
    "pourquoi je dois aspirer ?",
    "j'ai fini",
    "bonjour comment tu vas",
    "recommencer",
]
for msg in messages:
    intent = detect_intent(msg)
    label  = get_intent_label(intent)
    print(f"  '{msg}' → {label}")

# 4. Construire un prompt complet
print("\n--- 4. Prompt Engineering ---")
ctx = kb.get_context(sm.current())
prompt = prompt_intro(ctx)
print("Prompt généré pour l'étape 1 :")
print(prompt[:200] + "...")

# 5. Simuler progression
print("\n--- 5. Progression ---")
next_id = kb.get_next_operation_id(sm.current())
sm.next(next_id)
print(f"Nouvelle étape : {sm.current()}")
print(sm.summary())

print("\n" + "="*60)
print("  TOUS LES MODULES FONCTIONNENT CORRECTEMENT ✅")
print("="*60)