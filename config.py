"""
config.py
Configuration globale du projet GuideME.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# ============================================================
# LLM — Groq API
# ============================================================
LLM_PROVIDER  = "groq"                        # "groq" ou "ollama"
LLM_MODEL     = "llama-3.3-70b-versatile"     # modèle Groq
GROQ_API_KEY  = "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"             # ← remplace par ta clé gsk_...
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"

# Ollama (backup local)
LLM_BASE_URL  = "http://localhost:11434"
LLM_TIMEOUT   = 120

# ============================================================
# CHEMINS
# ============================================================
DATA_DIR      = BASE_DIR / "data"
MODELS_DIR    = BASE_DIR / "models"
JSON_DIR      = BASE_DIR / "json"
LOGS_DIR      = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================
# MODÈLES YOLO
# ============================================================
CLS_MODEL_PATH = MODELS_DIR / "drainage_cls"       / "weights" / "best.pt"
DET_MODEL_PATH = MODELS_DIR / "drainage_detection"  / "weights" / "best.pt"

# ============================================================
# AGENT
# ============================================================
AGENT_MAX_SENTENCES = 2
AGENT_TEMPERATURE   = 0.3
AGENT_MAX_TOKENS    = 150
# ============================================================
# JSON FILES
# ============================================================
TOOLS_JSON        = JSON_DIR / "tools.json"
OPERATIONS_JSON   = JSON_DIR / "operations.json"
MESSAGES_JSON     = JSON_DIR / "messages.json"
AGENT_CONFIG_JSON = JSON_DIR / "agent_config.json"

# ============================================================
# AGENT SETTINGS
# ============================================================
FIRST_OPERATION_ID  = 1
TOTAL_OPERATIONS    = 8
MAX_ERRORS_PER_OP   = 5
LOG_DIR  = LOGS_DIR   # alias pour logger.py
LAST_OPERATION_ID   = 8
LOG_TO_FILE = True