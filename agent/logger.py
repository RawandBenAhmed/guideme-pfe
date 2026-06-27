"""
agent/logger.py
Enregistre toutes les interactions de la session dans un fichier JSON.
"""

import json
import time
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class Logger:
    """Enregistre les événements de la session dans logs/."""

    def __init__(self, session_id: str):
        self.session_id  = session_id
        self.log_dir     = config.LOG_DIR
        self.log_dir.mkdir(exist_ok=True)
        self.start_time  = time.time()
        self.events      = []
        self.log_path    = self.log_dir / f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"[LOG] Session {session_id} — log : {self.log_path.name}")

    # ============================================================
    # ENREGISTREMENT
    # ============================================================

    def log(self, event_type: str, data: dict = None):
        """Enregistre un événement."""
        event = {
            "timestamp":  datetime.now().isoformat(),
            "elapsed":    round(time.time() - self.start_time, 1),
            "event_type": event_type,
            "data":       data or {}
        }
        self.events.append(event)
        if config.LOG_TO_FILE:
            self._save()

    def log_user_message(self, message: str, intent: str):
        self.log("user_message", {"message": message, "intent": intent})

    def log_agent_response(self, response: str, operation_id: int):
        self.log("agent_response", {"response": response, "operation_id": operation_id})

    def log_step_start(self, operation_id: int, operation_name: str):
        self.log("step_start", {"operation_id": operation_id, "operation_name": operation_name})

    def log_step_complete(self, operation_id: int, duration: int, errors: int):
        self.log("step_complete", {"operation_id": operation_id, "duration_seconds": duration, "errors": errors})

    def log_error(self, operation_id: int, error_description: str):
        self.log("error", {"operation_id": operation_id, "description": error_description})

    def log_session_end(self, progress: dict):
        self.log("session_end", {"progress": progress})

    # ============================================================
    # SAUVEGARDE
    # ============================================================

    def _save(self):
        data = {
            "session_id":     self.session_id,
            "start_time":     datetime.fromtimestamp(self.start_time).isoformat(),
            "total_duration": round(time.time() - self.start_time, 1),
            "events":         self.events,
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        total = round(time.time() - self.start_time, 1)
        nb_messages = sum(1 for e in self.events if e["event_type"] == "user_message")
        nb_errors   = sum(1 for e in self.events if e["event_type"] == "error")
        return (
            f"Session {self.session_id} | "
            f"Durée : {total}s | "
            f"Messages : {nb_messages} | "
            f"Erreurs : {nb_errors}"
        )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("\n=== TEST LOGGER ===\n")

    lg = Logger("test_abc")
    lg.log_step_start(1, "Diagnostique")
    lg.log_user_message("quel outil je prends ?", "tool")
    lg.log_agent_response("Prends le stéthoscope et le masque respiratoire.", 1)
    lg.log_user_message("j'ai fini", "next")
    lg.log_step_complete(1, 28, 0)
    lg.log_step_start(2, "Préparation patient")
    lg.log_error(2, "mauvais outil utilisé")
    lg.log_session_end({"done": 2, "total": 8})

    print(lg.summary())
    print(f"Log sauvegardé : {lg.log_path}")