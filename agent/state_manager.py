"""
agent/state_manager.py
Gère l'état de la session — où en est l'utilisateur,
quelles étapes sont terminées, combien d'erreurs, etc.
"""

import time
import uuid
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class StateManager:
    """
    Mémorise l'état complet de la session pendant toute
    la durée de l'expérience VR.
    """

    def __init__(self):
        self.session_id         = str(uuid.uuid4())[:8]
        self.current_operation  = config.FIRST_OPERATION_ID
        self.completed          = []
        self.errors             = {}   # operation_id → nb erreurs
        self.start_time         = None
        self.step_start_time    = None
        self.step_durations     = {}   # operation_id → durée en secondes
        self.is_finished        = False
        self.is_started         = False

    # ============================================================
    # DÉMARRAGE / FIN
    # ============================================================

    def start(self):
        """Démarre la session à l'étape 1."""
        self.start_time      = time.time()
        self.step_start_time = time.time()
        self.is_started      = True
        self.current_operation = config.FIRST_OPERATION_ID
        print(f"[STATE] Session {self.session_id} démarrée — étape {self.current_operation}")

    def finish(self):
        """Termine la session."""
        self.is_finished = True
        total = int(time.time() - self.start_time)
        print(f"[STATE] Session terminée en {total}s")

    def reset(self):
        """Réinitialise complètement la session."""
        self.__init__()
        print(f"[STATE] Session réinitialisée")

    # ============================================================
    # PROGRESSION
    # ============================================================

    def next(self, next_operation_id: int | None):
        """
        Passe à l'étape suivante.
        next_operation_id : ID de la prochaine étape (depuis KB)
        """
        # Enregistre la durée de l'étape actuelle
        duration = int(time.time() - self.step_start_time)
        self.step_durations[self.current_operation] = duration

        # Marque l'étape actuelle comme terminée
        if self.current_operation not in self.completed:
            self.completed.append(self.current_operation)

        print(f"[STATE] Étape {self.current_operation} terminée en {duration}s")

        if next_operation_id is None:
            self.finish()
            return

        # Passe à l'étape suivante
        self.current_operation = next_operation_id
        self.step_start_time   = time.time()
        print(f"[STATE] Passage à l'étape {self.current_operation}")

    def complete_current(self, next_operation_id: int | None):
        """Alias de next() — marque l'étape actuelle terminée."""
        self.next(next_operation_id)

    # ============================================================
    # ERREURS
    # ============================================================

    def add_error(self, operation_id: int = None):
        """Enregistre une erreur pour l'étape actuelle."""
        op_id = operation_id or self.current_operation
        self.errors[op_id] = self.errors.get(op_id, 0) + 1
        nb = self.errors[op_id]
        print(f"[STATE] Erreur #{nb} à l'étape {op_id}")
        return nb

    def get_errors(self, operation_id: int = None) -> int:
        """Retourne le nombre d'erreurs pour une étape."""
        op_id = operation_id or self.current_operation
        return self.errors.get(op_id, 0)

    def too_many_errors(self, operation_id: int = None) -> bool:
        """Retourne True si le nombre d'erreurs dépasse le maximum."""
        return self.get_errors(operation_id) >= config.MAX_ERRORS_PER_STEP

    # ============================================================
    # GETTERS
    # ============================================================

    def current(self) -> int:
        """Retourne l'ID de l'étape actuelle."""
        return self.current_operation

    def is_complete(self, operation_id: int) -> bool:
        """Retourne True si une étape est terminée."""
        return operation_id in self.completed

    def get_elapsed_time(self) -> int:
        """Retourne le temps écoulé depuis le début de la session en secondes."""
        if not self.start_time:
            return 0
        return int(time.time() - self.start_time)

    def get_step_elapsed(self) -> int:
        """Retourne le temps écoulé depuis le début de l'étape actuelle."""
        if not self.step_start_time:
            return 0
        return int(time.time() - self.step_start_time)

    def get_progress(self) -> dict:
        """Retourne un résumé de la progression."""
        total = config.LAST_OPERATION_ID - config.FIRST_OPERATION_ID + 1
        done  = len(self.completed)
        return {
            "session_id":        self.session_id,
            "current_operation": self.current_operation,
            "completed":         self.completed,
            "total_operations":  total,
            "done":              done,
            "remaining":         total - done,
            "percent":           int((done / total) * 100),
            "elapsed_seconds":   self.get_elapsed_time(),
            "errors":            self.errors,
            "is_finished":       self.is_finished,
        }

    def summary(self) -> str:
        """Retourne un résumé lisible de la session."""
        p = self.get_progress()
        lines = [
            f"Session     : {p['session_id']}",
            f"Progression : {p['done']}/{p['total_operations']} étapes ({p['percent']}%)",
            f"Étape actuelle : {p['current_operation']}",
            f"Temps écoulé : {p['elapsed_seconds']}s",
            f"Erreurs : {sum(p['errors'].values())}",
            f"Terminé : {'Oui' if p['is_finished'] else 'Non'}",
        ]
        return "\n".join(lines)


# ============================================================
# TEST RAPIDE
# ============================================================
if __name__ == "__main__":
    sm = StateManager()

    print("=== Test StateManager ===\n")

    # Démarrage
    sm.start()
    print(f"Étape actuelle : {sm.current()}")
    print()

    # Simule une erreur
    sm.add_error()
    sm.add_error()
    print(f"Erreurs étape 1 : {sm.get_errors()}")
    print(f"Trop d'erreurs ? : {sm.too_many_errors()}")
    print()

    # Passe à l'étape 2
    time.sleep(1)
    sm.next(2)
    print(f"Étape actuelle : {sm.current()}")
    print()

    # Passe à l'étape 3
    time.sleep(1)
    sm.next(3)
    print(f"Étape actuelle : {sm.current()}")
    print()

    # Résumé
    print("=== Résumé ===")
    print(sm.summary())
    print()

    # Progression
    print("=== Progression ===")
    p = sm.get_progress()
    print(f"  Complétées : {p['completed']}")
    print(f"  Restantes  : {p['remaining']}")
    print(f"  Pourcentage: {p['percent']}%")