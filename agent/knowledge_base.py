"""
agent/knowledge_base.py
Charge et interroge les 3 fichiers JSON du projet.
C'est la source de vérité de l'agent.
"""

import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class KnowledgeBase:
    def __init__(self):
        self.tools        = {}
        self.operations   = {}
        self.messages     = {}
        self.agent_config = {}
        self._load()

    def _load(self):
        # --- tools.json ---
        with open(config.TOOLS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for tool in data["tools"]:
            self.tools[tool["tool_id"]] = tool
        print(f"[KB] {len(self.tools)} outil(s) chargé(s)")

        # --- operations.json ---
        with open(config.OPERATIONS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for op in data["operations"]:
            self.operations[op["operation_id"]] = op
        print(f"[KB] {len(self.operations)} opération(s) chargée(s)")

        # --- messages.json ---
        with open(config.MESSAGES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for msg in data["messages"]:
            self.messages[msg["operation_id"]] = msg
        print(f"[KB] {len(self.messages)} message(s) chargé(s)")

        # --- agent_config.json ---
        with open(config.AGENT_CONFIG_JSON, encoding="utf-8") as f:
            self.agent_config = json.load(f)
        print(f"[KB] Agent '{self.agent_config['agent']['name']}' chargé")

    # ============================================================
    # OPÉRATIONS
    # ============================================================

    def get_operation(self, operation_id: int) -> dict:
        op = self.operations.get(operation_id)
        if op is None:
            raise ValueError(f"Opération {operation_id} introuvable")
        return op

    def get_all_operations(self) -> list:
        return sorted(self.operations.values(), key=lambda x: x["operation_id"])

    def get_next_operation_id(self, operation_id: int):
        op = self.get_operation(operation_id)
        return op.get("next_operation")

    def get_operation_by_phase(self, phase: str) -> list:
        return [op for op in self.operations.values() if op.get("phase") == phase]

    # ============================================================
    # OUTILS
    # ============================================================

    def get_tool(self, tool_id: str) -> dict:
        tool = self.tools.get(tool_id)
        if tool is None:
            raise ValueError(f"Outil {tool_id} introuvable")
        return tool

    def get_tools_for_operation(self, operation_id: int) -> list:
        op = self.get_operation(operation_id)
        return [self.tools[tid] for tid in op["tool_ids"] if tid in self.tools]

    def get_all_tools(self) -> list:
        return list(self.tools.values())

    # ============================================================
    # MESSAGES
    # ============================================================

    def get_messages(self, operation_id: int) -> dict:
        msg = self.messages.get(operation_id)
        if msg is None:
            raise ValueError(f"Messages pour opération {operation_id} introuvables")
        return msg

    def get_message(self, operation_id: int, message_key: str) -> str:
        msg = self.get_messages(operation_id)
        return msg["messages"].get(message_key, "")

    # ============================================================
    # CONTEXTE COMPLET
    # ============================================================

    def get_context(self, operation_id: int) -> dict:
        op    = self.get_operation(operation_id)
        tools = self.get_tools_for_operation(operation_id)
        msgs  = self.get_messages(operation_id)

        return {
            "operation_id":        op["operation_id"],
            "operation_name":      op["operation_name"],
            "phase":               op["phase"],
            "phase_label":         op["phase_label"],
            "tool_ids":            op["tool_ids"],
            "tool_names":          op["tool_names"],
            "tool_count":          op["tool_count"],
            "next_operation":      op["next_operation"],
            "estimated_duration":  op.get("estimated_duration_seconds", 30),
            "completion_criteria": op["completion_criteria"],
            "common_errors":       op["common_errors"],
            "tools_detail":        tools,
            "message_intro":       msgs["messages"]["intro"],
            "message_tool_prompt": msgs["messages"]["tool_prompt"],
            "message_action":      msgs["messages"]["action"],
            "message_wrong_tool":  msgs["messages"]["wrong_tool"],
            "message_completion":  msgs["messages"]["completion"],
            "message_alert":       msgs["messages"]["alert"],
        }

    # ============================================================
    # UTILITAIRES
    # ============================================================

    def is_last_operation(self, operation_id: int) -> bool:
        return self.get_next_operation_id(operation_id) is None

    def is_valid_operation(self, operation_id: int) -> bool:
        return operation_id in self.operations

    def get_common_errors(self, operation_id: int) -> list:
        op = self.get_operation(operation_id)
        return op.get("common_errors", [])

    def get_agent_name(self) -> str:
        return self.agent_config.get("agent", {}).get("name", "Agent")

    def get_agent_config(self) -> dict:
        return self.agent_config

    def summary(self) -> str:
        return (
            f"Knowledge Base chargée :\n"
            f"  {len(self.tools)} outils\n"
            f"  {len(self.operations)} opérations\n"
            f"  {len(self.messages)} messages\n"
            f"  Agent : {self.get_agent_name()}\n"
            f"  Première opération : {config.FIRST_OPERATION_ID}\n"
            f"  Dernière opération : {config.LAST_OPERATION_ID}"
        )


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    kb = KnowledgeBase()
    print()
    print(kb.summary())
    print()

    op = kb.get_operation(1)
    print(f"Opération 1 : {op['operation_name']}")
    print(f"  Phase     : {op['phase_label']}")
    print(f"  Outils    : {op['tool_names']}")
    print(f"  Suivante  : {op['next_operation']}")
    print()

    tools = kb.get_tools_for_operation(1)
    print(f"Outils de l'opération 1 :")
    for t in tools:
        print(f"  [{t['tool_id']}] {t['tool_name']} — {t['description']}")
    print()

    intro = kb.get_message(1, "intro")
    print(f"Message intro étape 1 :\n  {intro}")
    print()

    ctx = kb.get_context(3)
    print(f"Contexte étape 3 : {ctx['operation_name']}")
    print(f"  Erreurs fréquentes :")
    for err in ctx["common_errors"]:
        print(f"    - {err}")
    print()

    print(f"Nom agent : {kb.get_agent_name()}")