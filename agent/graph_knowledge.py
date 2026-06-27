"""
agent/graph_knowledge.py
Base de connaissance orientée graphe (NetworkX)
Remplace knowledge_base.py

Structure du graphe :
  Nœuds :
    step_1 ... step_8   (type='step')  — étapes chirurgicales
    tool_T01 ... tool_T12 (type='tool') — instruments médicaux

  Arêtes :
    (step_i) --[PRECEDE]------> (step_i+1)
    (step_i) --[NECESSITE]----> (tool_Tx)
    (tool_Tx)--[UTILISE_DANS]-> (step_i)
"""

import json
import networkx as nx
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config


class GraphKnowledge:
    """
    Base de connaissance sous forme de graphe orienté NetworkX.
    Charge les données JSON et les structure en nœuds et arêtes.
    """

    def __init__(self):
        self.G      = nx.DiGraph()
        self._load()
        print(f"[GRAPH] {self.G.number_of_nodes()} nœuds — "
              f"{self.G.number_of_edges()} arêtes")

    # ─────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────

    def _load(self):
        """Charge les JSON et construit le graphe."""
        base = Path(__file__).parent.parent / "json"

        with open(base / "tools.json",      encoding="utf-8") as f:
            tools_raw = json.load(f)
        with open(base / "operations.json", encoding="utf-8") as f:
            ops_raw   = json.load(f)
        with open(base / "messages.json",   encoding="utf-8") as f:
            msgs_raw  = json.load(f)

        tools      = tools_raw.get("tools", [])
        operations = ops_raw.get("operations", [])
        messages   = {m["operation_id"]: m["messages"]
                      for m in msgs_raw.get("messages", [])}

        self._add_tool_nodes(tools)
        self._add_step_nodes(operations, messages)
        self._add_edges(operations)

    def _add_tool_nodes(self, tools: list):
        """Ajoute les nœuds instruments médicaux."""
        for tool in tools:
            tid = tool["tool_id"]
            self.G.add_node(
                f"tool_{tid}",
                type        = "tool",
                id          = tid,
                name        = tool["tool_name"],
                image       = tool.get("tool_image", ""),
                description = tool.get("description", ""),
            )

    def _add_step_nodes(self, operations: list, messages: dict):
        """Ajoute les nœuds étapes chirurgicales."""
        for op in operations:
            oid  = op["operation_id"]
            msgs = messages.get(oid, {})
            self.G.add_node(
                f"step_{oid}",
                type          = "step",
                id            = oid,
                name          = op["operation_name"],
                phase         = op.get("phase_label", ""),
                tool_names    = op.get("tool_names", ""),
                duration      = op.get("estimated_duration", 30),
                message_intro = msgs.get("intro", ""),
                message_tool  = msgs.get("tool_prompt", ""),
                message_action= msgs.get("action", ""),
                message_alert = msgs.get("alert", ""),
            )

    def _add_edges(self, operations: list):
        """Ajoute les arêtes PRECEDE, NECESSITE, UTILISE_DANS."""
        op_ids = sorted([op["operation_id"] for op in operations])

        # PRECEDE : étape i → étape i+1
        for i in range(len(op_ids) - 1):
            self.G.add_edge(
                f"step_{op_ids[i]}",
                f"step_{op_ids[i+1]}",
                relation = "PRECEDE"
            )

        # NECESSITE + UTILISE_DANS
        for op in operations:
            oid      = op["operation_id"]
            tool_ids = op.get("tool_ids", [])
            for tid in tool_ids:
                tool_node = f"tool_{tid}"
                if tool_node in self.G:
                    self.G.add_edge(
                        f"step_{oid}", tool_node,
                        relation = "NECESSITE"
                    )
                    self.G.add_edge(
                        tool_node, f"step_{oid}",
                        relation = "UTILISE_DANS"
                    )

    # ─────────────────────────────────────────────────────────
    # REQUÊTES SUR LE GRAPHE
    # ─────────────────────────────────────────────────────────

    def get_step(self, step_id: int) -> dict:
        """Retourne toutes les données d'une étape."""
        node = f"step_{step_id}"
        if node not in self.G:
            return {}
        data  = dict(self.G.nodes[node])
        tools = self.get_tools_for_step(step_id)
        return {**data, "tools": tools}

    def get_next_step(self, step_id: int) -> dict:
        """Retourne l'étape suivante via l'arête PRECEDE."""
        node = f"step_{step_id}"
        for succ in self.G.successors(node):
            if self.G.nodes[succ].get("type") == "step":
                edge = self.G[node][succ]
                if edge.get("relation") == "PRECEDE":
                    return dict(self.G.nodes[succ])
        return {}

    def get_tools_for_step(self, step_id: int) -> list:
        """Retourne les outils nécessaires à une étape."""
        node = f"step_{step_id}"
        return [
            dict(self.G.nodes[n])
            for n in self.G.successors(node)
            if self.G.nodes[n].get("type") == "tool"
               and self.G[node][n].get("relation") == "NECESSITE"
        ]

    def get_steps_for_tool(self, tool_id: str) -> list:
        """Retourne les étapes qui utilisent un outil donné."""
        node = f"tool_{tool_id}"
        if node not in self.G:
            return []
        return [
            dict(self.G.nodes[n])
            for n in self.G.successors(node)
            if self.G.nodes[n].get("type") == "step"
        ]

    def get_all_steps(self) -> list:
        """Retourne toutes les étapes triées par id."""
        return sorted([
            dict(self.G.nodes[n])
            for n in self.G.nodes
            if self.G.nodes[n].get("type") == "step"
        ], key=lambda x: x["id"])

    def get_all_tools(self) -> list:
        """Retourne tous les outils."""
        return [
            dict(self.G.nodes[n])
            for n in self.G.nodes
            if self.G.nodes[n].get("type") == "tool"
        ]

    def is_valid_step(self, step_id: int) -> bool:
        return f"step_{step_id}" in self.G

    def is_last_step(self, step_id: int) -> bool:
        return not bool(self.get_next_step(step_id))

    def get_graph(self) -> nx.DiGraph:
        return self.G

    def print_summary(self):
        """Affiche un résumé du graphe."""
        print("\n" + "="*50)
        print("  GRAPHE DE CONNAISSANCE GUIDEME")
        print("="*50)
        print(f"Nœuds  : {self.G.number_of_nodes()}")
        print(f"Arêtes : {self.G.number_of_edges()}")

        print("\nÉtapes :")
        for step in self.get_all_steps():
            tools = self.get_tools_for_step(step["id"])
            tnames = [t["name"] for t in tools]
            next_s = self.get_next_step(step["id"])
            next_n = next_s.get("name", "FIN") if next_s else "FIN"
            print(f"  step_{step['id']} : {step['name']}")
            print(f"    → outils   : {tnames}")
            print(f"    → suivante : {next_n}")

        print("\nRelations par type :")
        relations = {}
        for u, v, d in self.G.edges(data=True):
            rel = d.get("relation", "?")
            relations[rel] = relations.get(rel, 0) + 1
        for rel, count in relations.items():
            print(f"  {rel:20} : {count} arêtes")


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gk = GraphKnowledge()
    gk.print_summary()

    print("\n--- TEST get_step(3) ---")
    step3 = gk.get_step(3)
    print(f"Nom    : {step3['name']}")
    print(f"Phase  : {step3['phase']}")
    print(f"Outils : {[t['name'] for t in step3['tools']]}")
    print(f"Action : {step3['message_action']}")

    print("\n--- TEST get_next_step(3) ---")
    next4 = gk.get_next_step(3)
    print(f"Étape suivante : {next4.get('name','?')}")

    print("\n--- TEST get_steps_for_tool('T01') ---")
    steps = gk.get_steps_for_tool("T01")
    print(f"Stéthoscope utilisé dans : {[s['name'] for s in steps]}")