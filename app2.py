"""
app2.py - GuideME v2
Agent LangGraph + Dataset orienté graphe (NetworkX)
Compare JSON classique vs Graphe pour la base de connaissance
"""

import json
import sys
from pathlib import Path
from typing import TypedDict, Literal
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
import config
from agent.llm_connector   import LLMConnector
from agent.intent_detector import detect_intent
from agent.logger          import Logger

from langgraph.graph import StateGraph, END

# ══════════════════════════════════════════════════════════════
# 1. DATASET JSON
# ══════════════════════════════════════════════════════════════

def load_json_dataset() -> dict:
    base = Path(__file__).parent / "json"

    with open(base / "operations.json", encoding="utf-8") as f:
        ops_raw = json.load(f)
    with open(base / "tools.json", encoding="utf-8") as f:
        tools_raw = json.load(f)
    with open(base / "messages.json", encoding="utf-8") as f:
        messages_raw = json.load(f)

    # Extraire listes depuis dicts si nécessaire
    operations = ops_raw.get("operations", ops_raw) \
                 if isinstance(ops_raw, dict) else ops_raw
    tools      = tools_raw.get("tools", tools_raw) \
                 if isinstance(tools_raw, dict) else tools_raw
    messages   = messages_raw.get("messages", messages_raw) \
                 if isinstance(messages_raw, dict) else messages_raw

    print(f"[JSON] {len(operations)} étapes chargées")
    print(f"[JSON] {len(tools)} outils chargés")
    return {"operations": operations, "tools": tools, "messages": messages}


# ══════════════════════════════════════════════════════════════
# 2. DATASET GRAPHE (NetworkX)
# ══════════════════════════════════════════════════════════════

def build_graph_dataset(json_data: dict) -> nx.DiGraph:
    """
    Construit un graphe orienté à partir des données JSON.

    Nœuds :
      - Étape (type='step')  : les 8 étapes chirurgicales
      - Outil (type='tool')  : les instruments médicaux

    Arêtes :
      - (Étape) --[PRECEDE]-----> (Étape suivante)
      - (Étape) --[NECESSITE]---> (Outil)
      - (Outil) --[UTILISE_DANS]-> (Étape)
    """
    G = nx.DiGraph()

    # ── Nœuds outils ──
    tools = json_data["tools"]
    for tool in tools:
        tid   = (tool.get("tool_id")
                 or tool.get("id")
                 or tool.get("tool_code", ""))
        tname = (tool.get("tool_name")
                 or tool.get("name", ""))
        G.add_node(
            f"tool_{tid}",
            type  = "tool",
            id    = tid,
            name  = tname,
            image = tool.get("tool_image", ""),
        )

    # ── Nœuds étapes ──
    operations = json_data["operations"]
    for op in operations:
        op_id   = op.get("operation_id") or op.get("id")
        op_name = op.get("operation_name") or op.get("name", "")
        G.add_node(
            f"step_{op_id}",
            type     = "step",
            id       = op_id,
            name     = op_name,
            action   = op.get("message_action", ""),
            alert    = op.get("message_alert", ""),
            duration = op.get("estimated_duration", 30),
            phase    = op.get("phase_label", ""),
        )

    # ── Arêtes PRECEDE (étape → étape suivante) ──
    op_ids = sorted([
        op.get("operation_id") or op.get("id")
        for op in operations
    ])
    for i in range(len(op_ids) - 1):
        G.add_edge(
            f"step_{op_ids[i]}",
            f"step_{op_ids[i+1]}",
            relation="PRECEDE"
        )

    # ── Arêtes NECESSITE + UTILISE_DANS ──
    for op in operations:
        op_id    = op.get("operation_id") or op.get("id")
        tool_ids = op.get("tool_ids", [])
        if isinstance(tool_ids, str):
            tool_ids = [t.strip() for t in tool_ids.split(",") if t.strip()]
        for tid in tool_ids:
            tool_node = f"tool_{tid}"
            if tool_node in G:
                G.add_edge(
                    f"step_{op_id}", tool_node,
                    relation="NECESSITE"
                )
                G.add_edge(
                    tool_node, f"step_{op_id}",
                    relation="UTILISE_DANS"
                )

    print(f"[GRAPH] {G.number_of_nodes()} nœuds")
    print(f"[GRAPH] {G.number_of_edges()} arêtes")
    return G


# ── Requêtes sur le graphe ──

def graph_get_step(G: nx.DiGraph, step_id: int) -> dict:
    node_id = f"step_{step_id}"
    if node_id not in G:
        return {}
    data  = dict(G.nodes[node_id])
    tools = [
        G.nodes[n]["name"]
        for n in G.successors(node_id)
        if G.nodes[n].get("type") == "tool"
    ]
    return {**data, "tools": tools}


def graph_get_next_step(G: nx.DiGraph, step_id: int) -> dict:
    node_id = f"step_{step_id}"
    for succ in G.successors(node_id):
        if G.nodes[succ].get("type") == "step":
            return dict(G.nodes[succ])
    return {}


def graph_get_tools_for_step(G: nx.DiGraph, step_id: int) -> list:
    node_id = f"step_{step_id}"
    return [
        dict(G.nodes[n])
        for n in G.successors(node_id)
        if G.nodes[n].get("type") == "tool"
    ]


def compare_json_vs_graph(json_data: dict, G: nx.DiGraph):
    print("\n" + "="*55)
    print("  COMPARAISON JSON vs GRAPHE")
    print("="*55)

    ops = json_data["operations"]

    # Requête 1 — étape 3
    print("\n[Requête 1] Données de l'étape 3")
    op_json = next((o for o in ops
                    if (o.get("operation_id") or o.get("id")) == 3), {})
    print(f"  JSON  : {op_json.get('operation_name','?')} "
          f"| outils IDs : {op_json.get('tool_ids','?')}")
    op_graph = graph_get_step(G, 3)
    print(f"  GRAPH : {op_graph.get('name','?')} "
          f"| outils : {op_graph.get('tools','?')}")

    # Requête 2 — étape suivante
    print("\n[Requête 2] Étape suivante après étape 3")
    next_json = next((o for o in ops
                      if (o.get("operation_id") or o.get("id")) == 4), {})
    print(f"  JSON  : {next_json.get('operation_name','?')}")
    next_graph = graph_get_next_step(G, 3)
    print(f"  GRAPH : {next_graph.get('name','?')}")

    # Requête 3 — outils étape 4
    print("\n[Requête 3] Outils nécessaires à l'étape 4")
    tools_graph = graph_get_tools_for_step(G, 4)
    print(f"  GRAPH : {[t['name'] for t in tools_graph]}")

    print("\n[INFO] Avantages du graphe :")
    print("  ✅ Navigation étape→suivante directe")
    print("  ✅ Requêtes relationnelles (outil→étapes)")
    print("  ✅ Extensible sans modifier le code")
    print("  ✅ Visualisable (networkx, gephi...)")


# ══════════════════════════════════════════════════════════════
# 3. ÉTAT LANGGRAPH
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    message:      str
    intent:       str
    step_id:      int
    step_name:    str
    step_tools:   list
    step_action:  str
    prompt:       str
    response:     str
    session_id:   str
    finished:     bool


# ══════════════════════════════════════════════════════════════
# 4. NŒUDS LANGGRAPH
# ══════════════════════════════════════════════════════════════

llm    = LLMConnector()
G_data = None  # graphe global


def node_detect_intent(state: AgentState) -> AgentState:
    intent = detect_intent(state["message"])
    print(f"[NODE detect_intent] '{state['message']}' → {intent}")
    return {**state, "intent": intent}


def node_build_context(state: AgentState) -> AgentState:
    step_data = graph_get_step(G_data, state["step_id"])
    tools     = step_data.get("tools", [])
    print(f"[NODE build_context] étape {state['step_id']} | outils : {tools}")
    return {
        **state,
        "step_name":   step_data.get("name", ""),
        "step_tools":  tools,
        "step_action": step_data.get("action", ""),
    }


def node_build_prompt(state: AgentState) -> AgentState:
    tools  = ', '.join(state["step_tools"]) if state["step_tools"] else "à définir"
    prompt = f"""Tu es GuideME, assistant pédagogique spécialisé
dans la procédure de drainage thoracique.
INSTRUCTIONS STRICTES — utilise UNIQUEMENT ces informations :
Etape actuelle : {state['step_id']} — {state['step_name']}
Outils requis  : {tools}
Action         : {state['step_action']}
Question       : {state['message']}
Réponds en maximum 2 phrases en français."""
    print(f"[NODE build_prompt] prompt prêt ({len(prompt)} chars)")
    return {**state, "prompt": prompt}


def node_call_llm(state: AgentState) -> AgentState:
    response = llm.call(state["prompt"])
    print(f"[NODE call_llm] → {response[:70]}...")
    return {**state, "response": response}


def node_next_step(state: AgentState) -> AgentState:
    next_data = graph_get_next_step(G_data, state["step_id"])
    if next_data:
        new_id = next_data.get("id", state["step_id"])
        print(f"[NODE next_step] {state['step_id']} → {new_id} : {next_data.get('name','')}")
        return {**state, "step_id": new_id}
    print("[NODE next_step] procédure terminée")
    return {**state, "finished": True}


def node_log(state: AgentState) -> AgentState:
    lg = Logger(state["session_id"])
    lg.log_user_message(state["message"], state["intent"])
    if state.get("response"):
        lg.log_agent_response(state["response"], state["step_id"])
    print(f"[NODE log] interaction sauvegardée")
    return state


# ══════════════════════════════════════════════════════════════
# 5. ROUTAGE CONDITIONNEL
# ══════════════════════════════════════════════════════════════

def route_intent(state: AgentState) -> Literal["build_context", "next_step"]:
    if state["intent"] == "next":
        return "next_step"
    return "build_context"


# ══════════════════════════════════════════════════════════════
# 6. CONSTRUCTION DU GRAPHE LANGGRAPH
# ══════════════════════════════════════════════════════════════

def build_langgraph():
    workflow = StateGraph(AgentState)

    workflow.add_node("detect_intent", node_detect_intent)
    workflow.add_node("build_context", node_build_context)
    workflow.add_node("build_prompt",  node_build_prompt)
    workflow.add_node("call_llm",      node_call_llm)
    workflow.add_node("next_step",     node_next_step)
    workflow.add_node("log",           node_log)

    workflow.set_entry_point("detect_intent")

    workflow.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "build_context": "build_context",
            "next_step":     "next_step",
        }
    )

    workflow.add_edge("build_context", "build_prompt")
    workflow.add_edge("build_prompt",  "call_llm")
    workflow.add_edge("call_llm",      "log")
    workflow.add_edge("log",           END)
    workflow.add_edge("next_step",     "build_context")

    agent = workflow.compile()
    print("[LANGGRAPH] Graphe compilé ✅")
    return agent


# ══════════════════════════════════════════════════════════════
# 7. TESTS
# ══════════════════════════════════════════════════════════════

def run_tests(agent, session_id: str):
    tests = [
        {"message": "que dois-je faire dans cette étape ?", "step_id": 1},
        {"message": "donne moi le stéthoscope",             "step_id": 1},
        {"message": "j'ai fini",                            "step_id": 2},
        {"message": "pourquoi aspirer avant d'injecter ?",  "step_id": 4},
    ]

    print("\n" + "="*55)
    print("  TEST AGENT LANGGRAPH")
    print("="*55)

    for i, t in enumerate(tests):
        print(f"\n{'─'*55}")
        print(f"Test {i+1} | Message : '{t['message']}' | Étape : {t['step_id']}")

        state = AgentState(
            message      = t["message"],
            intent       = "",
            step_id      = t["step_id"],
            step_name    = "",
            step_tools   = [],
            step_action  = "",
            prompt       = "",
            response     = "",
            session_id   = session_id,
            finished     = False,
        )

        result = agent.invoke(state)
        print(f"Intent   : {result['intent']}")
        print(f"Étape    : {result['step_id']} — {result['step_name']}")
        print(f"Réponse  : {result['response']}")
        if result.get("finished"):
            print("→ Procédure terminée ✅")


# ══════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uuid

    print("\n" + "="*55)
    print("  GuideME v2 — LangGraph + Graphe NetworkX")
    print("="*55)

    print("\n[1] Chargement dataset JSON...")
    json_data = load_json_dataset()

    print("\n[2] Construction graphe NetworkX...")
    G_data = build_graph_dataset(json_data)

    compare_json_vs_graph(json_data, G_data)

    print("\n[3] Construction agent LangGraph...")
    agent = build_langgraph()

    session_id = str(uuid.uuid4())[:8]
    run_tests(agent, session_id)

    print("\n✅ Terminé")