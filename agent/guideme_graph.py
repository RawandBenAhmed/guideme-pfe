"""
agent/guideme_graph.py
Agent GuideME construit avec LangGraph.
"""

import uuid
from typing import TypedDict, Literal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from agent.graph_knowledge import GraphKnowledge
from agent.intent_detector  import detect_intent
from agent.llm_connector    import LLMConnector
from agent.logger           import Logger


class GuideMEState(TypedDict):
    message:        str
    session_id:     str
    intent:         str
    step_id:        int
    step_name:      str
    step_phase:     str
    step_tools:     list
    step_action:    str
    step_alert:     str
    step_intro:     str
    next_step_id:   int
    next_step_name: str
    prompt:         str
    response:       str
    finished:       bool
    score:          int
    total:          int
    direct_response: str  # réponse directe sans LLM


_gk  = None
_llm = None


def init_agent():
    global _gk, _llm
    _gk  = GraphKnowledge()
    _llm = LLMConnector()
    print("[AGENT] GraphKnowledge + LLMConnector initialisés")


def node_detect_intent(state: GuideMEState) -> GuideMEState:
    intent = detect_intent(state["message"])
    print(f"[NODE detect_intent] '{state['message']}' → {intent}")
    return {**state, "intent": intent, "direct_response": ""}


def node_manage_state(state: GuideMEState) -> GuideMEState:
    intent  = state["intent"]
    step_id = state["step_id"]

    if intent == "restart":
        return {**state, "step_id": 1, "score": 0, "total": 0}

    if intent == "next":
        next_data = _gk.get_next_step(step_id)
        if next_data:
            new_id = next_data["id"]
            return {**state, "step_id": new_id,
                    "next_step_id": new_id,
                    "next_step_name": next_data["name"]}
        else:
            return {**state, "finished": True}

    return state


def node_build_context(state: GuideMEState) -> GuideMEState:
    step_data = _gk.get_step(state["step_id"])
    tools     = step_data.get("tools", [])
    print(f"[NODE build_context] step_{state['step_id']} | {step_data.get('name','')} | outils : {[t['name'] for t in tools]}")
    return {
        **state,
        "step_name":   step_data.get("name", ""),
        "step_phase":  step_data.get("phase", ""),
        "step_tools":  tools,
        "step_action": step_data.get("message_action", ""),
        "step_alert":  step_data.get("message_alert", ""),
        "step_intro":  step_data.get("message_intro", ""),
    }


def node_build_prompt(state: GuideMEState) -> GuideMEState:
    # tools_str construit depuis step_tools déjà chargé par node_build_context
    tools_str = ', '.join([t["name"] for t in state["step_tools"]])
    intent    = state["intent"]
    step_id   = state["step_id"]
    step_name = state["step_name"]
    action    = state["step_action"]
    alert     = state["step_alert"]
    message   = state["message"]

    base = (
        "IMPORTANT : Reponds UNIQUEMENT en francais. "
        "Tu es GuideME, assistant pedagogique en drainage thoracique. "
        "Reponds directement a la question. Maximum 2 phrases courtes. "
        "N invente AUCUNE information medicale."
    )

    direct = ""  # réponse directe sans LLM
    prompt = ""

    if intent == "tool":
        # ✅ FIX : réponse directe — le LLM ne peut pas halluciner les outils
        direct = "Pour cette étape, tu as besoin de : " + tools_str + "."

    elif intent == "action":
        msg_lower = message.lower()
        if any(k in msg_lower for k in [
            "quelle etape", "quelle étape", "dans quelle",
            "je suis dans", "suis dans", "quel etape", "quel étape",
            "etape suis", "étape suis"
        ]):
            direct = "Tu es dans l'étape " + str(step_id) + " : " + step_name + "."
        else:
            prompt = (
                base + "\n\n"
                "Etape " + str(step_id) + " : " + step_name + ".\n"
                "Action a effectuer : " + action + ".\n"
                "Reponds UNIQUEMENT avec l action a effectuer. Une seule phrase courte."
            )

    elif intent == "why":
        prompt = (
            base + "\n\n"
            "Etape " + str(step_id) + " : " + step_name + ".\n"
            "Action : " + action + ". Alerte : " + alert + ".\n"
            "Question : " + message + "\n"
            "Explique le raisonnement medical en 2 phrases maximum."
        )

    elif intent == "error":
        prompt = (
            base + "\n\n"
            "Etape " + str(step_id) + " : " + step_name + ".\n"
            "Outils : " + tools_str + ". Action : " + action + ".\n"
            "Corrige l etudiant en 2 phrases maximum."
        )

    elif intent == "next":
        # ✅ FIX : utilise tools_str depuis state["step_tools"] — déjà mis à jour
        # par node_build_context avec le bon step_id. Suppression du double fetch
        # get_tools_for_step() qui causait une désynchronisation.
        prompt = (
            base + "\n\n"
            "L etudiant passe a l etape " + str(step_id) + " : " + step_name + ".\n"
            "Outils : " + tools_str + ". Action : " + action + ".\n"
            "Dis-lui UNIQUEMENT : Tu passes a l etape " + str(step_id) + " : " + step_name + ". "
            "Prends : " + tools_str + ". N invente rien. Maximum 2 phrases."
        )

    elif intent == "restart":
        direct = "On recommence depuis le début. Étape 1 : Diagnostique."

    else:
        prompt = (
            base + "\n\n"
            "Etape " + str(step_id) + " : " + step_name + ".\n"
            "Question : " + message + "\n"
            "Si hors sujet, reponds : Je suis specialise uniquement dans le drainage thoracique. "
            "Sinon reponds en 2 phrases."
        )

    print(f"[NODE build_prompt] intent={intent} | direct={bool(direct)}")
    return {**state, "prompt": prompt, "direct_response": direct}


def node_call_llm(state: GuideMEState) -> GuideMEState:
    # Si réponse directe → pas de LLM
    if state.get("direct_response"):
        response = state["direct_response"]
        print(f"[NODE call_llm] DIRECT → {response}")
        return {**state, "response": response}

    response = _llm.call(state["prompt"])
    print(f"[NODE call_llm] → {response[:70]}...")
    return {**state, "response": response}


def node_log_interaction(state: GuideMEState) -> GuideMEState:
    lg = Logger(state["session_id"])
    lg.log_user_message(state["message"], state["intent"])
    if state.get("response"):
        lg.log_agent_response(state["response"], state["step_id"])
    print(f"[NODE log] session={state['session_id']} | step={state['step_id']} | intent={state['intent']}")
    return state


def route_after_intent(state: GuideMEState) -> Literal["manage_state", "build_context"]:
    if state["intent"] in ("next", "restart"):
        return "manage_state"
    return "build_context"


def route_after_manage(state: GuideMEState) -> Literal["build_context", "__end__"]:
    if state.get("finished"):
        return "__end__"
    return "build_context"


def build_agent():
    workflow = StateGraph(GuideMEState)

    workflow.add_node("detect_intent",   node_detect_intent)
    workflow.add_node("manage_state",    node_manage_state)
    workflow.add_node("build_context",   node_build_context)
    workflow.add_node("build_prompt",    node_build_prompt)
    workflow.add_node("call_llm",        node_call_llm)
    workflow.add_node("log_interaction", node_log_interaction)

    workflow.set_entry_point("detect_intent")

    workflow.add_conditional_edges("detect_intent", route_after_intent,
        {"manage_state": "manage_state", "build_context": "build_context"})

    workflow.add_conditional_edges("manage_state", route_after_manage,
        {"build_context": "build_context", "__end__": END})

    workflow.add_edge("build_context",   "build_prompt")
    workflow.add_edge("build_prompt",    "call_llm")
    workflow.add_edge("call_llm",        "log_interaction")
    workflow.add_edge("log_interaction", END)

    agent = workflow.compile()
    print("[AGENT] Graphe LangGraph compilé ✅")
    return agent


def initial_state(message: str, step_id: int = 1,
                  session_id: str = None,
                  score: int = 0, total: int = 0) -> GuideMEState:
    return GuideMEState(
        message=message,
        session_id=session_id or str(uuid.uuid4())[:8],
        intent="", step_id=step_id,
        step_name="", step_phase="",
        step_tools=[], step_action="",
        step_alert="", step_intro="",
        next_step_id=0, next_step_name="",
        prompt="", response="",
        finished=False, score=score, total=total,
        direct_response="",
    )


if __name__ == "__main__":
    init_agent()
    agent = build_agent()
    sid = str(uuid.uuid4())[:8]
    tests = [
        ("je suis dans quelle etape", 1),
        ("donne moi le masque", 1),
        ("quoi faire", 1),
        ("pourquoi on utilise le ciseau", 6),
        ("j'ai fini", 3),
        ("quelle est la recette de la pizza", 2),
    ]
    for msg, step in tests:
        state  = initial_state(msg, step, sid)
        result = agent.invoke(state)
        print(f"\n[{result['intent']}] {msg}")
        print(f"-> {result['response']}")