"""
visualize_agent_blanc.py
Exactement la meme image mais fond blanc et texte bleu marine
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_facecolor("#FFFFFF")
fig.patch.set_facecolor("#FFFFFF")
ax.set_xlim(-1, 19)
ax.set_ylim(-1, 13)
ax.axis("off")

def draw_box(ax, x, y, w, h, color, title, subtitle="", border="#333333"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border,
        linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x + w/2, y + h*0.65, title,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white", zorder=4)
        ax.text(x + w/2, y + h*0.25, subtitle,
                ha="center", va="center",
                fontsize=7.5, color="#CBD5E1",
                zorder=4, style="italic")
    else:
        ax.text(x + w/2, y + h/2, title,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white", zorder=4,
                multialignment="center")

def draw_dashed_box(ax, x, y, w, h, color, title, lines):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor="#F1F5F9", edgecolor=color,
        linewidth=1.8, linestyle="--", zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, title,
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color=color, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.7 - i*0.38, line,
                ha="center", va="top",
                fontsize=8, color="#0F172A", zorder=3)

def arrow(ax, x1, y1, x2, y2, color="#374151", label="",
          style="arc3,rad=0", lw=1.5, ls="solid"):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=lw, connectionstyle=style,
            mutation_scale=16, linestyle=ls), zorder=2)
    if label:
        mx = (x1+x2)/2
        my = (y1+y2)/2
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor=color,
                          linewidth=0.8, alpha=0.95), zorder=5)

# ─── BLOCS RESSOURCES PARTAGÉES (HAUT) ──────────────────────

draw_dashed_box(ax, 0.2, 9.5, 4.2, 3.0, "#EA580C",
    "graph_knowledge.py",
    ["Graphe NetworkX DiGraph",
     "20 nœuds (8 step + 12 tool)",
     "33 arêtes",
     "PRECEDE · NECESSITE",
     "UTILISE_DANS",
     "get_step() · get_next_step()",
     "get_tools_for_step()"])

draw_dashed_box(ax, 5.2, 9.5, 4.2, 3.0, "#0EA5E9",
    "GuideMEState (TypedDict)",
    ["message · session_id",
     "intent · step_id",
     "step_name · step_tools",
     "step_action · step_alert",
     "prompt · response",
     "finished · score · total"])

draw_dashed_box(ax, 10.2, 9.5, 4.2, 3.0, "#7C3AED",
    "app.py — Flask",
    ["Orchestrateur principal",
     "Routes HTTP",
     "YOLOv8n-cls",
     "Groq Whisper ASR",
     "Hortense TTS",
     "init_models()"])

# ─── TITRE SECTION NOEUDS ────────────────────────────────────
ax.text(9, 9.1, "── 6 Nœuds LangGraph ──",
        ha="center", va="center",
        fontsize=11, color="#64748B", fontweight="bold")

# ─── 6 NOEUDS LANGGRAPH (BAS) ────────────────────────────────
NODE_W = 2.6
NODE_H = 1.0

nodes_info = [
    (0.2,  7.5, "#DC2626", "Nœud 1",   "detect_intent",
     "Mots-clés → intention\nnext/tool/action/why\nerror/restart"),
    (0.2,  5.8, "#D97706", "Nœud 2",   "manage_state",
     "Navigation PRECEDE\nstep_i → step_i+1\nMise à jour GuideMEState"),
    (0.2,  4.1, "#059669", "Nœud 3",   "build_context",
     "get_step() NetworkX\nContexte médical\nOutils + action + alerte"),
    (0.2,  2.4, "#1D4ED8", "Nœud 4",   "build_prompt",
     "Construit le prompt LLM\nSelon intention + contexte\nMax 2 phrases · FR"),
    (0.2,  0.7, "#EA580C", "Nœud 5",   "call_llm",
     "Modèle de langage\nRéponse pédagogique\nEn temps réel"),
    (5.2,  0.7, "#6D28D9", "Nœud 6",   "log_interaction",
     "Sauvegarde JSON horodaté\nMessage + intent + réponse\n+ étape + timestamp"),
]

for (x, y, color, num, name, desc) in nodes_info:
    draw_box(ax, x, y, NODE_W, NODE_H, color, f"{num}\n{name}")
    ax.text(x + NODE_W + 0.2, y + NODE_H/2, desc,
            ha="left", va="center",
            fontsize=7.5, color="#0F172A",
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="#F1F5F9",
                      edgecolor=color,
                      linewidth=0.8, alpha=0.95), zorder=3)

# ─── START / END ─────────────────────────────────────────────
start_circ = plt.Circle((1.5, 9.0), 0.28, color="#1E293B",
                          ec="#555", lw=1.5, zorder=3)
ax.add_patch(start_circ)
ax.text(1.5, 9.0, "START", ha="center", va="center",
        fontsize=7, color="white", fontweight="bold", zorder=4)

end_circ = plt.Circle((6.7, 0.7+0.5), 0.28, color="#1E293B",
                        ec="#555", lw=1.5, zorder=3)
ax.add_patch(end_circ)
ax.text(6.7, 1.2, "END", ha="center", va="center",
        fontsize=7, color="white", fontweight="bold", zorder=4)

# ─── FLÈCHES ENTRE NOEUDS ────────────────────────────────────
cx = 1.5

arrow(ax, cx, 8.72, cx, 8.5,   "#374151")
arrow(ax, cx, 7.5,  cx, 6.8,   "#D97706", "next/restart")
arrow(ax, cx+0.3, 7.5, cx+0.3, 4.1, "#059669", "autre", "arc3,rad=0.4")
arrow(ax, cx, 5.8,  cx, 5.1,   "#374151", "continue")
arrow(ax, cx-0.2, 5.8, 6.7, 1.5, "#EF4444", "finished", "arc3,rad=-0.3")
arrow(ax, cx, 4.1,  cx, 3.4,   "#374151")
arrow(ax, cx, 2.4,  cx, 1.7,   "#374151")
arrow(ax, cx+NODE_W/2, 1.2, 5.2+NODE_W/2, 1.2, "#374151")
arrow(ax, 6.7, 0.7, 6.7, 1.0-0.3, "#374151")

# ─── FLÈCHES RESSOURCES → NOEUDS ─────────────────────────────
arrow(ax, 2.3, 9.5, 1.3, 6.8,
      "#EA580C", "get_next_step()", "arc3,rad=0.2", 1.2, "dashed")
arrow(ax, 2.3, 9.5, 1.5, 5.1,
      "#EA580C", "get_step()", "arc3,rad=0.3", 1.2, "dashed")
arrow(ax, 7.3, 9.5, 2.0, 8.5,
      "#0EA5E9", "état partagé", "arc3,rad=-0.1", 1.2, "dashed")
arrow(ax, 12.3, 9.5, 1.8, 9.0,
      "#7C3AED", "invoque", "arc3,rad=0.2", 1.2, "dashed")

# ─── LÉGENDE ─────────────────────────────────────────────────
legend = [
    mpatches.Patch(color="#DC2626", label="Nœud 1 — detect_intent"),
    mpatches.Patch(color="#D97706", label="Nœud 2 — manage_state"),
    mpatches.Patch(color="#059669", label="Nœud 3 — build_context"),
    mpatches.Patch(color="#1D4ED8", label="Nœud 4 — build_prompt"),
    mpatches.Patch(color="#EA580C", label="Nœud 5 — call_llm"),
    mpatches.Patch(color="#6D28D9", label="Nœud 6 — log_interaction"),
    mpatches.Patch(color="#EA580C", label="graph_knowledge.py (ressource)",
                   linestyle="--", fill=False, edgecolor="#EA580C"),
    mpatches.Patch(color="#0EA5E9", label="GuideMEState (état partagé)",
                   linestyle="--", fill=False, edgecolor="#0EA5E9"),
    mpatches.Patch(color="#7C3AED", label="app.py (orchestrateur)",
                   linestyle="--", fill=False, edgecolor="#7C3AED"),
]
ax.legend(handles=legend, loc="upper right",
          facecolor="white", edgecolor="#CBD5E1",
          labelcolor="#0F172A", fontsize=8, framealpha=1)

ax.set_title(
    "Architecture de l'agent GuideME — LangGraph + graph_knowledge + GuideMEState",
    color="#0F172A", fontsize=13, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("graphe_agent_blanc.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Sauvegardé : graphe_agent_blanc.png")