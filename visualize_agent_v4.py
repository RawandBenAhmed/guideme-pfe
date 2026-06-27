"""
visualize_agent_v4.py — Graphe LangGraph agent GuideME
Fond blanc, grands noeuds, grande écriture bleu marine
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(26, 18))
ax.set_facecolor("#FFFFFF")
fig.patch.set_facecolor("#FFFFFF")
ax.set_xlim(-1, 22)
ax.set_ylim(-1, 14)
ax.axis("off")

def draw_box(ax, x, y, w, h, color, title, border="#333333"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18",
        facecolor=color, edgecolor=border,
        linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    lines = title.split("\n")
    for i, line in enumerate(lines):
        offset = h/2 + (len(lines)-1)*0.22 - i*0.44
        ax.text(x + w/2, y + offset, line,
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white", zorder=4)

def draw_dashed_box(ax, x, y, w, h, color, title, lines):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18",
        facecolor="#F8FAFC", edgecolor=color,
        linewidth=2.5, linestyle="--", zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.3, title,
            ha="center", va="top",
            fontsize=12, fontweight="bold",
            color=color, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.72 - i*0.38, line,
                ha="center", va="top",
                fontsize=10, color="#0F172A", zorder=3)

def arrow(ax, x1, y1, x2, y2, color="#374151", label="",
          style="arc3,rad=0", lw=2.2, ls="solid"):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=lw, connectionstyle=style,
            mutation_scale=24, linestyle=ls), zorder=2)
    if label:
        mx = (x1+x2)/2
        my = (y1+y2)/2
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=10, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=color,
                          linewidth=1.2, alpha=0.98), zorder=5)

# ── RESSOURCES EN HAUT ───────────────────────────────────────
draw_dashed_box(ax, 0.2, 10.5, 5.5, 3.0, "#EA580C",
    "graph_knowledge.py",
    ["Graphe NetworkX DiGraph",
     "20 noeuds (8 step + 12 tool)",
     "33 arêtes",
     "PRECEDE · NECESSITE · UTILISE_DANS",
     "get_step() · get_next_step()",
     "get_tools_for_step()"])

draw_dashed_box(ax, 6.5, 10.5, 5.5, 3.0, "#0EA5E9",
    "GuideMEState (TypedDict)",
    ["message · session_id",
     "intent · step_id",
     "step_name · step_tools",
     "step_action · step_alert",
     "prompt · response",
     "finished · score · total"])

draw_dashed_box(ax, 12.8, 10.5, 5.5, 3.0, "#7C3AED",
    "app.py — Flask",
    ["Orchestrateur principal",
     "Routes HTTP",
     "YOLOv8n-cls (classification VR)",
     "Groq Whisper (ASR)",
     "Microsoft Hortense (TTS)",
     "init_models()"])

# ── 6 NOEUDS ─────────────────────────────────────────────────
NW, NH = 4.0, 1.3
CX = 0.5

nodes_data = [
    (0.0, 8.5,  "#DC2626", "detect_intent\n(Nœud 1)"),
    (0.0, 6.7,  "#D97706", "manage_state\n(Nœud 2)"),
    (0.0, 4.9,  "#059669", "build_context\n(Nœud 3)"),
    (0.0, 3.1,  "#1D4ED8", "build_prompt\n(Nœud 4)"),
    (0.0, 1.3,  "#EA580C", "call_llm\n(Nœud 5)"),
    (6.5, 1.3,  "#6D28D9", "log_interaction\n(Nœud 6)"),
]

descs = [
    "Mots-clés → intention\nnext / tool / action\nwhy / error / restart",
    "Navigation PRECEDE\nstep_i → step_i+1\nMise à jour GuideMEState",
    "get_step() NetworkX\nContexte médical complet\nOutils + action + alerte",
    "Construit le prompt LLM\nSelon intention + contexte\nMax 2 phrases · Français",
    "Modèle de langage\nRéponse pédagogique\nEn temps réel",
    "Sauvegarde JSON horodaté\nMessage + intention + réponse\n+ étape + timestamp",
]

for (x, y, color, title), desc in zip(nodes_data, descs):
    draw_box(ax, x, y, NW, NH, color, title)
    ax.text(x + NW + 0.3, y + NH/2, desc,
            ha="left", va="center",
            fontsize=10, color="#0F172A",
            multialignment="left",
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#F1F5F9",
                      edgecolor=color,
                      linewidth=1.5, alpha=0.98), zorder=3)

# ── START / END ───────────────────────────────────────────────
for (cx, cy, txt) in [(CX+NW/2, 10.0, "START"), (8.5, 1.95, "END")]:
    c = plt.Circle((cx, cy), 0.42, color="#1E293B",
                   ec="#555", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=10, color="white",
            fontweight="bold", zorder=4)

# ── FLÈCHES ───────────────────────────────────────────────────
MX = CX + NW/2

arrow(ax, MX, 9.58, MX, 9.8,   "#374151")
arrow(ax, MX, 8.5,  MX, 7.75,  "#D97706", "next/restart")
arrow(ax, MX+0.5, 8.5, MX+0.5, 4.9, "#059669", "autre", "arc3,rad=0.45")
arrow(ax, MX, 6.7,  MX, 6.2,   "#374151", "continue")
arrow(ax, MX-0.3, 6.7, 8.5, 2.4, "#EF4444", "finished", "arc3,rad=-0.35")
arrow(ax, MX, 4.9,  MX, 4.4,   "#374151")
arrow(ax, MX, 3.1,  MX, 2.6,   "#374151")
arrow(ax, MX+NW/2, 1.95, 6.5+NW/2, 1.95, "#374151")
arrow(ax, 8.5, 1.3, 8.5, 1.53, "#374151")

# Flèches ressources
arrow(ax, 3.0, 10.5, 1.5, 8.0,  "#EA580C", "", "arc3,rad=0.2", 1.5, "dashed")
arrow(ax, 3.0, 10.5, 1.8, 6.2,  "#EA580C", "", "arc3,rad=0.3", 1.5, "dashed")
arrow(ax, 9.2, 10.5, 3.5, 9.8,  "#0EA5E9", "état partagé", "arc3,rad=-0.1", 1.5, "dashed")
arrow(ax, 15.5, 10.5, 4.0, 9.8, "#7C3AED", "invoque", "arc3,rad=0.25", 1.5, "dashed")

# ── LÉGENDE ───────────────────────────────────────────────────
legend = [
    mpatches.Patch(color="#DC2626", label="Nœud 1 — detect_intent"),
    mpatches.Patch(color="#D97706", label="Nœud 2 — manage_state"),
    mpatches.Patch(color="#059669", label="Nœud 3 — build_context"),
    mpatches.Patch(color="#1D4ED8", label="Nœud 4 — build_prompt"),
    mpatches.Patch(color="#EA580C", label="Nœud 5 — call_llm"),
    mpatches.Patch(color="#6D28D9", label="Nœud 6 — log_interaction"),
]
ax.legend(handles=legend, loc="lower right",
          facecolor="white", edgecolor="#CBD5E1",
          labelcolor="#0F172A", fontsize=12,
          framealpha=1, prop={"size": 12})

ax.set_title(
    "Architecture de l'agent GuideME — LangGraph + graph_knowledge + GuideMEState",
    color="#0F172A", fontsize=18, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("graphe_agent_v4.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Sauvegardé : graphe_agent_v4.png")