"""
visualize_graph.py
Visualise le graphe NetworkX du dataset GuideME
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from agent.graph_knowledge import GraphKnowledge

# ── Construit le graphe ──────────────────────────────────────
gk = GraphKnowledge()
G  = gk.G  # DiGraph NetworkX

print(f"Noeuds : {G.number_of_nodes()}")
print(f"Aretes : {G.number_of_edges()}")

# ── Sépare les types de noeuds ───────────────────────────────
step_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "step"]
tool_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "tool"]

print(f"Noeuds step : {step_nodes}")
print(f"Noeuds tool : {tool_nodes}")

# ── Labels lisibles ──────────────────────────────────────────
labels = {}
for n, d in G.nodes(data=True):
    name = d.get("name", n)
    # Raccourcit le nom si trop long
    if len(name) > 18:
        name = name[:16] + ".."
    labels[n] = name

# ── Layout : steps en cercle, tools autour ───────────────────
pos = {}

# Steps disposés en cercle au centre
import math
n_steps = len(step_nodes)
for i, node in enumerate(sorted(step_nodes)):
    angle = 2 * math.pi * i / n_steps
    pos[node] = (1.5 * math.cos(angle), 1.5 * math.sin(angle))

# Tools disposés en cercle extérieur
n_tools = len(tool_nodes)
for i, node in enumerate(sorted(tool_nodes)):
    angle = 2 * math.pi * i / n_tools
    pos[node] = (3.2 * math.cos(angle), 3.2 * math.sin(angle))

# ── Sépare les arêtes par type ───────────────────────────────
precede_edges     = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "PRECEDE"]
necessite_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "NECESSITE"]
utilise_edges     = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "UTILISE_DANS"]

print(f"PRECEDE     : {len(precede_edges)}")
print(f"NECESSITE   : {len(necessite_edges)}")
print(f"UTILISE_DANS: {len(utilise_edges)}")

# ── Dessin ───────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_facecolor("#0F172A")
fig.patch.set_facecolor("#0F172A")

# Noeuds step (bleu)
nx.draw_networkx_nodes(G, pos,
    nodelist=step_nodes,
    node_color="#2563EB",
    node_size=1800,
    alpha=0.95, ax=ax)

# Noeuds tool (orange)
nx.draw_networkx_nodes(G, pos,
    nodelist=tool_nodes,
    node_color="#EA580C",
    node_size=1200,
    alpha=0.95, ax=ax)

# Arêtes PRECEDE (blanc)
nx.draw_networkx_edges(G, pos,
    edgelist=precede_edges,
    edge_color="#FFFFFF",
    arrows=True,
    arrowsize=20,
    width=2.5,
    connectionstyle="arc3,rad=0.1",
    ax=ax)

# Arêtes NECESSITE (vert)
nx.draw_networkx_edges(G, pos,
    edgelist=necessite_edges,
    edge_color="#10B981",
    arrows=True,
    arrowsize=15,
    width=1.2,
    alpha=0.7,
    connectionstyle="arc3,rad=0.05",
    ax=ax)

# Arêtes UTILISE_DANS (rose)
nx.draw_networkx_edges(G, pos,
    edgelist=utilise_edges,
    edge_color="#F43F5E",
    arrows=True,
    arrowsize=12,
    width=1.0,
    alpha=0.5,
    connectionstyle="arc3,rad=-0.05",
    ax=ax)

# Labels
nx.draw_networkx_labels(G, pos,
    labels=labels,
    font_size=7.5,
    font_color="white",
    font_weight="bold",
    ax=ax)

# Légende
legend = [
    mpatches.Patch(color="#2563EB", label=f"Étape (step) — {len(step_nodes)} noeuds"),
    mpatches.Patch(color="#EA580C", label=f"Instrument (tool) — {len(tool_nodes)} noeuds"),
    mpatches.Patch(color="#FFFFFF", label=f"PRECEDE — {len(precede_edges)} arêtes"),
    mpatches.Patch(color="#10B981", label=f"NECESSITE — {len(necessite_edges)} arêtes"),
    mpatches.Patch(color="#F43F5E", label=f"UTILISE_DANS — {len(utilise_edges)} arêtes"),
]
ax.legend(handles=legend, loc="upper left",
          facecolor="#1E293B", edgecolor="#475569",
          labelcolor="white", fontsize=10)

ax.set_title("Graphe de connaissances GuideME — NetworkX DiGraph",
             color="white", fontsize=14, fontweight="bold", pad=15)
ax.axis("off")

plt.tight_layout()
output = Path(__file__).parent / "graphe_dataset.png"
plt.savefig(output, dpi=150, bbox_inches="tight",
            facecolor="#0F172A")
plt.close()

print(f"\nGraphe sauvegardé : {output}")
print("Ouvre le fichier graphe_dataset.png dans ton dossier projet.")