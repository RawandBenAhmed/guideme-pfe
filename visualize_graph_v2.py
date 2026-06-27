"""
visualize_graph_v3.py — Graphe dataset NetworkX
Fond blanc, très grands cercles, grande écriture bleu marine
"""

import math, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from agent.graph_knowledge import GraphKnowledge

gk = GraphKnowledge()
G  = gk.G

step_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "step"]
tool_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "tool"]

# Labels complets sans troncature
labels = {}
for n, d in G.nodes(data=True):
    name = d.get("name", n)
    # Coupe sur les mots longs
    words = name.split()
    if len(words) > 2:
        labels[n] = " ".join(words[:2]) + "\n" + " ".join(words[2:])
    else:
        labels[n] = name

pos = {}
# Steps : cercle intérieur plus grand
for i, node in enumerate(sorted(step_nodes)):
    angle = 2 * math.pi * i / len(step_nodes)
    pos[node] = (3.0 * math.cos(angle), 3.0 * math.sin(angle))

# Tools : cercle extérieur bien espacé
for i, node in enumerate(sorted(tool_nodes)):
    angle = 2 * math.pi * i / len(tool_nodes)
    pos[node] = (6.5 * math.cos(angle), 6.5 * math.sin(angle))

precede_edges   = [(u,v) for u,v,d in G.edges(data=True) if d.get("relation")=="PRECEDE"]
necessite_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get("relation")=="NECESSITE"]
utilise_edges   = [(u,v) for u,v,d in G.edges(data=True) if d.get("relation")=="UTILISE_DANS"]

fig, ax = plt.subplots(figsize=(26, 22))
ax.set_facecolor("#FFFFFF")
fig.patch.set_facecolor("#FFFFFF")

# Arêtes
nx.draw_networkx_edges(G, pos, edgelist=precede_edges,
    edge_color="#1D4ED8", arrows=True, arrowsize=30,
    width=3.0, connectionstyle="arc3,rad=0.12", ax=ax)

nx.draw_networkx_edges(G, pos, edgelist=necessite_edges,
    edge_color="#059669", arrows=True, arrowsize=22,
    width=1.8, alpha=0.85, connectionstyle="arc3,rad=0.06", ax=ax)

nx.draw_networkx_edges(G, pos, edgelist=utilise_edges,
    edge_color="#DC2626", arrows=True, arrowsize=18,
    width=1.4, alpha=0.65, connectionstyle="arc3,rad=-0.06", ax=ax)

# Noeuds step (bleu)
nx.draw_networkx_nodes(G, pos, nodelist=step_nodes,
    node_color="#2563EB", node_size=9000, alpha=0.95, ax=ax)

# Noeuds tool (orange)
nx.draw_networkx_nodes(G, pos, nodelist=tool_nodes,
    node_color="#EA580C", node_size=6500, alpha=0.95, ax=ax)

# Labels — bleu marine, grande taille
nx.draw_networkx_labels(G, pos, labels=labels,
    font_size=13, font_color="#0F172A",
    font_weight="bold", ax=ax)

# Légende
legend = [
    mpatches.Patch(color="#2563EB", label=f"Étape procédurale (step) — {len(step_nodes)} noeuds"),
    mpatches.Patch(color="#EA580C", label=f"Instrument médical (tool) — {len(tool_nodes)} noeuds"),
    mpatches.Patch(color="#1D4ED8", label=f"PRECEDE — {len(precede_edges)} arêtes"),
    mpatches.Patch(color="#059669", label=f"NECESSITE — {len(necessite_edges)} arêtes"),
    mpatches.Patch(color="#DC2626", label=f"UTILISE_DANS — {len(utilise_edges)} arêtes"),
]
ax.legend(handles=legend, loc="upper left",
          facecolor="white", edgecolor="#CBD5E1",
          labelcolor="#0F172A", fontsize=14,
          framealpha=1, prop={"size": 14})

ax.set_title("Graphe de connaissances GuideME — NetworkX DiGraph",
             color="#0F172A", fontsize=20,
             fontweight="bold", pad=20)
ax.axis("off")

plt.tight_layout()
plt.savefig("graphe_dataset_v3.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Sauvegardé : graphe_dataset_v3.png")