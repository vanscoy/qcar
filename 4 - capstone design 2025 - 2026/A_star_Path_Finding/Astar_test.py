# coords_fit.py
# Fit 2D coordinates to the hand-drawn map edges by least-squares.
# Requires: numpy, scipy
# pip install numpy scipy

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import networkx as nx

# Edge list from your sketch (treat as undirected for distances)
edges = [
    ("A","B", 8.0),
    ("A","E",10.0),
    ("B","C", 7.0),
    ("B","D",14.0),
    ("B","H",30.0),
    ("C","D", 7.0),
    ("C","E", 4.0),
    ("E","F",10.0),
    ("F","G",15.0),
    ("F","H", 9.0),
    ("G","H", 7.0),
    ("G","D", 5.0),
]

# Node list (collect all unique nodes)
nodes = sorted({u for u,_,_ in edges} | {v for _,v,_ in edges})
node_index = {n:i for i,n in enumerate(nodes)}
print("Nodes:", nodes)

# Anchors: fix A at (10,0). We'll also fix B to remove rotation ambiguity.
# You can change B's coordinates if you'd like a different orientation.
anchors = {
    "A": (10.0, 0.0),
    # choose a plausible B anchor to set orientation (change if desired)
    "B": (2.0, 8.0),
}

# Build initial guess for coordinates (everything else near the sketch)
x0 = []
free_idx = []   # mapping free variable index -> node index
for n in nodes:
    if n in anchors:
        x0.extend([anchors[n][0], anchors[n][1]])
    else:
        # initial guess: place roughly where the sketch suggests
        # (you can tweak these to aid convergence)
        guess = {
            "C": (9.0, 8.0),
            "D": (9.0, 15.0),
            "E": (13.0, 8.0),
            "F": (20.0, 8.0),
            "G": (6.5, 15.0),
            "H": (12.0, 15.0),
            "A": (10.0, 0.0),
            "B": (2.0, 8.0),
        }.get(n, (10.0, 5.0))
        x0.extend([guess[0], guess[1]])
        free_idx.append(node_index[n])

x0 = np.array(x0)

# Construct mask array that marks which variables are free vs anchored
is_free = []
for n in nodes:
    is_free.extend([n not in anchors, n not in anchors])
is_free = np.array(is_free, dtype=bool)

# residual function: for each edge, residual = ||pi - pj|| - L
def residuals(free_vars):
    # rebuild full coordinate vector
    full = x0.copy()
    full[is_free] = free_vars
    # coords array shape (N_nodes, 2)
    coords = full.reshape(-1,2)
    res = []
    for u,v,L in edges:
        iu = node_index[u]
        iv = node_index[v]
        pu = coords[iu]
        pv = coords[iv]
        dij = np.linalg.norm(pu - pv)
        res.append(dij - L)
    return np.array(res)

# initial free var vector
free0 = x0[is_free]

# run least squares
res = least_squares(residuals, free0, verbose=2, xtol=1e-8, ftol=1e-8, max_nfev=5000)

# reconstruct full coordinate vector
full = x0.copy()
full[is_free] = res.x
coords = full.reshape(-1,2)
coords_dict = {n: tuple(coords[node_index[n]]) for n in nodes}

print("\nFitted coordinates (meters):")
for n in nodes:
    x,y = coords_dict[n]
    print(f" {n:>1s}: ({x:7.3f}, {y:7.3f})")

# print residuals and actual distances
print("\nEdge comparisons (desired vs achieved, residual):")
for u,v,L in edges:
    pu = coords[node_index[u]]
    pv = coords[node_index[v]]
    dij = np.linalg.norm(pu - pv)
    print(f" {u}-{v}: target {L:5.2f} m, actual {dij:6.3f} m, residual {dij-L:6.3f} m")


# ----------------------- Visualization -----------------------
# Build a NetworkX graph using the fitted coordinates and visualize it.
G = nx.Graph()
for n in nodes:
    G.add_node(n, pos=coords_dict[n])
for u, v, L in edges:
    G.add_edge(u, v, weight=L)

# Example shortest-path pairs to highlight (customize as needed)
PATH_PAIRS = [("A", "G")]

# Compute positions for drawing
pos = {n: coords_dict[n] for n in nodes}

# Draw base graph
plt.figure(figsize=(8, 6))
ax = plt.gca()
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgreen", edgecolors='k')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
# draw edges; scale widths inversely with length for visibility
edge_widths = [max(0.8, 4.0 - (d['weight'] / 8.0)) for _,_,d in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.9)

# Annotate edges with their target lengths
edge_labels = {(u,v): f"{w['weight']:.1f}m" for u,v,w in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')

# Highlight shortest paths
for idx,(a,b) in enumerate(PATH_PAIRS):
    try:
        path = nx.shortest_path(G, a, b, weight='weight')
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4.0, edge_color='red')
        # annotate the path length on the figure
        path_len = nx.shortest_path_length(G, a, b, weight='weight')
        ax.text(0.02, 0.98 - 0.03 * idx, f"Shortest {a}->{b}: {path_len:.2f} m",
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='red')
    except Exception as e:
        print(f"Could not compute shortest path {a}->{b}: {e}")

plt.title('A* test: fitted map with shortest path(s)')
plt.axis('equal')
plt.tight_layout()
out_fn = 'astar_graph.png'
plt.savefig(out_fn, dpi=150)
print(f"Saved visualization to {out_fn}")
try:
    plt.show()
except Exception:
    pass
