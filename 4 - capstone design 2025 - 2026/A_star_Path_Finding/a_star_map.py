#!/usr/bin/env python3
"""
A* shortest path on the hand-drawn map (nodes A..H).

Usage:
  Run the script and type the start and goal node letters when prompted.
  Example start: A
  Example goal:  H

The script prints the shortest path (sequence of nodes) and the total distance in meters.

Notes on the map used:
 - Edge lengths are taken from the distances written on your image.
 - Two edges are treated as one-way because your sketch labels them "one way":
     B -> C   (C -> B not allowed)
     G -> D   (D -> G not allowed)
 - All other listed edges are treated as bidirectional.
"""

from heapq import heappush, heappop
import math
import matplotlib.pyplot as plt
try:
    import networkx as nx
except Exception:
    nx = None

# Edge list from the image (weights in meters).
# By default the edges below are treated as undirected (added both directions)
# except where explicitly listed as one_way_edges.
edges = [
    ("A", "B", 9),
    ("A", "E", 8),
    ("B", "C", 4),
    ("B", "D", 8),
    ("B", "H", 23),
    ("C", "D", 4),
    ("C", "E", 3),
    ("E", "F", 6),
    ("F", "G", 10),
    ("F", "H", 10),
    ("G", "H", 8),
    ("G", "D", 7),
]

# Edges in this set are one-way in the direction given in `edges` above
one_way_edges = {("B", "C"), ("G", "D")}

# Optional spatial coordinates for nodes (origin at bottom-left as provided).
# If present, the visualization will place nodes at these coordinates.
node_positions = {
    'A': (7.0, 0.0),
    'B': (3.0, 5.0),
    'C': (7.0, 5.0),
    'D': (7.0, 9.0),
    'E': (10.0, 5.0),
    'F': (13.0, 8.0),
    'G': (9.0, 14.0),
    'H': (16.0, 15.0),
}


def euclidean_heuristic(u, v):
    """Admissible straight-line heuristic (meters) using node_positions.

    Returns 0.0 if a coordinate for either node is missing so the search
    gracefully falls back to Dijkstra behavior.
    """
    pu = node_positions.get(u)
    pv = node_positions.get(v)
    if pu is None or pv is None:
        return 0.0
    return math.hypot(pu[0] - pv[0], pu[1] - pv[1])

# Build adjacency dictionary
def build_graph(edges, one_way_edges):
    graph = {}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w))
        if (u, v) not in one_way_edges:
            # add reverse edge for bidirectional
            graph.setdefault(v, []).append((u, w))
    # Ensure all nodes appear in graph even if isolated
    for u, v, w in edges:
        graph.setdefault(u, [])
        graph.setdefault(v, [])
    return graph

graph = build_graph(edges, one_way_edges)

# A* implementation with heuristic = 0 (i.e., Dijkstra).
# We keep heuristic as a function so it can be replaced by a spatial heuristic if coordinates are available.
def a_star(start, goal, graph, heuristic=None, explain=False):
    """
    A* search from start to goal on weighted graph represented by adjacency dict.
    graph[node] = [(neighbor, weight), ...]
    heuristic(node1, node2) returns estimated cost from node1 to node2.
    If heuristic is None, zero heuristic is used (equivalent to Dijkstra).
    Returns (path_list, total_cost) or (None, inf) if no path.
    """
    if heuristic is None:
        def heuristic(u, v): return 0.0

    # g_cost: best known cost from start to node
    # g_cost[n] stores the accumulated path cost from `start` to node n (sum of edge weights).
    # This is essential: A* uses the accumulated g(n) + heuristic h(n) to rank nodes.
    g_cost = {start: 0.0}
    # parent pointers for path reconstruction
    parent = {}

    # priority queue of heap tuples: (f_cost, h_cost, g_cost, node)
    # - f_cost = g_cost + h_cost is the primary key A* minimizes
    # - We include h_cost as the second field so ties on f break toward the node
    #   with smaller heuristic (closer in straight-line to the goal).
    # - g_cost is included for a stable ordering and to enable stale-entry checks.
    open_heap = []
    start_h = heuristic(start, goal)
    start_f = start_h  # at start, g=0 so f = h
    heappush(open_heap, (start_f, start_h, 0.0, start))

    closed = set()

    while open_heap:
        if explain:
            # Snapshot the open set and print only the current best entry per node (no stale entries).
            # This shows the authoritative f,g,h values used to choose the next expansion.
            try:
                # Collect nodes present in heap (there may be stale duplicates in the heap itself,
                # but here we report only the authoritative g_cost for each node).
                nodes_in_heap = {entry[3] for entry in open_heap}
                entries = []
                for node_e in nodes_in_heap:
                    best_g = g_cost.get(node_e, float("inf"))
                    # Skip nodes that haven't been discovered yet (no finite g)
                    if not math.isfinite(best_g):
                        continue
                    # Compute the heuristic and f for the current best g
                    h_e = heuristic(node_e, goal)
                    f_e = best_g + h_e
                    entries.append((f_e, h_e, best_g, node_e))
                # Sort and format the entries for readable output
                entries.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
                snap_str = ", ".join([f"{n}: f={f:.3f}, g={g:.3f}, h={h:.3f}" for f,h,g,n in entries])
                if snap_str == "":
                    print("Open set snapshot: (no discovered nodes in open set)")
                else:
                    print(f"Open set snapshot: {snap_str}")
            except Exception:
                
                # Be defensive in case heap entries vary; fall back to a simple repr
                print(f"Open set (raw): {open_heap}")

        # Pop the best candidate from the heap. This is the node A* will attempt to expand next.
        # Note: due to earlier improvements we may have multiple heap entries for the same node
        # (old/stale copies). We detect stale entries below and skip them.
        f, h, g, current = heappop(open_heap)

        # If we popped a node with stale g (higher than the recorded best g_cost), skip it.
        # Reason: we keep pushing improved entries instead of removing old ones from the heap
        # (removing arbitrary entries is expensive). When an old entry is popped we simply
        # ignore it because g_cost[current] holds the authoritative, lower value.
        if g_cost.get(current, float("inf")) < g:
            if explain:
                best = g_cost.get(current, float("inf"))
                print(f"Skipping stale entry for {current}: heap g={g:.3f} > best g={best:.3f}")
            continue

        if explain:
            # Show the node we're about to expand. At this point we consider all its outgoing edges.
            print(f"\nExpanding node {current} with g={g:.3f}, h={h:.3f}, f={f:.3f}")

        if current == goal:
            # reconstruct path
            path = []
            node = goal
            while node in parent:
                path.append(node)
                node = parent[node]
            path.append(start)
            path.reverse()
            if explain:
                print("\nGoal reached. Reconstructing path:")
                for i in range(len(path)-1):
                    a = path[i]; b = path[i+1]
                    # find weight
                    for nb, w in graph[a]:
                        if nb == b:
                            print(f"  {a} -> {b}: cost {w:.3f}")
                            break
                print(f"Total cost: {g_cost[goal]:.3f}")
            return path, g_cost[goal]

        # Mark the node as closed (finished). Closed nodes are not reconsidered later.
        # With a consistent heuristic this is safe: once closed, the node's g is final.
        closed.add(current)

        # Consider each neighbor: compute the tentative accumulated cost to reach it via `current`.
        for neighbor, weight in graph.get(current, []):
            # Skip neighbors already closed (we've finished them)
            if neighbor in closed:
                if explain:
                    print(f"  Neighbor {neighbor} already closed; skipping")
                continue

            # tentative_g = g(current) + weight(current->neighbor)
            tentative_g = g_cost[current] + weight
            # heuristic for neighbor to goal
            h = heuristic(neighbor, goal)
            f_cost = tentative_g + h

            if explain:
                # Explain the arithmetic: current g, the edge weight, tentative g, h, and resulting f
                print(
                    f"  Evaluate {current}->{neighbor}: "
                    f"g({current})={g_cost[current]:.3f}; "
                    f"weight={weight:.3f}; "
                    f"tentative g({neighbor})={g_cost[current]:.3f}+{weight:.3f}={tentative_g:.3f}; "
                    f"h({neighbor})=dist({neighbor},{goal})={h:.3f}; "
                    f"f({neighbor})={tentative_g:.3f}+{h:.3f}={f_cost:.3f}"
                )

            # If this path to neighbor is better than any previous one, record it and push to heap.
            # We do not remove old heap entries; instead we will skip them when they are popped
            # if they are stale (see the stale-entry check above).
            if tentative_g < g_cost.get(neighbor, float("inf")):
                if explain:
                    prev = g_cost.get(neighbor, float("inf"))
                    if prev == float("inf"):
                        print(f"    --> Improve {neighbor}: g({neighbor})={tentative_g:.3f} (was inf); f({neighbor})={f_cost:.3f}")
                    else:
                        print(f"    --> Improve {neighbor}: g({neighbor})={tentative_g:.3f} (was {prev:.3f}); f({neighbor})={f_cost:.3f}")
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current
                # push (f, h, g, node) so ties on f break by smaller heuristic h
                heappush(open_heap, (f_cost, h, tentative_g, neighbor))

    # no path found
    return None, float("inf")


def print_graph(graph):
    print("Graph adjacency (node: [(neighbor, meters), ...]):")
    for node in sorted(graph.keys()):
        print(f" {node}: {graph[node]}")


def visualize_graph(graph, path=None, out_fn='a_star_map.png'):
    """Create and save a visualization of the graph and optional path.

    - graph: adjacency dict {node: [(neighbor, weight), ...]}
    - path: list of node names in order (e.g., ['A','B','C']) to highlight
    - out_fn: output filename for saved PNG
    """
    if nx is None:
        print("Visualization skipped: networkx not available")
        return
    G = nx.Graph()
    # Add edges (undirected view for visualization)
    for u, nbrs in graph.items():
        for v, w in nbrs:
            if not G.has_edge(u, v):
                G.add_edge(u, v, weight=w)

    # Layout: use provided spatial coordinates if available, otherwise spring layout
    if 'node_positions' in globals():
        # filter positions to nodes present in G
        pos = {n: node_positions[n] for n in G.nodes() if n in node_positions}
    else:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    # draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='k')
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=14)

    # draw edges and annotate weights
    edge_widths = [max(0.8, 3.0 - (d['weight'] / 10.0)) for _,_,d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_widths)
    
    # highlight path if provided (draw BEFORE labels so labels appear on top)
    if path and len(path) >= 2:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4.0, edge_color='red')
    
    # draw edge labels last so they appear on top of everything
    edge_labels = {(u,v): f"{d['weight']:.1f}m" for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', 
                                   font_size=20, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.95))

    # compute total path length from graph weights
    if path and len(path) >= 2:
        total = 0.0
        for a, b in path_edges:
            w = G[a][b]['weight']
            total += w
        ax.text(0.02, 0.98, f"Path: {'->'.join(path)}  Length: {total:.2f} m",
                transform=ax.transAxes, fontsize=10, verticalalignment='top', color='red')

    plt.title('A* map visualization')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    print(f"Saved graph visualization to {out_fn}")
    try:
        plt.show()
    except Exception:
        # headless environment: saving the file is sufficient
        pass

def main():
    print("Nodes available:", " ".join(sorted(graph.keys())))
    print_graph(graph)
    start = input("Enter start node (e.g., A): ").strip().upper()
    if start not in graph:
        print("Unknown start node.")
        return
    goal = input("Enter goal node (e.g., H): ").strip().upper()
    if goal not in graph:
        print("Unknown goal node.")
        return

    path, cost = a_star(start, goal, graph, heuristic=euclidean_heuristic, explain=True)
    if path is None:
        print(f"No path found from {start} to {goal}.")
    else:
        print(f"Shortest path from {start} to {goal}: {' -> '.join(path)}")
        print(f"Total distance: {cost:.2f} meters")
        # also create a visualization PNG showing nodes, edges and the path
        try:
            visualize_graph(graph, path=path, out_fn='a_star_map.png')
        except Exception as e:
            print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()