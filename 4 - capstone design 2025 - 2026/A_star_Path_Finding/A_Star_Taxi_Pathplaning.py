#!/usr/bin/env python3
"""
A* shortest path for Quanser QCar autonomous taxi competition map.

Usage:
    Run the script and type pickup and dropoff node letters when prompted.
    Route is always planned as: HUB -> pickup -> dropoff -> HUB.
    Example pickup:  A
    Example dropoff: K

The script prints the shortest path (sequence of nodes) and the total distance in meters.

Competition Map Notes:
 - Map features a central 4-way intersection with upper and lower loops
 - Taxi Hub Area is located on the right side
 - Distances are PLACEHOLDERS - update with actual measured values
 - All edges are bidirectional unless competition rules specify otherwise
 - Crosswalks are present at major intersections
"""

from heapq import heappush, heappop
import math
import matplotlib.pyplot as plt
try:
    import networkx as nx
except Exception:
    nx = None

# Competition map edge list (weights in meters - PLACEHOLDERS, update with measurements!)
# Map topology based on competition track:
#  - HUB: Taxi hub parking area at (18,6)
#  - A-L: Decision nodes throughout the track
#  - Edges are DIRECTED based on allowed travel directions

edges = [
    # HUB connections (HUB only goes to A)
    ("HUB", "A", 3.0),
    
    # A connections (A only goes to B, K, or J)
    ("A", "B", 3.0),
    ("A", "K", 3.0),
    ("A", "J", 3.0),
    
    # B connections (B goes to F, I, J, A)
    ("B", "F", 3.0),
    ("B", "I", 3.0),
    ("B", "J", 3.0),
    ("B", "A", 3.0),
    
    # C connections (C goes to D or G)
    ("C", "D", 3.0),
    ("C", "G", 3.0),
    
    # D connections (D goes to E or F)
    ("D", "E", 3.0),
    ("D", "F", 3.0),
    
    # E connections (E goes to C or K)
    ("E", "C", 3.0),
    ("E", "K", 3.0),
    
    # F connections (F goes to D, B, J, I)
    ("F", "D", 3.0),
    ("F", "B", 3.0),
    ("F", "J", 3.0),
    ("F", "I", 3.0),
    
    # G connections (G goes to L or H)
    ("G", "L", 3.0),
    ("G", "H", 3.0),
    
    # H connections (H goes to HUB, I, or G)
    ("H", "HUB", 3.0),
    ("H", "I", 3.0),
    ("H", "G", 3.0),
    
    # I connections (I goes to F, B, J, H)
    ("I", "F", 3.0),
    ("I", "B", 3.0),
    ("I", "J", 3.0),
    ("I", "H", 3.0),
    
    # J connections (J goes to A, B, F, I)
    ("J", "A", 3.0),
    ("J", "B", 3.0),
    ("J", "F", 3.0),
    ("J", "I", 3.0),
    
    # K connections (K goes to E, A, B)
    ("K", "E", 3.0),
    ("K", "A", 3.0),
    ("K", "B", 3.0),
    
    # L connections (L goes back to G)
    ("L", "G", 3.0),
]

# All edges are directed (one-way) based on the competition track layout
# Mark ALL edges as one-way so build_graph doesn't add reverse edges
one_way_edges = set((u, v) for u, v, w in edges)

# Turn restrictions: defines allowed exits based on entry direction
# Format: {node: {from_node: [allowed_exits]}}
# If a node is not in this dict, all outgoing edges from that node are allowed
turn_restrictions = {
    'A': {
        'HUB': ['B', 'K'],    # If entering A from HUB, can only exit to B or K
        'J': ['B', 'K'],      # If entering A from J, can only exit to B or K
        'K': ['J'],           # If entering A from K, can only exit to J
        'B': ['J'],           # If entering A from B, can only exit to J
    },
    'B': {
        'A': ['F', 'I', 'J'], # If entering B from A, can only exit to F, I, or J
        'K': ['F', 'I', 'J'], # If entering B from K, can only exit to F, I, or J
        'J': ['K', 'A'], # If entering B from J, can only exit to K or A
        'F': ['A', 'K'], # If entering B from F, can only exit to A or K
        'I': ['A', 'K'], # If entering B from I, can only exit to A or K
        # Add other entry points to B if needed

    },
    'C': {
        'E': ['D', 'G'],     # If entering C from E, can exit to D or G
        'D': ['G'],          # If entering C from D, can only exit to G
    },
    'D': {
        'C': ['E', 'F'],     # If entering D from C, can only exit to E or F
        'F': ['E'],          # If entering D from F, can only exit to E
    },
    'E': {
        'D': ['C', 'K'],     # If entering E from D, can only exit to C or K
        'K': ['C'],          # If entering E from K, can only exit to C
    },
    'F': {
        'D': ['B', 'J', 'I'], # If entering F from D, can only exit to B, J, or I
        'B': ['D'], # If entering F from B, can only exit to D
        'J': ['D'], # If entering F from J, can only exit to D
        'I': ['D'], # If entering F from I, can only exit to D
    },
    'G': {
        'L': ['H', 'C'],          # If entering G from L, can only exit to H or C
        'H': ['L', 'C'],          # If entering G from H, can only exit to L or C
        'C': ['L', 'H'],          # If entering G from C, can only exit to L or H
    },
    'H': {
        'G': ['HUB', 'I'],        # If entering H from G, can only exit to HUB or I
        'I': ['HUB', 'G'],             # If entering H from I, can only exit to HUB or G
    },
    'I': {
        'F': ['H'],     # If entering I from F, can only exit to H
        'B': ['H'],     # If entering I from B, can only exit to H
        'J': ['H'],     # If entering I from J, can only exit to H
        'H': ['F', 'B', 'J'], # If entering I from H, can only exit to F, B, or J
    },
    'J': {
        'A': ['B', 'F', 'I'], # If entering J from A, can only exit to B, F, or I
        'B': ['A'], # If entering J from B, can only exit to A
        'F': ['A'], # If entering J from F, can only exit to A
        'I': ['A'], # If entering J from I, can only exit to A
    },
    'K': {
        'E': ['A', 'B'], # If entering K from E, can only exit to A or B
        'A': ['E'], # If entering K from A, can only exit to E
        'B': ['E'], # If entering K from B, can only exit to E
    },
    'L': {
        'G': ['G'], # If entering L from G, can only exit to G
    },
    'HUB': {
        'H': ['A'], # If entering HUB from H, can only exit to A
    },
}

# Spatial coordinates for visualization (from competition track measurements)
# Coordinates are in the competition track reference frame
node_positions = {
    'HUB': (18.0, 6.0),
    'A': (16.0, 14.0),
    'B': (15.0, 10.0),
    'C': (5.0, 10.0),
    'D': (9.0, 10.0),
    'E': (9.0, 14.0),
    'F': (14.0, 8.0),
    'G': (9.0, 5.0),
    'H': (15.0, 5.0),
    'I': (15.0, 7.0),
    'J': (16.0, 8.0),
    'K': (11.0, 14.0),
    'L': (9.0, 7.0),
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

def get_allowed_neighbors(current, prev_node, graph, turn_restrictions):
    """Get neighbors from current node that are allowed given we came from prev_node.
    
    Args:
        current: current node
        prev_node: node we came from (None if starting node)
        graph: adjacency dict
        turn_restrictions: dict of turn restrictions
    
    Returns:
        list of (neighbor, weight) tuples that are allowed
    """
    # Get all possible neighbors from graph
    all_neighbors = graph.get(current, [])
    
    # If no turn restrictions for this node, return all neighbors
    if current not in turn_restrictions:
        return all_neighbors
    
    # If we don't have a previous node (starting position), allow all neighbors
    if prev_node is None:
        return all_neighbors
    
    # Get the restrictions for this specific entry direction
    restrictions = turn_restrictions[current]
    
    # If the previous node isn't in the restrictions, allow all neighbors
    if prev_node not in restrictions:
        return all_neighbors
    
    # Filter neighbors based on allowed exits
    allowed_exits = restrictions[prev_node]
    return [(neighbor, weight) for neighbor, weight in all_neighbors 
            if neighbor in allowed_exits]

def a_star_with_entry(start, goal, entry_prev_node, graph, turn_restrictions=None, heuristic=None, explain=False):
    """
    A* search that respects the entry direction at the start node.
    
    Args:
        start: starting node
        goal: goal node
        entry_prev_node: the node we came from to reach 'start' (None if this is the very first segment)
        graph, turn_restrictions, heuristic, explain: same as a_star
    
    Returns:
        (path_list, total_cost) or (None, inf) if no path
    """
    if heuristic is None:
        def heuristic(u, v): return 0.0
    
    if turn_restrictions is None:
        turn_restrictions = {}

    # g_cost: best known cost from start to state (node, prev_node)
    g_cost = {(start, entry_prev_node): 0.0}
    parent = {}

    # priority queue: (f_cost, h_cost, g_cost, node, prev_node)
    open_heap = []
    start_h = heuristic(start, goal)
    start_f = start_h
    heappush(open_heap, (start_f, start_h, 0.0, start, entry_prev_node))

    closed = set()

    while open_heap:
        if explain:
            try:
                nodes_in_heap = {(entry[3], entry[4]) for entry in open_heap}
                entries = []
                for (node_e, prev_e) in nodes_in_heap:
                    best_g = g_cost.get((node_e, prev_e), float("inf"))
                    if not math.isfinite(best_g):
                        continue
                    h_e = heuristic(node_e, goal)
                    f_e = best_g + h_e
                    prev_str = prev_e if prev_e else "START"
                    entries.append((f_e, h_e, best_g, node_e, prev_str))
                entries.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
                snap_str = ", ".join([f"{n}(from {p}): f={f:.3f}, g={g:.3f}, h={h:.3f}" 
                                     for f,h,g,n,p in entries])
                if snap_str == "":
                    print("Open set snapshot: (no discovered nodes in open set)")
                else:
                    print(f"Open set snapshot: {snap_str}")
            except Exception:
                print(f"Open set (raw): {open_heap}")

        f, h, g, current, prev_node = heappop(open_heap)
        state = (current, prev_node)

        if g_cost.get(state, float("inf")) < g:
            if explain:
                best = g_cost.get(state, float("inf"))
                prev_str = prev_node if prev_node else "START"
                print(f"Skipping stale entry for {current}(from {prev_str}): heap g={g:.3f} > best g={best:.3f}")
            continue

        if explain:
            prev_str = prev_node if prev_node else "START"
            print(f"\nExpanding node {current} (from {prev_str}) with g={g:.3f}, h={h:.3f}, f={f:.3f}")

        if current == goal:
            path = []
            node_state = state
            while node_state in parent:
                path.append(node_state[0])
                node_state = parent[node_state]
            path.append(start)
            path.reverse()
            if explain:
                print("\nGoal reached. Reconstructing path:")
                for i in range(len(path)-1):
                    a = path[i]; b = path[i+1]
                    for nb, w in graph[a]:
                        if nb == b:
                            print(f"  {a} -> {b}: cost {w:.3f}")
                            break
                print(f"Total cost: {g_cost[state]:.3f}")
            return path, g_cost[state]

        closed.add(state)

        allowed_neighbors = get_allowed_neighbors(current, prev_node, graph, turn_restrictions)
        
        for neighbor, weight in allowed_neighbors:
            neighbor_state = (neighbor, current)
            
            if neighbor_state in closed:
                if explain:
                    print(f"  Neighbor {neighbor} (from {current}) already closed; skipping")
                continue

            tentative_g = g_cost[state] + weight
            h_neighbor = heuristic(neighbor, goal)
            f_cost = tentative_g + h_neighbor

            if explain:
                print(
                    f"  Evaluate {current}->{neighbor}: "
                    f"g({current})={g_cost[state]:.3f}; "
                    f"weight={weight:.3f}; "
                    f"tentative g({neighbor})={tentative_g:.3f}; "
                    f"h({neighbor})={h_neighbor:.3f}; "
                    f"f({neighbor})={f_cost:.3f}"
                )

            if tentative_g < g_cost.get(neighbor_state, float("inf")):
                if explain:
                    prev = g_cost.get(neighbor_state, float("inf"))
                    if prev == float("inf"):
                        print(f"    --> Improve {neighbor}(from {current}): g={tentative_g:.3f} (was inf); f={f_cost:.3f}")
                    else:
                        print(f"    --> Improve {neighbor}(from {current}): g={tentative_g:.3f} (was {prev:.3f}); f={f_cost:.3f}")
                g_cost[neighbor_state] = tentative_g
                parent[neighbor_state] = state
                heappush(open_heap, (f_cost, h_neighbor, tentative_g, neighbor, current))

    return None, float("inf")

graph = build_graph(edges, one_way_edges)

# A* implementation with turn restrictions based on previous node
def a_star(start, goal, graph, turn_restrictions=None, heuristic=None, explain=False):
    """
    A* search with turn restrictions from start to goal on weighted graph.
    
    State is (node, prev_node) to track entry direction for turn restrictions.
    
    graph[node] = [(neighbor, weight), ...]
    turn_restrictions = {node: {from_node: [allowed_exits]}}
    heuristic(node1, node2) returns estimated cost from node1 to node2.
    If heuristic is None, zero heuristic is used (equivalent to Dijkstra).
    Returns (path_list, total_cost) or (None, inf) if no path.
    """
    if heuristic is None:
        def heuristic(u, v): return 0.0
    
    if turn_restrictions is None:
        turn_restrictions = {}

    # g_cost: best known cost from start to state (node, prev_node)
    # We track states as (node, prev_node) to handle turn restrictions
    g_cost = {(start, None): 0.0}
    # parent pointers for path reconstruction: (node, prev) -> (parent_node, grandparent)
    parent = {}

    # priority queue: (f_cost, h_cost, g_cost, node, prev_node)
    open_heap = []
    start_h = heuristic(start, goal)
    start_f = start_h
    heappush(open_heap, (start_f, start_h, 0.0, start, None))

    closed = set()

    while open_heap:
        if explain:
            # Simplified explain for state-based search
            try:
                nodes_in_heap = {(entry[3], entry[4]) for entry in open_heap}
                entries = []
                for (node_e, prev_e) in nodes_in_heap:
                    best_g = g_cost.get((node_e, prev_e), float("inf"))
                    if not math.isfinite(best_g):
                        continue
                    h_e = heuristic(node_e, goal)
                    f_e = best_g + h_e
                    prev_str = prev_e if prev_e else "START"
                    entries.append((f_e, h_e, best_g, node_e, prev_str))
                entries.sort(key=lambda x: (x[0], x[1], x[2], str(x[3])))
                snap_str = ", ".join([f"{n}(from {p}): f={f:.3f}, g={g:.3f}, h={h:.3f}" 
                                     for f,h,g,n,p in entries])
                if snap_str == "":
                    print("Open set snapshot: (no discovered nodes in open set)")
                else:
                    print(f"Open set snapshot: {snap_str}")
            except Exception:
                print(f"Open set (raw): {open_heap}")

        # Pop the best candidate: (f, h, g, current_node, prev_node)
        f, h, g, current, prev_node = heappop(open_heap)
        
        state = (current, prev_node)

        # Skip stale entries
        if g_cost.get(state, float("inf")) < g:
            if explain:
                best = g_cost.get(state, float("inf"))
                prev_str = prev_node if prev_node else "START"
                print(f"Skipping stale entry for {current}(from {prev_str}): heap g={g:.3f} > best g={best:.3f}")
            continue

        if explain:
            prev_str = prev_node if prev_node else "START"
            print(f"\nExpanding node {current} (from {prev_str}) with g={g:.3f}, h={h:.3f}, f={f:.3f}")

        if current == goal:
            # Reconstruct path from parent pointers
            path = []
            node_state = state
            while node_state in parent:
                path.append(node_state[0])
                node_state = parent[node_state]
            path.append(start)
            path.reverse()
            if explain:
                print("\nGoal reached. Reconstructing path:")
                for i in range(len(path)-1):
                    a = path[i]; b = path[i+1]
                    for nb, w in graph[a]:
                        if nb == b:
                            print(f"  {a} -> {b}: cost {w:.3f}")
                            break
                print(f"Total cost: {g_cost[state]:.3f}")
            return path, g_cost[state]

        closed.add(state)

        # Get allowed neighbors based on turn restrictions
        allowed_neighbors = get_allowed_neighbors(current, prev_node, graph, turn_restrictions)
        
        for neighbor, weight in allowed_neighbors:
            neighbor_state = (neighbor, current)  # new state: at neighbor, came from current
            
            if neighbor_state in closed:
                if explain:
                    print(f"  Neighbor {neighbor} (from {current}) already closed; skipping")
                continue

            tentative_g = g_cost[state] + weight
            h_neighbor = heuristic(neighbor, goal)
            f_cost = tentative_g + h_neighbor

            if explain:
                print(
                    f"  Evaluate {current}->{neighbor}: "
                    f"g({current})={g_cost[state]:.3f}; "
                    f"weight={weight:.3f}; "
                    f"tentative g({neighbor})={tentative_g:.3f}; "
                    f"h({neighbor})={h_neighbor:.3f}; "
                    f"f({neighbor})={f_cost:.3f}"
                )

            if tentative_g < g_cost.get(neighbor_state, float("inf")):
                if explain:
                    prev = g_cost.get(neighbor_state, float("inf"))
                    if prev == float("inf"):
                        print(f"    --> Improve {neighbor}(from {current}): g={tentative_g:.3f} (was inf); f={f_cost:.3f}")
                    else:
                        print(f"    --> Improve {neighbor}(from {current}): g={tentative_g:.3f} (was {prev:.3f}); f={f_cost:.3f}")
                g_cost[neighbor_state] = tentative_g
                parent[neighbor_state] = state
                heappush(open_heap, (f_cost, h_neighbor, tentative_g, neighbor, current))

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
        
        # Count how many times each edge is used (normalize undirected edges)
        edge_usage_count = {}
        for a, b in path_edges:
            # Normalize edge to (min, max) so A->B and B->A are the same
            edge_key = tuple(sorted([a, b]))
            edge_usage_count[edge_key] = edge_usage_count.get(edge_key, 0) + 1
        
        # Separate edges by usage count
        single_use_edges = []
        multi_use_edges = []
        
        for a, b in path_edges:
            edge_key = tuple(sorted([a, b]))
            if edge_usage_count[edge_key] == 1:
                single_use_edges.append((a, b))
            elif edge_usage_count[edge_key] >= 2:
                multi_use_edges.append((a, b))
        
        # Draw single-use edges in red
        if single_use_edges:
            nx.draw_networkx_edges(G, pos, edgelist=single_use_edges, width=4.0, edge_color='red')
        
        # Draw multi-use edges in purple
        if multi_use_edges:
            nx.draw_networkx_edges(G, pos, edgelist=multi_use_edges, width=4.0, edge_color='purple')
    
    # draw edge labels last so they appear on top of everything
    edge_labels = {(u,v): f"{d['weight']:.1f}m" for u,v,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', 
                                   font_size=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))

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
    print("\nTaxi mission flow is fixed: HUB -> pickup -> dropoff -> HUB")

    pickup = input("Enter pickup node (not HUB): ").strip().upper()
    if pickup not in graph:
        print(f"Unknown pickup node '{pickup}'.")
        return
    if pickup == "HUB":
        print("Pickup node must be different from HUB.")
        return

    dropoff = input("Enter dropoff node (not HUB): ").strip().upper()
    if dropoff not in graph:
        print(f"Unknown dropoff node '{dropoff}'.")
        return
    if dropoff == "HUB":
        print("Dropoff node must be different from HUB.")
        return

    waypoints = ["HUB", pickup, dropoff, "HUB"]
    
    # Plan route through all waypoints
    print(f"\nPlanning route through: {' -> '.join(waypoints)}")
    full_path = []
    total_cost = 0.0
    prev_node_at_start = None  # Track where we came from for turn restrictions
    
    for i in range(len(waypoints) - 1):
        segment_start = waypoints[i]
        segment_goal = waypoints[i + 1]
        
        # For turn restrictions: if this isn't the first segment, we need to know
        # which node we came from to reach segment_start
        if i > 0:
            # The node before segment_start in the full path
            prev_node_at_start = full_path[-2] if len(full_path) >= 2 else None
            print(f"\n--- Segment {i+1}: {segment_start} to {segment_goal} (entering {segment_start} from {prev_node_at_start}) ---")
        else:
            prev_node_at_start = None
            print(f"\n--- Segment {i+1}: {segment_start} to {segment_goal} ---")
        
        # Modified A* call that respects entry direction at the start node
        path, cost = a_star_with_entry(segment_start, segment_goal, prev_node_at_start, 
                                       graph, turn_restrictions=turn_restrictions, 
                                       heuristic=euclidean_heuristic, explain=True)
        
        if path is None:
            print(f"ERROR: No path found from {segment_start} to {segment_goal}.")
            print(f"Cannot complete full route.")
            return
        
        # Add to full path (avoid duplicating nodes at segment boundaries)
        if i == 0:
            full_path.extend(path)
        else:
            full_path.extend(path[1:])  # Skip first node (already in path)
        
        total_cost += cost
        print(f"Segment path: {' -> '.join(path)}, distance: {cost:.2f}m")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE ROUTE: {' -> '.join(full_path)}")
    print(f"TOTAL DISTANCE: {total_cost:.2f} meters")
    print(f"{'='*60}")
    
    # Visualize the complete route
    try:
        visualize_graph(graph, path=full_path, out_fn='a_star_map.png')
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
