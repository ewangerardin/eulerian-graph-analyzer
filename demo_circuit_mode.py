"""
Demonstration script for the circuit-only mode feature.

This script demonstrates the difference between normal mode and circuit-only mode
with easy-to-understand examples.
"""

import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def visualize_graph_ascii(graph: Graph, name: str) -> None:
    """Print a simple ASCII visualization of the graph."""
    print(f"{name}:")
    if not graph.directed:
        print("  (Undirected Graph)")
    else:
        print("  (Directed Graph)")

    print("\n  Adjacency Matrix:")
    print("    ", end="")
    for i in range(graph.num_vertices):
        print(f"{i:3}", end=" ")
    print()

    for i in range(graph.num_vertices):
        print(f"  {i} ", end="")
        for j in range(graph.num_vertices):
            print(f"{graph.adjacency_matrix[i][j]:3}", end=" ")
        print()
    print()


def compare_modes(graph: Graph, description: str) -> None:
    """Compare normal mode vs circuit-only mode for a given graph."""
    print_header(description)

    visualize_graph_ascii(graph, "Graph Structure")

    # Show vertex degrees
    print("  Vertex Degrees:")
    if graph.directed:
        for v in range(graph.num_vertices):
            in_deg = graph.get_in_degree(v)
            out_deg = graph.get_out_degree(v)
            print(f"    Vertex {v}: in={in_deg}, out={out_deg}, diff={out_deg-in_deg}")
    else:
        for v in range(graph.num_vertices):
            deg = graph.get_degree(v)
            parity = "even" if deg % 2 == 0 else "odd"
            print(f"    Vertex {v}: degree={deg} ({parity})")
    print()

    # Test normal mode
    solver = EulerianSolver(graph)
    result_normal = solver.analyze(circuit_only=False)

    print("  NORMAL MODE (Path or Circuit):")
    print(f"    Has Circuit: {result_normal.has_circuit}")
    print(f"    Has Path:    {result_normal.has_path}")
    if result_normal.has_path:
        print(f"    Start:       Vertex {result_normal.start_vertex}")
        print(f"    Path:        {' -> '.join(map(str, result_normal.path))}")
        if result_normal.path[0] == result_normal.path[-1]:
            print(f"    Note:        Path returns to start (forms a circuit)")
    print(f"    Reason:      {result_normal.reason}")
    print()

    # Test circuit-only mode
    result_circuit = solver.analyze(circuit_only=True)

    print("  CIRCUIT-ONLY MODE:")
    print(f"    Has Circuit: {result_circuit.has_circuit}")
    print(f"    Has Path:    {result_circuit.has_path}")
    if result_circuit.has_path:
        print(f"    Start:       Vertex {result_circuit.start_vertex}")
        print(f"    Path:        {' -> '.join(map(str, result_circuit.path))}")
        if result_circuit.path[0] == result_circuit.path[-1]:
            print(f"    Note:        Path returns to start (forms a circuit)")
    print(f"    Reason:      {result_circuit.reason}")
    print()

    # Compare results
    print("  COMPARISON:")
    if result_normal.has_path and result_circuit.has_path:
        print("    Both modes found a solution (Eulerian circuit exists)")
    elif result_normal.has_path and not result_circuit.has_path:
        print("    Normal mode: Found Eulerian path")
        print("    Circuit-only mode: REJECTED (path doesn't return to start)")
    else:
        print("    Both modes: No Eulerian path or circuit exists")
    print()


def main():
    """Run demonstration examples."""
    print("\n")
    print("*" * 70)
    print("  CIRCUIT-ONLY MODE FEATURE DEMONSTRATION")
    print("  Eulerian Graph Analyzer")
    print("*" * 70)

    # Example 1: Pentagon (has Eulerian circuit)
    print("\n\nEXAMPLE 1: Graph with Eulerian Circuit")
    print("-" * 70)
    graph1 = Graph(5, directed=False)
    edges1 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    for u, v in edges1:
        graph1.add_edge(u, v)

    print("\nEdges: ", edges1)
    print("\nVisual representation:")
    print("       0")
    print("      / \\")
    print("     4   1")
    print("      \\ /")
    print("    3---2")

    compare_modes(graph1, "Pentagon - All vertices have even degree")

    # Example 2: Pentagon with extra edge (has only Eulerian path)
    print("\n\nEXAMPLE 2: Graph with Eulerian Path Only (Not a Circuit)")
    print("-" * 70)
    graph2 = Graph(5, directed=False)
    edges2 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]
    for u, v in edges2:
        graph2.add_edge(u, v)

    print("\nEdges: ", edges2)
    print("\nVisual representation:")
    print("       0")
    print("      / \\")
    print("     4===1  (double edge 1-4)")
    print("      \\ /")
    print("    3---2")

    compare_modes(graph2, "Pentagon + Extra Edge - Two vertices have odd degree")

    # Example 3: Directed cycle (has Eulerian circuit)
    print("\n\nEXAMPLE 3: Directed Graph with Eulerian Circuit")
    print("-" * 70)
    graph3 = Graph(4, directed=True)
    edges3 = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for u, v in edges3:
        graph3.add_edge(u, v)

    print("\nDirected Edges: ", edges3)
    print("\nVisual representation:")
    print("     0 --> 1")
    print("     ^     |")
    print("     |     v")
    print("     3 <-- 2")

    compare_modes(graph3, "Directed Cycle - All vertices balanced (in=out)")

    # Example 4: Directed path (has only Eulerian path)
    print("\n\nEXAMPLE 4: Directed Graph with Eulerian Path Only")
    print("-" * 70)
    graph4 = Graph(4, directed=True)
    edges4 = [(0, 1), (1, 2), (2, 3)]
    for u, v in edges4:
        graph4.add_edge(u, v)

    print("\nDirected Edges: ", edges4)
    print("\nVisual representation:")
    print("     0 --> 1 --> 2 --> 3")

    compare_modes(graph4, "Directed Path - Start vertex (0) and end vertex (3)")

    # Example 5: Complete graph K4 (no Eulerian path)
    print("\n\nEXAMPLE 5: Graph with No Eulerian Path or Circuit")
    print("-" * 70)
    graph5 = Graph(4, directed=False)
    for i in range(4):
        for j in range(i + 1, 4):
            graph5.add_edge(i, j)

    print("\nComplete Graph K4 (all vertices connected to all others)")
    print("\nVisual representation:")
    print("     0 --- 1")
    print("     |\\   /|")
    print("     | \\ / |")
    print("     | / \\ |")
    print("     |/   \\|")
    print("     3 --- 2")

    compare_modes(graph5, "Complete K4 - All vertices have odd degree")

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n  When to use each mode:\n")
    print("  PATH OR CIRCUIT MODE (default):")
    print("    - Use when you want to find any Eulerian path")
    print("    - Accepts paths that start and end at different vertices")
    print("    - Also finds circuits (circuits are special cases of paths)")
    print()
    print("  CIRCUIT-ONLY MODE:")
    print("    - Use when you need the path to return to the starting vertex")
    print("    - Useful for scenarios like:")
    print("      * Delivery routes that must return to depot")
    print("      * Patrol routes that start and end at same location")
    print("      * Circuit board traces that must close")
    print()
    print("  Key Difference:")
    print("    Circuit-only mode REJECTS graphs that have Eulerian paths")
    print("    but not Eulerian circuits (paths that don't return to start)")
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
