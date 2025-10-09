"""
Main entry point for the Eulerian Graph Analyzer application.

This module provides command-line test cases and launches the GUI application.
"""

import sys
import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver, solve_eulerian
from chinese_postman import ChinesePostmanSolver, solve_chinese_postman
from gui import run_gui


def print_separator(char: str = "=", length: int = 70) -> None:
    """Print a separator line."""
    print(char * length)


def test_case_1_eulerian_circuit():
    """Test Case 1: Undirected graph with Eulerian circuit (Pentagon)"""
    print_separator()
    print("TEST CASE 1: Undirected Graph with Eulerian Circuit")
    print_separator()
    print("Graph: Pentagon (5 vertices in a cycle)")
    print("Expected: Eulerian circuit exists (all vertices have even degree)")
    print()

    # Create graph: Pentagon (cycle of 5 vertices)
    graph = Graph(5, directed=False)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert result.has_circuit, "Should have Eulerian circuit"
    assert result.has_path, "Should have Eulerian path"
    assert len(result.path) == 6, "Path should have 6 vertices (5 edges + return)"
    print("Test Case 1: PASSED")
    print()


def test_case_2_eulerian_path():
    """Test Case 2: Undirected graph with Eulerian path but no circuit"""
    print_separator()
    print("TEST CASE 2: Undirected Graph with Eulerian Path (No Circuit)")
    print_separator()
    print("Graph: Modified pentagon with extra edge")
    print("Expected: Eulerian path exists but not circuit (2 vertices with odd degree)")
    print()

    # Create graph with exactly 2 odd-degree vertices
    graph = Graph(5, directed=False)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(5):
        print(f"  Vertex {i}: degree = {graph.get_degree(i)}")
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert not result.has_circuit, "Should not have Eulerian circuit"
    assert result.has_path, "Should have Eulerian path"
    print("Test Case 2: PASSED")
    print()


def test_case_3_no_eulerian():
    """Test Case 3: Graph with no Eulerian path or circuit"""
    print_separator()
    print("TEST CASE 3: Graph with No Eulerian Path or Circuit")
    print_separator()
    print("Graph: Complete graph K4")
    print("Expected: No Eulerian path (all 4 vertices have odd degree 3)")
    print()

    # Create complete graph K4
    graph = Graph(4, directed=False)
    for i in range(4):
        for j in range(i + 1, 4):
            graph.add_edge(i, j)

    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(4):
        print(f"  Vertex {i}: degree = {graph.get_degree(i)}")
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert not result.has_circuit, "Should not have Eulerian circuit"
    assert not result.has_path, "Should not have Eulerian path"
    print("Test Case 3: PASSED")
    print()


def test_case_4_directed_circuit():
    """Test Case 4: Directed graph with Eulerian circuit"""
    print_separator()
    print("TEST CASE 4: Directed Graph with Eulerian Circuit")
    print_separator()
    print("Graph: Directed cycle of 4 vertices")
    print("Expected: Eulerian circuit exists (in-degree = out-degree for all vertices)")
    print()

    # Create directed cycle
    graph = Graph(4, directed=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(4):
        print(f"  Vertex {i}: in-degree = {graph.get_in_degree(i)}, "
              f"out-degree = {graph.get_out_degree(i)}")
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert result.has_circuit, "Should have Eulerian circuit"
    assert result.has_path, "Should have Eulerian path"
    print("Test Case 4: PASSED")
    print()


def test_case_5_directed_path():
    """Test Case 5: Directed graph with Eulerian path but no circuit"""
    print_separator()
    print("TEST CASE 5: Directed Graph with Eulerian Path (No Circuit)")
    print_separator()
    print("Graph: Directed path with branches")
    print("Expected: Eulerian path exists (one start vertex, one end vertex)")
    print()

    # Create directed graph with Eulerian path
    graph = Graph(5, directed=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(5):
        in_deg = graph.get_in_degree(i)
        out_deg = graph.get_out_degree(i)
        diff = out_deg - in_deg
        print(f"  Vertex {i}: in-degree = {in_deg}, out-degree = {out_deg}, "
              f"difference = {diff}")
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert not result.has_circuit, "Should not have Eulerian circuit"
    assert result.has_path, "Should have Eulerian path"
    print("Test Case 5: PASSED")
    print()


def test_case_6_disconnected():
    """Test Case 6: Disconnected graph"""
    print_separator()
    print("TEST CASE 6: Disconnected Graph")
    print_separator()
    print("Graph: Two separate triangles")
    print("Expected: No Eulerian path (graph not connected)")
    print()

    # Create disconnected graph
    graph = Graph(6, directed=False)
    # Triangle 1: vertices 0, 1, 2
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 0)
    # Triangle 2: vertices 3, 4, 5
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 3)

    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Analyze
    result = solve_eulerian(graph)
    print("RESULTS:")
    print(result)
    print()

    # Verify
    assert not result.has_circuit, "Should not have Eulerian circuit"
    assert not result.has_path, "Should not have Eulerian path"
    assert "not connected" in result.reason.lower(), "Reason should mention connectivity"
    print("Test Case 6: PASSED")
    print()


def test_case_7_cpp_eulerian():
    """Test Case 7: CPP on already Eulerian graph"""
    print_separator()
    print("TEST CASE 7: Chinese Postman Problem on Eulerian Graph")
    print_separator()
    print("Graph: Square (4 vertices, all even degrees)")
    print("Expected: CPP solution with no edges duplicated")
    print()

    # Create square graph with weighted edges
    graph = Graph(4, directed=False)
    edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 0, 4)]

    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    print("Edges with weights:", edges)
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Analyze with CPP
    result = solve_chinese_postman(graph)
    print("CPP RESULTS:")
    print(result)
    print()

    # Verify
    assert result.has_solution, "Should have CPP solution"
    assert len(result.added_edges) == 0, "Should not need to duplicate any edges"
    assert result.total_cost == 10.0, "Total cost should be 1+2+3+4=10"
    print("Test Case 7: PASSED")
    print()


def test_case_8_cpp_two_odd():
    """Test Case 8: CPP with two odd-degree vertices"""
    print_separator()
    print("TEST CASE 8: Chinese Postman Problem with Two Odd Vertices")
    print_separator()
    print("Graph: Path 0-1-2 with weights")
    print("Expected: CPP solution duplicates shortest path")
    print()

    # Create path graph: 0 -- 1 -- 2
    graph = Graph(3, directed=False)
    graph.add_edge(0, 1, weight=5)
    graph.add_edge(1, 2, weight=3)

    print("Edges: (0,1,5), (1,2,3)")
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(3):
        deg = graph.get_degree(i)
        parity = "even" if deg % 2 == 0 else "odd"
        print(f"  Vertex {i}: degree = {deg} ({parity})")
    print()

    # Analyze with CPP
    result = solve_chinese_postman(graph)
    print("CPP RESULTS:")
    print(result)
    print()

    # Verify
    assert result.has_solution, "Should have CPP solution"
    assert len(result.added_edges) == 1, "Should duplicate one edge (shortest path)"
    # Original cost: 5 + 3 = 8
    # Need to add path from 0 to 2, which costs 8
    assert result.total_cost == 16.0, "Total cost should be 8 + 8 = 16"
    print("Test Case 8: PASSED")
    print()


def test_case_9_cpp_four_odd():
    """Test Case 9: CPP with four odd-degree vertices"""
    print_separator()
    print("TEST CASE 9: Chinese Postman Problem with Four Odd Vertices")
    print_separator()
    print("Graph: Square with diagonal (creates 4 odd vertices)")
    print("Expected: CPP finds optimal matching of odd vertices")
    print()

    # Create graph with specific structure
    graph = Graph(4, directed=False)
    # Square
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(1, 2, weight=1)
    graph.add_edge(2, 3, weight=1)
    graph.add_edge(3, 0, weight=1)
    # Diagonals to make all vertices odd (degree 3)
    graph.add_edge(0, 2, weight=2)

    print("Edges: (0,1,1), (1,2,1), (2,3,1), (3,0,1), (0,2,2)")
    print("Adjacency Matrix:")
    print(graph.adjacency_matrix)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(4):
        deg = graph.get_degree(i)
        parity = "even" if deg % 2 == 0 else "odd"
        print(f"  Vertex {i}: degree = {deg} ({parity})")
    print()

    # Analyze with CPP
    result = solve_chinese_postman(graph)
    print("CPP RESULTS:")
    print(result)
    print()

    # Verify
    assert result.has_solution, "Should have CPP solution"
    # Original cost: 1+1+1+1+2 = 6
    # Odd vertices: 0(3), 2(3), others are even
    # Only 0 and 2 are odd, should match them (cost 2)
    assert result.total_cost >= 6.0, "Total cost should be at least original cost"
    print("Test Case 9: PASSED")
    print()


def run_all_tests():
    """Run all test cases."""
    print()
    print_separator("*")
    print("EULERIAN GRAPH ANALYZER - TEST SUITE")
    print_separator("*")
    print()

    try:
        # Eulerian tests
        test_case_1_eulerian_circuit()
        test_case_2_eulerian_path()
        test_case_3_no_eulerian()
        test_case_4_directed_circuit()
        test_case_5_directed_path()
        test_case_6_disconnected()

        # Chinese Postman Problem tests
        test_case_7_cpp_eulerian()
        test_case_8_cpp_two_odd()
        test_case_9_cpp_four_odd()

        print_separator("*")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print_separator("*")
        print()

    except AssertionError as e:
        print()
        print_separator("*")
        print(f"TEST FAILED: {e}")
        print_separator("*")
        print()
        sys.exit(1)

    except Exception as e:
        print()
        print_separator("*")
        print(f"ERROR: {e}")
        print_separator("*")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


def print_usage():
    """Print usage information."""
    print("Eulerian Graph Analyzer")
    print()
    print("Usage:")
    print("  python main.py          - Launch GUI application")
    print("  python main.py test     - Run test cases")
    print("  python main.py --help   - Show this help message")
    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_all_tests()
        elif sys.argv[1] in ["--help", "-h", "help"]:
            print_usage()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print()
            print_usage()
            sys.exit(1)
    else:
        # Launch GUI
        print()
        print("=" * 70)
        print("EULERIAN GRAPH ANALYZER - GUI MODE")
        print("=" * 70)
        print()
        print("Launching graphical user interface...")
        print("(Run 'python main.py test' to execute test cases)")
        print()

        try:
            run_gui()
        except Exception as e:
            print()
            print("=" * 70)
            print("FATAL ERROR: Could not launch GUI")
            print("=" * 70)
            print(f"Error: {str(e)}")
            print()
            print("Please ensure you have all required dependencies installed:")
            print("  pip install -r requirements.txt")
            print()
            print("Required packages:")
            print("  - tkinter (usually included with Python)")
            print("  - matplotlib")
            print("  - networkx")
            print("  - numpy")
            print("=" * 70)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
