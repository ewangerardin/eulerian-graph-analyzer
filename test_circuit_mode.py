"""
Test script for the new circuit-only mode feature.

This script tests the circuit-only constraint to ensure it properly
rejects graphs that only have Eulerian paths but not circuits.
"""

import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver


def print_separator(char: str = "=", length: int = 70) -> None:
    """Print a separator line."""
    print(char * length)


def test_circuit_mode_with_circuit():
    """Test circuit-only mode with a graph that has an Eulerian circuit."""
    print_separator()
    print("TEST 1: Circuit-Only Mode with Eulerian Circuit")
    print_separator()
    print("Graph: Pentagon (5 vertices in a cycle)")
    print("Expected: Should find Eulerian circuit")
    print()

    # Create graph: Pentagon (cycle of 5 vertices)
    graph = Graph(5, directed=False)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print()

    # Test with circuit_only=False (normal mode)
    solver = EulerianSolver(graph)
    result_normal = solver.analyze(circuit_only=False)
    print("NORMAL MODE (Path or Circuit):")
    print(f"  Has Circuit: {result_normal.has_circuit}")
    print(f"  Has Path: {result_normal.has_path}")
    print(f"  Reason: {result_normal.reason}")
    print(f"  Path: {' -> '.join(map(str, result_normal.path))}")
    print()

    # Test with circuit_only=True
    result_circuit = solver.analyze(circuit_only=True)
    print("CIRCUIT-ONLY MODE:")
    print(f"  Has Circuit: {result_circuit.has_circuit}")
    print(f"  Has Path: {result_circuit.has_path}")
    print(f"  Reason: {result_circuit.reason}")
    print(f"  Path: {' -> '.join(map(str, result_circuit.path))}")
    print()

    # Verify results
    assert result_normal.has_circuit, "Normal mode should find circuit"
    assert result_normal.has_path, "Normal mode should find path"
    assert result_circuit.has_circuit, "Circuit-only mode should find circuit"
    assert result_circuit.has_path, "Circuit-only mode should find path"
    assert result_circuit.path[0] == result_circuit.path[-1], "Circuit should return to start"

    print("Test 1: PASSED")
    print()


def test_circuit_mode_with_path_only():
    """Test circuit-only mode with a graph that has only an Eulerian path."""
    print_separator()
    print("TEST 2: Circuit-Only Mode with Eulerian Path Only")
    print_separator()
    print("Graph: Modified pentagon with extra edge")
    print("Expected: Normal mode finds path, Circuit-only mode rejects")
    print()

    # Create graph with exactly 2 odd-degree vertices
    graph = Graph(5, directed=False)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 4)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(5):
        degree = graph.get_degree(i)
        print(f"  Vertex {i}: degree = {degree} ({'odd' if degree % 2 == 1 else 'even'})")
    print()

    # Test with circuit_only=False (normal mode)
    solver = EulerianSolver(graph)
    result_normal = solver.analyze(circuit_only=False)
    print("NORMAL MODE (Path or Circuit):")
    print(f"  Has Circuit: {result_normal.has_circuit}")
    print(f"  Has Path: {result_normal.has_path}")
    print(f"  Reason: {result_normal.reason}")
    if result_normal.has_path:
        print(f"  Path: {' -> '.join(map(str, result_normal.path))}")
    print()

    # Test with circuit_only=True
    result_circuit = solver.analyze(circuit_only=True)
    print("CIRCUIT-ONLY MODE:")
    print(f"  Has Circuit: {result_circuit.has_circuit}")
    print(f"  Has Path: {result_circuit.has_path}")
    print(f"  Reason: {result_circuit.reason}")
    if result_circuit.has_path:
        print(f"  Path: {' -> '.join(map(str, result_circuit.path))}")
    print()

    # Verify results
    assert not result_normal.has_circuit, "Normal mode should not find circuit"
    assert result_normal.has_path, "Normal mode should find path"
    assert not result_circuit.has_circuit, "Circuit-only mode should not find circuit"
    assert not result_circuit.has_path, "Circuit-only mode should not find path"
    assert "Circuit-only mode" in result_circuit.reason, "Reason should mention circuit-only mode"

    print("Test 2: PASSED")
    print()


def test_circuit_mode_directed_circuit():
    """Test circuit-only mode with a directed graph that has an Eulerian circuit."""
    print_separator()
    print("TEST 3: Circuit-Only Mode with Directed Eulerian Circuit")
    print_separator()
    print("Graph: Directed cycle of 4 vertices")
    print("Expected: Both modes should find circuit")
    print()

    # Create directed cycle
    graph = Graph(4, directed=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print()

    # Test with circuit_only=False (normal mode)
    solver = EulerianSolver(graph)
    result_normal = solver.analyze(circuit_only=False)
    print("NORMAL MODE (Path or Circuit):")
    print(f"  Has Circuit: {result_normal.has_circuit}")
    print(f"  Has Path: {result_normal.has_path}")
    print(f"  Reason: {result_normal.reason}")
    print(f"  Path: {' -> '.join(map(str, result_normal.path))}")
    print()

    # Test with circuit_only=True
    result_circuit = solver.analyze(circuit_only=True)
    print("CIRCUIT-ONLY MODE:")
    print(f"  Has Circuit: {result_circuit.has_circuit}")
    print(f"  Has Path: {result_circuit.has_path}")
    print(f"  Reason: {result_circuit.reason}")
    print(f"  Path: {' -> '.join(map(str, result_circuit.path))}")
    print()

    # Verify results
    assert result_normal.has_circuit, "Normal mode should find circuit"
    assert result_circuit.has_circuit, "Circuit-only mode should find circuit"

    print("Test 3: PASSED")
    print()


def test_circuit_mode_directed_path_only():
    """Test circuit-only mode with a directed graph that has only an Eulerian path."""
    print_separator()
    print("TEST 4: Circuit-Only Mode with Directed Eulerian Path Only")
    print_separator()
    print("Graph: Directed path with branches")
    print("Expected: Normal mode finds path, Circuit-only mode rejects")
    print()

    # Create directed graph with Eulerian path
    graph = Graph(5, directed=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]

    for u, v in edges:
        graph.add_edge(u, v)

    print("Edges:", edges)
    print()

    # Show degrees
    print("Vertex Degrees:")
    for i in range(5):
        in_deg = graph.get_in_degree(i)
        out_deg = graph.get_out_degree(i)
        diff = out_deg - in_deg
        print(f"  Vertex {i}: in-degree = {in_deg}, out-degree = {out_deg}, diff = {diff}")
    print()

    # Test with circuit_only=False (normal mode)
    solver = EulerianSolver(graph)
    result_normal = solver.analyze(circuit_only=False)
    print("NORMAL MODE (Path or Circuit):")
    print(f"  Has Circuit: {result_normal.has_circuit}")
    print(f"  Has Path: {result_normal.has_path}")
    print(f"  Reason: {result_normal.reason}")
    if result_normal.has_path:
        print(f"  Path: {' -> '.join(map(str, result_normal.path))}")
    print()

    # Test with circuit_only=True
    result_circuit = solver.analyze(circuit_only=True)
    print("CIRCUIT-ONLY MODE:")
    print(f"  Has Circuit: {result_circuit.has_circuit}")
    print(f"  Has Path: {result_circuit.has_path}")
    print(f"  Reason: {result_circuit.reason}")
    if result_circuit.has_path:
        print(f"  Path: {' -> '.join(map(str, result_circuit.path))}")
    print()

    # Verify results
    assert not result_normal.has_circuit, "Normal mode should not find circuit"
    assert result_normal.has_path, "Normal mode should find path"
    assert not result_circuit.has_circuit, "Circuit-only mode should not find circuit"
    assert not result_circuit.has_path, "Circuit-only mode should not find path"
    assert "Circuit-only mode" in result_circuit.reason, "Reason should mention circuit-only mode"

    print("Test 4: PASSED")
    print()


def run_all_tests():
    """Run all test cases."""
    print()
    print_separator("*")
    print("CIRCUIT-ONLY MODE TEST SUITE")
    print_separator("*")
    print()

    try:
        test_circuit_mode_with_circuit()
        test_circuit_mode_with_path_only()
        test_circuit_mode_directed_circuit()
        test_circuit_mode_directed_path_only()

        print_separator("*")
        print("ALL CIRCUIT-ONLY MODE TESTS PASSED SUCCESSFULLY!")
        print_separator("*")
        print()

    except AssertionError as e:
        print()
        print_separator("*")
        print(f"TEST FAILED: {e}")
        print_separator("*")
        print()
        raise

    except Exception as e:
        print()
        print_separator("*")
        print(f"ERROR: {e}")
        print_separator("*")
        print()
        raise


if __name__ == "__main__":
    run_all_tests()
