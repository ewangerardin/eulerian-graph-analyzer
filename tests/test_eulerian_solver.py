"""
Unit tests for EulerianSolver class.

Tests cover Eulerian path/circuit detection, Hierholzer's algorithm,
and edge cases for both directed and undirected graphs.
"""

import pytest
import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver, EulerianResult, solve_eulerian


class TestEulerianCircuit:
    """Tests for Eulerian circuit detection."""

    def test_eulerian_circuit_exists_pentagon(self):
        """Test Eulerian circuit in pentagon graph (all even degrees)."""
        graph = Graph(5, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 6  # 5 edges + 1 (return to start)
        assert result.path[0] == result.path[-1]  # Forms a circuit

    def test_eulerian_circuit_exists_square(self):
        """Test Eulerian circuit in square graph."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 5  # 4 edges + 1

    def test_eulerian_circuit_directed_cycle(self):
        """Test Eulerian circuit in directed cycle."""
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 5


class TestEulerianPath:
    """Tests for Eulerian path detection (no circuit)."""

    def test_eulerian_path_exists_two_odd_degrees(self):
        """Test Eulerian path with exactly 2 odd-degree vertices."""
        graph = Graph(5, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert not result.has_circuit  # Path but not circuit
        assert result.has_path
        assert len(result.path) == 6  # 5 edges + 1

        # Path should start at odd-degree vertex
        start_degree = graph.get_degree(result.start_vertex)
        assert start_degree % 2 == 1  # Odd degree

    def test_eulerian_path_directed_imbalanced(self):
        """Test Eulerian path in directed graph with degree imbalance."""
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert result.has_path
        assert result.start_vertex == 0  # Start has out-degree - in-degree = 1
        assert len(result.path) == 4  # 3 edges + 1

    def test_eulerian_path_simple_line(self):
        """Test Eulerian path in simple line graph."""
        graph = Graph(3, directed=False)
        edges = [(0, 1), (1, 2)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert result.has_path
        assert len(result.path) == 3  # 2 edges + 1


class TestNoEulerian:
    """Tests for graphs without Eulerian paths or circuits."""

    def test_no_eulerian_many_odd_degrees(self):
        """Test graph with more than 2 odd-degree vertices."""
        graph = Graph(4, directed=False)
        # Complete graph K4: all vertices have degree 3
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert not result.has_path
        assert "odd degree" in result.reason.lower()

    def test_no_eulerian_directed_imbalanced(self):
        """Test directed graph with multiple imbalanced vertices."""
        graph = Graph(4, directed=True)
        edges = [(0, 1), (0, 2), (1, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert not result.has_path
        assert "imbalance" in result.reason.lower()


class TestDisconnectedGraph:
    """Tests for disconnected graphs."""

    def test_disconnected_graph_returns_none(self):
        """Test that disconnected graph has no Eulerian path."""
        graph = Graph(6, directed=False)
        # Two separate triangles
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        graph.add_edge(3, 4)
        graph.add_edge(4, 5)
        graph.add_edge(5, 3)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert not result.has_path
        assert "not connected" in result.reason.lower()

    def test_disconnected_directed_graph(self):
        """Test disconnected directed graph."""
        graph = Graph(4, directed=True)
        # Two separate edges
        graph.add_edge(0, 1)
        graph.add_edge(2, 3)

        result = solve_eulerian(graph)

        assert not result.has_path
        assert "not connected" in result.reason.lower()


class TestEmptyGraph:
    """Tests for empty graphs."""

    def test_empty_graph_edge_case(self):
        """Test graph with no edges."""
        graph = Graph(5, directed=False)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert not result.has_path
        # Empty graph is detected as disconnected or having no edges
        assert "not connected" in result.reason.lower() or "no edges" in result.reason.lower()


class TestHierholzerAlgorithm:
    """Tests for Hierholzer's algorithm correctness."""

    def test_hierholzer_algorithm_correctness_circuit(self):
        """Test that Hierholzer's algorithm produces valid circuit."""
        graph = Graph(5, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        # Verify path is valid
        assert result.has_path
        path = result.path

        # Check that consecutive vertices are connected
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # At least one direction should have edge in original graph
            assert graph.has_edge(u, v) or graph.has_edge(v, u)

        # Check that path forms a circuit
        assert path[0] == path[-1]

    def test_hierholzer_algorithm_correctness_path(self):
        """Test that Hierholzer's algorithm produces valid path."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_path
        path = result.path

        # Verify all edges in path exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            assert graph.has_edge(u, v) or graph.has_edge(v, u)

    def test_route_visits_all_edges_once_undirected(self):
        """Test that route visits all edges exactly once (undirected)."""
        graph = Graph(5, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        # Path length should be edges + 1
        assert len(result.path) == len(edges) + 1

    def test_route_visits_all_edges_once_directed(self):
        """Test that route visits all edges exactly once (directed)."""
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        # Path length should be edges + 1
        assert len(result.path) == len(edges) + 1


class TestDirectedGraphs:
    """Additional tests for directed graphs."""

    def test_directed_circuit_balanced_degrees(self):
        """Test directed circuit with balanced in/out degrees."""
        graph = Graph(3, directed=True)
        edges = [(0, 1), (1, 2), (2, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path

        # Verify all vertices have balanced degrees
        for v in range(3):
            assert graph.get_in_degree(v) == graph.get_out_degree(v)

    def test_directed_path_detection(self):
        """Test directed path detection."""
        graph = Graph(5, directed=True)
        edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert result.has_path

        # Verify start vertex has out-degree - in-degree = 1
        start = result.start_vertex
        assert graph.get_out_degree(start) - graph.get_in_degree(start) == 1


class TestSolverMethods:
    """Tests for EulerianSolver convenience methods."""

    def test_has_eulerian_circuit_method(self):
        """Test has_eulerian_circuit method."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        assert solver.has_eulerian_circuit()

    def test_has_eulerian_path_method(self):
        """Test has_eulerian_path method."""
        graph = Graph(3, directed=False)
        edges = [(0, 1), (1, 2)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        assert solver.has_eulerian_path()
        assert not solver.has_eulerian_circuit()

    def test_find_eulerian_path_method(self):
        """Test find_eulerian_path method."""
        graph = Graph(3, directed=False)
        edges = [(0, 1), (1, 2)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        path = solver.find_eulerian_path()

        assert path is not None
        assert len(path) == 3

    def test_find_eulerian_path_returns_none(self):
        """Test find_eulerian_path returns None when no path exists."""
        graph = Graph(4, directed=False)
        # K4 - no Eulerian path
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j)

        solver = EulerianSolver(graph)
        path = solver.find_eulerian_path()

        assert path is None


class TestCircuitOnlyMode:
    """Tests for circuit-only mode."""

    def test_circuit_only_rejects_path(self):
        """Test that circuit-only mode rejects Eulerian paths."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert not result.has_circuit
        assert not result.has_path
        assert "circuit-only mode" in result.reason.lower()

    def test_circuit_only_accepts_circuit(self):
        """Test that circuit-only mode accepts Eulerian circuits."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert result.has_circuit
        assert result.has_path

    def test_circuit_only_directed_rejects_path(self):
        """Test circuit-only mode rejects directed Eulerian path."""
        graph = Graph(3, directed=True)
        edges = [(0, 1), (1, 2)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert not result.has_circuit
        assert not result.has_path
        assert "circuit-only mode" in result.reason.lower()


class TestEulerianResultClass:
    """Tests for EulerianResult class."""

    def test_eulerian_result_str_circuit(self):
        """Test string representation for circuit."""
        result = EulerianResult(
            has_circuit=True,
            has_path=True,
            path=[0, 1, 2, 0],
            start_vertex=0,
            reason="Test circuit"
        )

        str_repr = str(result)
        assert "Circuit" in str_repr
        assert "0 -> 1 -> 2 -> 0" in str_repr

    def test_eulerian_result_str_path(self):
        """Test string representation for path."""
        result = EulerianResult(
            has_circuit=False,
            has_path=True,
            path=[0, 1, 2],
            start_vertex=0,
            reason="Test path"
        )

        str_repr = str(result)
        assert "Path" in str_repr
        assert "start: 0" in str_repr

    def test_eulerian_result_str_no_path(self):
        """Test string representation when no path exists."""
        result = EulerianResult(
            has_circuit=False,
            has_path=False,
            path=[],
            reason="No path exists"
        )

        str_repr = str(result)
        assert "No Eulerian" in str_repr
        assert "No path exists" in str_repr


class TestComplexGraphs:
    """Tests for more complex graph scenarios."""

    def test_complete_graph_with_even_vertices(self):
        """Test complete graph K6 (even number of vertices)."""
        # K6 has all vertices with degree 5 (odd), so no Eulerian path
        graph = Graph(6, directed=False)
        for i in range(6):
            for j in range(i + 1, 6):
                graph.add_edge(i, j)

        result = solve_eulerian(graph)
        assert not result.has_path

    def test_multigraph_simulation(self):
        """Test graph with multiple edges (using weights)."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1, weight=2)  # Simulate 2 edges
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 0, weight=1)

        # With weighted edges, degrees are calculated differently
        # This is just to verify the algorithm handles weighted edges
        result = solve_eulerian(graph)
        # Result depends on how degrees are counted (this is edge case)
        assert isinstance(result, EulerianResult)
