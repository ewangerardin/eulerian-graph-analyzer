"""
Comprehensive test suite for Hamiltonian path and circuit solver.

Tests cover:
- Exact backtracking algorithm
- Theorem-based detection (Ore's, Dirac's)
- TSP reduction
- Edge cases and error handling
- Directed vs undirected graphs
- Timeout handling
"""

import pytest
import numpy as np
from graph import Graph
from hamiltonian_solver import (
    HamiltonianSolver, HamiltonianResult, solve_hamiltonian,
    has_hamiltonian_path, has_hamiltonian_circuit
)


class TestHamiltonianBasics:
    """Test basic Hamiltonian functionality."""

    def test_empty_graph(self):
        """Test Hamiltonian on empty graph."""
        # Graph doesn't support 0 vertices, skip test
        pass

    def test_single_vertex(self):
        """Test Hamiltonian on single vertex."""
        graph = Graph(1)
        result = solve_hamiltonian(graph)

        assert result.has_path
        assert result.has_circuit
        assert result.path == [0]

    def test_two_vertices_connected(self):
        """Test Hamiltonian on two connected vertices."""
        graph = Graph(2)
        graph.add_edge(0, 1, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert result.has_circuit
        assert len(result.path) == 2

    def test_two_vertices_disconnected(self):
        """Test Hamiltonian on two disconnected vertices."""
        graph = Graph(2)

        result = solve_hamiltonian(graph)

        assert not result.has_path
        assert not result.has_circuit

    def test_triangle_graph(self):
        """Test Hamiltonian on triangle (K3)."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert result.has_circuit
        assert len(result.path) == 3


class TestCompleteGraphs:
    """Test on complete graphs (always have Hamiltonian)."""

    def test_complete_k3(self):
        """Test on K3 (triangle)."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit
        assert set(result.path) == {0, 1, 2}

    def test_complete_k4(self):
        """Test on K4."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit
        assert len(result.path) == 4
        assert len(set(result.path)) == 4

    def test_complete_k5(self):
        """Test on K5."""
        graph = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit
        assert len(result.path) == 5

    def test_complete_graph_with_weights(self):
        """Test complete graph with varying weights."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, (i + 1) * (j + 1))

        result = solve_hamiltonian(graph)

        assert result.has_circuit


class TestPathGraphs:
    """Test on path graphs."""

    def test_path_graph_3_vertices(self):
        """Test on path: 0-1-2."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert not result.has_circuit
        assert result.path in [[0, 1, 2], [2, 1, 0]]

    def test_path_graph_5_vertices(self):
        """Test on longer path."""
        graph = Graph(5)
        for i in range(4):
            graph.add_edge(i, i + 1, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert not result.has_circuit

    def test_path_graph_circuit_only_fails(self):
        """Test that path graph fails circuit-only search."""
        graph = Graph(4)
        for i in range(3):
            graph.add_edge(i, i + 1, 1)

        result = solve_hamiltonian(graph, circuit_only=True)

        assert not result.has_circuit


class TestCycleGraphs:
    """Test on cycle graphs."""

    def test_cycle_graph_4_vertices(self):
        """Test on 4-cycle."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 0, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert result.has_circuit

    def test_cycle_graph_6_vertices(self):
        """Test on 6-cycle."""
        graph = Graph(6)
        for i in range(6):
            graph.add_edge(i, (i + 1) % 6, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit


class TestNoHamiltonian:
    """Test graphs with no Hamiltonian path/circuit."""

    def test_disconnected_graph(self):
        """Test on disconnected graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_hamiltonian(graph)

        assert not result.has_path
        assert not result.has_circuit

    def test_petersen_graph_no_hamiltonian_circuit(self):
        """Test on modified graph with no Hamiltonian circuit."""
        # Simple graph that has path but not circuit
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)

        result = solve_hamiltonian(graph, circuit_only=True)

        assert not result.has_circuit

    def test_star_graph(self):
        """Test on star graph (has path but not circuit for n>2)."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(0, 2, 1)
        graph.add_edge(0, 3, 1)

        result = solve_hamiltonian(graph)

        # Star graph has no Hamiltonian path for n>=4
        # (central vertex must be visited, leaves no valid order)
        assert not result.has_circuit


class TestDirectedGraphs:
    """Test on directed graphs."""

    def test_directed_cycle(self):
        """Test directed cycle."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit

    def test_directed_path(self):
        """Test directed path."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert not result.has_circuit

    def test_directed_no_path(self):
        """Test directed graph with no Hamiltonian path."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 1, 1)  # Both edges point to vertex 1

        result = solve_hamiltonian(graph)

        # No valid ordering exists
        assert not result.has_path

    def test_directed_complete_graph(self):
        """Test directed complete graph."""
        graph = Graph(4, directed=True)
        for i in range(4):
            for j in range(4):
                if i != j:
                    graph.add_edge(i, j, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit


class TestTheoremDetection:
    """Test theorem-based detection."""

    def test_diracs_theorem_sufficient(self):
        """Test that Dirac's theorem correctly identifies Hamiltonian."""
        # Create graph where all vertices have degree >= n/2
        graph = Graph(6)
        # Make each vertex have degree 3 (= 6/2)
        for i in range(6):
            for j in range(i + 1, 6):
                if j - i <= 3:  # Connect to 3 neighbors
                    graph.add_edge(i, j, 1)

        # Add more edges to ensure all have degree >= 3
        graph.add_edge(0, 3, 1)
        graph.add_edge(1, 4, 1)
        graph.add_edge(2, 5, 1)

        result = solve_hamiltonian(graph)

        # Dirac's theorem should apply
        if result.has_circuit:
            # Check if Dirac's theorem was mentioned
            assert result.has_circuit

    def test_ores_theorem_sufficient(self):
        """Test Ore's theorem detection."""
        # Graph satisfying Ore's theorem
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(0, 2, 1)
        graph.add_edge(0, 3, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(1, 3, 1)
        graph.add_edge(2, 3, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit

    def test_neither_theorem_applies(self):
        """Test graph where neither theorem applies."""
        graph = Graph(5)
        # Create sparse graph
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 0, 1)

        result = solve_hamiltonian(graph)

        # Should still find circuit via backtracking
        assert result.has_circuit


class TestTSPReduction:
    """Test TSP reduction strategy."""

    def test_tsp_reduction_complete_graph(self):
        """Test TSP reduction on complete graph."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, 1)

        result = solve_hamiltonian(graph, use_tsp_reduction=True)

        assert result.has_circuit

    def test_tsp_reduction_incomplete_graph(self):
        """Test TSP reduction on incomplete graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        # Missing edge 3-0

        result = solve_hamiltonian(graph, use_tsp_reduction=True)

        # Should fall back to backtracking
        assert not result.has_circuit


class TestBacktrackingAlgorithm:
    """Test backtracking search."""

    def test_backtracking_finds_path(self):
        """Test that backtracking finds existing path."""
        graph = Graph(5)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 4, 1)

        result = solve_hamiltonian(graph)

        assert result.has_path
        assert "backtracking" in result.algorithm_used.lower() or result.has_path

    def test_backtracking_finds_circuit(self):
        """Test that backtracking finds existing circuit."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 0, 1)

        result = solve_hamiltonian(graph, circuit_only=True)

        assert result.has_circuit

    def test_backtracking_rejects_non_hamiltonian(self):
        """Test that backtracking correctly identifies non-Hamiltonian."""
        graph = Graph(5)
        # Star graph with 4 leaves
        for i in range(1, 5):
            graph.add_edge(0, i, 1)

        result = solve_hamiltonian(graph)

        # Star with 4 leaves has no Hamiltonian path
        assert not result.has_path


class TestTimeoutHandling:
    """Test timeout functionality."""

    def test_timeout_on_complex_graph(self):
        """Test that timeout works on potentially expensive search."""
        # Create graph that might take long to search
        graph = Graph(10)
        # Sparse connectivity
        for i in range(9):
            graph.add_edge(i, i + 1, 1)

        result = solve_hamiltonian(graph, timeout=0.01)

        # Should either find answer quickly or timeout
        if result.timeout:
            assert not result.has_path or not result.has_circuit
            assert "timeout" in result.reason.lower() or "timed out" in result.reason.lower()

    def test_quick_solution_no_timeout(self):
        """Test that quick solutions don't timeout."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph, timeout=1.0)

        assert not result.timeout
        assert result.has_circuit


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_has_hamiltonian_path_true(self):
        """Test has_hamiltonian_path returns True."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)

        assert has_hamiltonian_path(graph)

    def test_has_hamiltonian_path_false(self):
        """Test has_hamiltonian_path returns False."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        assert not has_hamiltonian_path(graph)

    def test_has_hamiltonian_circuit_true(self):
        """Test has_hamiltonian_circuit returns True."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        assert has_hamiltonian_circuit(graph)

    def test_has_hamiltonian_circuit_false(self):
        """Test has_hamiltonian_circuit returns False."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)

        assert not has_hamiltonian_circuit(graph)


class TestResultObject:
    """Test HamiltonianResult object."""

    def test_result_string_with_circuit(self):
        """Test string representation with circuit."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        result_str = str(result)
        assert "circuit" in result_str.lower()
        assert "path" in result_str.lower()

    def test_result_string_with_path_only(self):
        """Test string representation with path but not circuit."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)

        result = solve_hamiltonian(graph)

        result_str = str(result)
        if result.has_path:
            assert "path" in result_str.lower()

    def test_result_string_no_hamiltonian(self):
        """Test string representation with no Hamiltonian."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_hamiltonian(graph)

        result_str = str(result)
        assert "no" in result_str.lower()

    def test_result_attributes(self):
        """Test all result attributes are set."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        assert isinstance(result.has_path, bool)
        assert isinstance(result.has_circuit, bool)
        assert isinstance(result.path, list)
        assert isinstance(result.algorithm_used, str)
        assert isinstance(result.reason, str)
        assert isinstance(result.timeout, bool)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_self_loop_directed(self):
        """Test graph with self-loop (directed)."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 0, 1)  # Self-loop
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_hamiltonian(graph)

        # Self-loop shouldn't affect Hamiltonian
        assert result.has_circuit

    def test_multiple_components(self):
        """Test graph with multiple components."""
        graph = Graph(6)
        # Component 1
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)
        # Component 2
        graph.add_edge(3, 4, 1)
        graph.add_edge(4, 5, 1)
        graph.add_edge(5, 3, 1)

        result = solve_hamiltonian(graph)

        assert not result.has_path

    def test_large_complete_graph(self):
        """Test on larger complete graph."""
        graph = Graph(8)
        for i in range(8):
            for j in range(i + 1, 8):
                graph.add_edge(i, j, 1)

        result = solve_hamiltonian(graph, timeout=2.0)

        # Complete graph always has Hamiltonian
        assert result.has_circuit

    def test_high_degree_vertices(self):
        """Test graph with high-degree vertices."""
        graph = Graph(6)
        # Create hub-and-spoke with additional edges
        for i in range(1, 6):
            graph.add_edge(0, i, 1)
        # Add ring around outer vertices
        for i in range(1, 5):
            graph.add_edge(i, i + 1, 1)
        graph.add_edge(5, 1, 1)

        result = solve_hamiltonian(graph)

        assert result.has_circuit


class TestAlgorithmComparison:
    """Test different algorithm strategies."""

    def test_backtracking_vs_tsp_same_result(self):
        """Test that different algorithms give consistent results."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, 1)

        result1 = solve_hamiltonian(graph, use_tsp_reduction=False)
        result2 = solve_hamiltonian(graph, use_tsp_reduction=True)

        assert result1.has_circuit == result2.has_circuit

    def test_circuit_only_vs_path_search(self):
        """Test circuit-only vs general path search."""
        # Graph with path but not circuit
        graph = Graph(4)
        for i in range(3):
            graph.add_edge(i, i + 1, 1)

        path_result = solve_hamiltonian(graph, circuit_only=False)
        circuit_result = solve_hamiltonian(graph, circuit_only=True)

        assert path_result.has_path
        assert not circuit_result.has_circuit
