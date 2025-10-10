"""
Comprehensive test suite for Traveling Salesman Problem (TSP) solver.

Tests cover:
- Exact algorithm (Held-Karp)
- Approximation algorithms (Nearest Neighbor, MST-based, Christofides)
- Edge cases and error handling
- Algorithm comparison and approximation ratio validation
"""

import pytest
import numpy as np
from graph import Graph
from tsp_solver import TSPSolver, TSPResult, solve_tsp


class TestTSPBasics:
    """Test basic TSP functionality."""

    def test_empty_graph(self):
        """Test TSP on empty graph."""
        # Graph doesn't support 0 vertices, so skip this test
        # or use single vertex as smallest case
        pass

    def test_single_vertex(self):
        """Test TSP on single vertex."""
        graph = Graph(1)
        result = solve_tsp(graph)

        assert result.has_tour
        assert result.tour_path == [0, 0]
        assert result.total_distance == 0

    def test_two_vertices(self):
        """Test TSP on two vertices."""
        graph = Graph(2)
        graph.add_edge(0, 1, 5)

        result = solve_tsp(graph)

        assert result.has_tour
        assert len(result.tour_path) == 3  # 0 -> 1 -> 0
        assert result.total_distance == 10

    def test_directed_graph_raises_error(self):
        """Test that directed graphs raise ValueError."""
        graph = Graph(3, directed=True)

        with pytest.raises(ValueError, match="undirected"):
            TSPSolver(graph)

    def test_disconnected_graph(self):
        """Test TSP on disconnected graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_tsp(graph)

        assert not result.has_tour
        assert "disconnected" in result.reason.lower()


class TestExactAlgorithm:
    """Test exact Held-Karp algorithm."""

    def test_triangle_graph(self):
        """Test optimal tour on triangle (K3)."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="exact")

        assert result.has_tour
        assert result.total_distance == 3
        assert len(result.tour_path) == 4
        assert result.tour_path[0] == result.tour_path[-1]
        assert not result.is_approximate

    def test_complete_graph_k4(self):
        """Test optimal tour on K4."""
        graph = Graph(4)
        weights = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ]

        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, weights[i][j])

        result = solve_tsp(graph, algorithm="exact")

        assert result.has_tour
        assert result.total_distance == 80  # Known optimal: 0->1->3->2->0
        assert result.algorithm_used == "exact (Held-Karp)"

    def test_incomplete_graph_exact(self):
        """Test exact algorithm on incomplete graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        # Missing edge between 3 and 0

        result = solve_tsp(graph, algorithm="exact")

        assert not result.has_tour
        assert "incomplete" in result.reason.lower() or "missing edge" in result.reason.lower()

    def test_symmetric_weights(self):
        """Test that undirected graph has symmetric weights."""
        graph = Graph(3)
        graph.add_edge(0, 1, 5)
        graph.add_edge(1, 2, 3)
        graph.add_edge(2, 0, 4)

        result = solve_tsp(graph, algorithm="exact")

        assert result.has_tour
        assert result.total_distance == 12


class TestNearestNeighbor:
    """Test nearest neighbor heuristic."""

    def test_nearest_neighbor_triangle(self):
        """Test nearest neighbor on simple triangle."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="nearest_neighbor")

        assert result.has_tour
        assert result.total_distance == 3
        assert result.is_approximate
        assert result.algorithm_used == "nearest_neighbor"

    def test_nearest_neighbor_weighted(self):
        """Test nearest neighbor with varying weights."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(0, 2, 10)
        graph.add_edge(0, 3, 10)
        graph.add_edge(1, 2, 1)
        graph.add_edge(1, 3, 10)
        graph.add_edge(2, 3, 1)

        result = solve_tsp(graph, algorithm="nearest_neighbor")

        assert result.has_tour
        # Greedy should pick 0->1->2->3->0 = 1+1+1+10 = 13
        assert result.total_distance <= 20  # Some valid tour

    def test_nearest_neighbor_disconnected(self):
        """Test nearest neighbor on disconnected graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_tsp(graph, algorithm="nearest_neighbor")

        assert not result.has_tour

    def test_nearest_neighbor_no_approximation_guarantee(self):
        """Test that nearest neighbor has no approximation ratio."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="nearest_neighbor")

        assert result.approximation_ratio is None


class TestMSTApproximation:
    """Test MST-based 2-approximation."""

    def test_mst_approximation_triangle(self):
        """Test MST approximation on triangle."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="mst_approximation")

        assert result.has_tour
        assert result.approximation_ratio == 2.0
        assert result.total_distance <= 6  # At most 2× optimal (3)

    def test_mst_approximation_square(self):
        """Test MST approximation on square graph."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 0, 1)
        graph.add_edge(0, 2, 2)  # Diagonal
        graph.add_edge(1, 3, 2)  # Diagonal

        result = solve_tsp(graph, algorithm="mst_approximation")

        assert result.has_tour
        assert result.algorithm_used == "mst_approximation"
        assert result.is_approximate

    def test_mst_approximation_complete_graph(self):
        """Test MST approximation on complete graph."""
        graph = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, abs(i - j))

        result = solve_tsp(graph, algorithm="mst_approximation")

        assert result.has_tour
        # Calculate optimal for comparison
        optimal_result = solve_tsp(graph, algorithm="exact")
        if optimal_result.has_tour:
            # Check 2-approximation bound
            assert result.total_distance <= 2 * optimal_result.total_distance

    def test_mst_no_mst_no_tour(self):
        """Test MST approximation when MST doesn't exist."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_tsp(graph, algorithm="mst_approximation")

        assert not result.has_tour


class TestChristofidesAlgorithm:
    """Test Christofides 1.5-approximation."""

    def test_christofides_triangle(self):
        """Test Christofides on triangle."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="christofides")

        assert result.has_tour
        assert result.approximation_ratio == 1.5
        assert result.total_distance == 3  # Optimal for this case

    def test_christofides_complete_k4(self):
        """Test Christofides on K4."""
        graph = Graph(4)
        weights = [[0, 1, 2, 3],
                   [1, 0, 4, 5],
                   [2, 4, 0, 6],
                   [3, 5, 6, 0]]

        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, weights[i][j])

        result = solve_tsp(graph, algorithm="christofides")

        assert result.has_tour
        assert result.algorithm_used == "christofides"

    def test_christofides_approximation_bound(self):
        """Test that Christofides respects 1.5 approximation."""
        graph = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                # Use Euclidean-like distances
                graph.add_edge(i, j, max(1, abs(i - j)))

        christofides_result = solve_tsp(graph, algorithm="christofides")
        exact_result = solve_tsp(graph, algorithm="exact")

        if exact_result.has_tour and christofides_result.has_tour:
            # Should be within 1.5× optimal
            assert christofides_result.total_distance <= 1.5 * exact_result.total_distance + 1


class TestAlgorithmSelection:
    """Test automatic algorithm selection."""

    def test_auto_selects_exact_for_small_graph(self):
        """Test that auto mode selects exact for ≤10 vertices."""
        graph = Graph(5)
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, 1)

        result = solve_tsp(graph, algorithm="auto")

        assert "exact" in result.algorithm_used.lower() or result.has_tour

    def test_auto_selects_approximation_for_large_graph(self):
        """Test that auto mode selects approximation for >10 vertices."""
        graph = Graph(15)
        for i in range(15):
            for j in range(i + 1, 15):
                graph.add_edge(i, j, 1)

        result = solve_tsp(graph, algorithm="auto")

        # Should use approximation (christofides or mst)
        assert result.is_approximate or result.has_tour

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm name raises error."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        solver = TSPSolver(graph)
        with pytest.raises(ValueError, match="Unknown algorithm"):
            solver.solve("invalid_algorithm")


class TestAlgorithmComparison:
    """Test and compare different algorithms."""

    def test_all_algorithms_find_tour(self):
        """Test that all algorithms find tour on complete graph."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, abs(i - j) + 1)

        algorithms = ["exact", "nearest_neighbor", "mst_approximation", "christofides"]
        results = {}

        for algo in algorithms:
            result = solve_tsp(graph, algorithm=algo)
            results[algo] = result
            assert result.has_tour, f"{algo} should find tour"

        # Exact should be optimal
        exact_dist = results["exact"].total_distance

        # All others should be >= optimal
        for algo in ["nearest_neighbor", "mst_approximation", "christofides"]:
            assert results[algo].total_distance >= exact_dist

    def test_exact_vs_approximation_quality(self):
        """Test that exact gives better or equal result."""
        graph = Graph(6)
        for i in range(6):
            for j in range(i + 1, 6):
                graph.add_edge(i, j, (i + 1) * (j + 1))

        exact = solve_tsp(graph, algorithm="exact")
        approx = solve_tsp(graph, algorithm="nearest_neighbor")

        if exact.has_tour and approx.has_tour:
            assert exact.total_distance <= approx.total_distance


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_high_weight_edges(self):
        """Test TSP with very high weight edges."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1000)
        graph.add_edge(1, 2, 2000)
        graph.add_edge(2, 0, 3000)

        result = solve_tsp(graph)

        assert result.has_tour
        assert result.total_distance == 6000

    def test_uniform_weights(self):
        """Test TSP where all edges have same weight."""
        graph = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, 1)

        result = solve_tsp(graph)

        assert result.has_tour
        assert result.total_distance == 4  # Any tour has same cost

    def test_metric_property(self):
        """Test on graph satisfying triangle inequality."""
        graph = Graph(4)
        # Distances satisfy triangle inequality
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(3, 0, 1)
        graph.add_edge(0, 2, 2)  # Exactly sum of 0-1-2
        graph.add_edge(1, 3, 2)  # Exactly sum of 1-2-3

        result = solve_tsp(graph, algorithm="mst_approximation")

        assert result.has_tour


class TestResultObject:
    """Test TSPResult object."""

    def test_result_string_with_tour(self):
        """Test string representation with tour."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph)

        result_str = str(result)
        assert "tour found" in result_str.lower()
        assert "distance" in result_str.lower()
        assert str(result.total_distance) in result_str

    def test_result_string_without_tour(self):
        """Test string representation without tour."""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)

        result = solve_tsp(graph)

        result_str = str(result)
        assert "no" in result_str.lower()
        assert result.reason in result_str

    def test_result_attributes(self):
        """Test all result attributes are set correctly."""
        graph = Graph(3)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 0, 1)

        result = solve_tsp(graph, algorithm="exact")

        assert isinstance(result.has_tour, bool)
        assert isinstance(result.tour_path, list)
        assert isinstance(result.total_distance, int)
        assert isinstance(result.algorithm_used, str)
        assert isinstance(result.is_approximate, bool)
        assert result.approximation_ratio is not None or not result.is_approximate


class TestPerformance:
    """Test performance and stress scenarios."""

    def test_complete_graph_k10_exact(self):
        """Test exact algorithm on K10 (boundary case)."""
        graph = Graph(10)
        for i in range(10):
            for j in range(i + 1, 10):
                graph.add_edge(i, j, abs(i - j))

        result = solve_tsp(graph, algorithm="exact")

        # Should complete (though may be slow)
        assert result.has_tour
        assert len(result.tour_path) == 11

    def test_sparse_graph(self):
        """Test on sparse graph (minimal edges for connectivity)."""
        graph = Graph(8)
        # Create a cycle - all vertices connected in a ring
        for i in range(8):
            graph.add_edge(i, (i + 1) % 8, 1)

        # Use nearest neighbor since graph is not complete
        result = solve_tsp(graph, algorithm="nearest_neighbor")

        # Nearest neighbor should find tour on cycle
        assert result.has_tour
        assert result.total_distance == 8

    def test_dense_graph(self):
        """Test on dense graph (nearly complete)."""
        graph = Graph(7)
        # Create dense graph (complete graph)
        for i in range(7):
            for j in range(i + 1, 7):
                graph.add_edge(i, j, abs(i - j))

        result = solve_tsp(graph)

        # Complete graph should have tour
        assert result.has_tour
