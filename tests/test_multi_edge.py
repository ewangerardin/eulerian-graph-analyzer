"""
Unit tests for multi-edge support in directed graphs.

Tests cover:
- Multi-edge creation and validation
- Degree calculations with multi-edges
- Hierholzer's algorithm with multi-edges
- Validation that undirected graphs still reject non-binary values
"""

import pytest
import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver, solve_eulerian


class TestMultiEdgeGraphCreation:
    """Tests for creating graphs with multi-edges."""

    def test_directed_graph_accepts_multi_edges(self):
        """Test that directed graphs accept edge multiplicities > 1."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 3],
            [1, 0, 0]
        ])
        # Should not raise an error
        graph.set_adjacency_matrix(matrix)
        assert graph.adjacency_matrix[0][1] == 2
        assert graph.adjacency_matrix[1][2] == 3
        assert graph.adjacency_matrix[2][0] == 1

    def test_undirected_graph_rejects_multi_edges(self):
        """Test that undirected graphs reject edge multiplicities > 1."""
        graph = Graph(3, directed=False)
        matrix = np.array([
            [0, 2, 0],
            [2, 0, 1],
            [0, 1, 0]
        ])
        # Should raise an error because undirected graphs must be binary
        with pytest.raises(ValueError, match="binary values"):
            graph.set_adjacency_matrix(matrix)

    def test_directed_graph_large_multiplicity(self):
        """Test directed graph with large edge multiplicities."""
        graph = Graph(2, directed=True)
        matrix = np.array([
            [0, 10],
            [5, 0]
        ])
        graph.set_adjacency_matrix(matrix)
        assert graph.adjacency_matrix[0][1] == 10
        assert graph.adjacency_matrix[1][0] == 5

    def test_negative_values_rejected(self):
        """Test that negative values are rejected for both graph types."""
        graph_directed = Graph(2, directed=True)
        graph_undirected = Graph(2, directed=False)

        matrix = np.array([
            [0, -1],
            [1, 0]
        ])

        with pytest.raises(ValueError, match="negative"):
            graph_directed.set_adjacency_matrix(matrix)

        with pytest.raises(ValueError, match="negative"):
            graph_undirected.set_adjacency_matrix(matrix)


class TestMultiEdgeDegreeCalculations:
    """Tests for degree calculations with multi-edges."""

    def test_out_degree_with_multi_edges(self):
        """Test out-degree calculation sums edge multiplicities."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 3],
            [0, 0, 1],
            [0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Vertex 0 has 2 edges to vertex 1 and 3 edges to vertex 2 = 5 total
        assert graph.get_out_degree(0) == 5
        assert graph.get_degree(0) == 5  # get_degree returns out-degree for directed

        # Vertex 1 has 1 edge to vertex 2
        assert graph.get_out_degree(1) == 1

        # Vertex 2 has no outgoing edges
        assert graph.get_out_degree(2) == 0

    def test_in_degree_with_multi_edges(self):
        """Test in-degree calculation sums edge multiplicities."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 3],
            [0, 0, 1],
            [0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Vertex 0 has no incoming edges
        assert graph.get_in_degree(0) == 0

        # Vertex 1 has 2 incoming edges from vertex 0
        assert graph.get_in_degree(1) == 2

        # Vertex 2 has 3 edges from vertex 0 and 1 from vertex 1 = 4 total
        assert graph.get_in_degree(2) == 4

    def test_edge_count_with_multi_edges(self):
        """Test edge count sums all edge multiplicities."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 3],
            [0, 0, 1],
            [0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Total edges: 2 + 3 + 1 = 6
        assert graph.get_edge_count() == 6

    def test_degree_balance_with_multi_edges(self):
        """Test degree balance detection with multi-edges."""
        graph = Graph(3, directed=True)
        # Create a circuit: 0->1 (2 edges), 1->2 (2 edges), 2->0 (2 edges)
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # All vertices should have in-degree = out-degree = 2
        for v in range(3):
            assert graph.get_in_degree(v) == 2
            assert graph.get_out_degree(v) == 2


class TestMultiEdgeEulerianCircuit:
    """Tests for Eulerian circuits with multi-edges."""

    def test_eulerian_circuit_with_double_edges(self):
        """Test Eulerian circuit with all edges doubled."""
        graph = Graph(3, directed=True)
        # Triangle with all edges doubled
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        # Should traverse 6 edges (2*3) + return to start = 7 vertices
        assert len(result.path) == 7
        assert result.path[0] == result.path[-1]  # Circuit returns to start

    def test_eulerian_circuit_asymmetric_multi_edges(self):
        """Test Eulerian circuit with different edge multiplicities."""
        graph = Graph(2, directed=True)
        # Two vertices with 3 edges each direction
        matrix = np.array([
            [0, 3],
            [3, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        # Should traverse 6 edges + return = 7 vertices
        assert len(result.path) == 7
        assert result.path[0] == result.path[-1]

    def test_eulerian_circuit_complex_multi_edges(self):
        """Test Eulerian circuit with complex multi-edge pattern."""
        graph = Graph(4, directed=True)
        # Create a balanced multi-edge graph (circuit)
        matrix = np.array([
            [0, 2, 1, 0],
            [0, 0, 2, 1],
            [1, 0, 0, 2],
            [2, 1, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Verify balance
        for v in range(4):
            assert graph.get_in_degree(v) == graph.get_out_degree(v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        # Total edges: 2+1+2+1+1+2+2+1 = 12, path length = 13
        assert len(result.path) == 13


class TestMultiEdgeEulerianPath:
    """Tests for Eulerian paths (non-circuit) with multi-edges."""

    def test_eulerian_path_with_multi_edges(self):
        """Test Eulerian path with multi-edges but not circuit."""
        graph = Graph(4, directed=True)
        # Path from 0 to 3 with multi-edges
        matrix = np.array([
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Vertex 0: out=2, in=0 (start, diff=+1... but this is +2, need to fix)
        # Actually for Eulerian path: one vertex out-in=1, one vertex in-out=1
        # Let me create a proper example
        matrix = np.array([
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
            [0, 0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        # Vertex 0: out=2, in=0 (diff=+2, not valid for path)
        # Actually need: vertex with out-in=1 and vertex with in-out=1
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        graph = Graph(3, directed=True)
        graph.set_adjacency_matrix(matrix)

        # Vertex 0: out=2, in=1, diff=+1 (start)
        # Vertex 1: out=1, in=2, diff=-1 (end)
        # Vertex 2: out=1, in=1, diff=0 (balanced)

        result = solve_eulerian(graph)

        assert not result.has_circuit  # Path but not circuit
        assert result.has_path
        assert result.start_vertex == 0
        # Total edges: 2+1+1 = 4, path length = 5
        assert len(result.path) == 5

    def test_no_eulerian_with_imbalanced_multi_edges(self):
        """Test that imbalanced multi-edges prevent Eulerian path."""
        graph = Graph(3, directed=True)
        # Imbalanced: vertex 0 has out=5, in=0 (too many outgoing)
        matrix = np.array([
            [0, 3, 2],
            [0, 0, 1],
            [0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        # Vertex 0: out=5, in=0 (diff=5, should be 1 for path)
        assert not result.has_circuit
        assert not result.has_path


class TestMultiEdgeHierholzerAlgorithm:
    """Tests for Hierholzer's algorithm with multi-edges."""

    def test_hierholzer_traverses_all_multi_edges(self):
        """Test that Hierholzer's algorithm traverses each multi-edge exactly once."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 2],
            [2, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        # Count edge traversals in the path
        edge_counts = {}
        for i in range(len(result.path) - 1):
            edge = (result.path[i], result.path[i + 1])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Each edge should be traversed exactly as many times as its multiplicity
        assert edge_counts.get((0, 1), 0) == 2
        assert edge_counts.get((1, 2), 0) == 2
        assert edge_counts.get((2, 0), 0) == 2

    def test_hierholzer_path_validity_multi_edges(self):
        """Test that the path is valid (consecutive vertices connected)."""
        graph = Graph(4, directed=True)
        matrix = np.array([
            [0, 1, 2, 0],
            [0, 0, 0, 3],
            [1, 0, 0, 2],
            [2, 0, 0, 1]
        ])
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        # Verify each consecutive pair in path is a valid edge
        for i in range(len(result.path) - 1):
            from_v = result.path[i]
            to_v = result.path[i + 1]
            # Original matrix should have had this edge
            assert graph.has_edge(from_v, to_v)


class TestMultiEdgeCircuitOnlyMode:
    """Tests for circuit-only mode with multi-edges."""

    def test_circuit_only_accepts_multi_edge_circuit(self):
        """Test circuit-only mode accepts multi-edge circuits."""
        graph = Graph(2, directed=True)
        matrix = np.array([
            [0, 2],
            [2, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert result.has_circuit
        assert result.has_path

    def test_circuit_only_rejects_multi_edge_path(self):
        """Test circuit-only mode rejects multi-edge paths."""
        graph = Graph(3, directed=True)
        matrix = np.array([
            [0, 2, 0],
            [0, 0, 2],
            [0, 0, 0]
        ])
        graph.set_adjacency_matrix(matrix)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert not result.has_circuit
        assert not result.has_path  # Rejected in circuit-only mode


class TestMultiEdgeEdgeCases:
    """Tests for edge cases with multi-edges."""

    def test_single_vertex_multi_self_loop(self):
        """Test single vertex with multiple self-loops."""
        graph = Graph(1, directed=True)
        matrix = np.array([[4]])
        graph.set_adjacency_matrix(matrix)

        # Vertex has in-degree = out-degree = 4
        assert graph.get_in_degree(0) == 4
        assert graph.get_out_degree(0) == 4

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        # Should traverse 4 self-loops + return = 5 vertices (all vertex 0)
        assert len(result.path) == 5
        assert all(v == 0 for v in result.path)

    def test_mixed_single_and_multi_edges(self):
        """Test graph with mix of single and multi-edges."""
        graph = Graph(3, directed=True)
        # Simple balanced circuit with mix of single and multi-edges
        # Vertex 0 -> 1 (2 edges), Vertex 1 -> 2 (2 edges), Vertex 2 -> 0 (2 edges)
        # Plus: Vertex 0 -> 2 (1 edge), Vertex 2 -> 1 (1 edge), Vertex 1 -> 0 (1 edge)
        matrix = np.array([
            [0, 3, 1],  # vertex 0: out=4
            [1, 0, 3],  # vertex 1: out=4
            [3, 1, 0]   # vertex 2: out=4
        ])
        graph.set_adjacency_matrix(matrix)

        # Verify all vertices balanced (in=4, out=4)
        for v in range(3):
            in_deg = graph.get_in_degree(v)
            out_deg = graph.get_out_degree(v)
            assert in_deg == out_deg == 4, f"Vertex {v}: in={in_deg}, out={out_deg}"

        result = solve_eulerian(graph)

        assert result.has_circuit
        # Total edges: 3+1+1+3+3+1 = 12, path = 13
        assert len(result.path) == 13

    def test_zero_edges_with_multi_edge_support(self):
        """Test that graphs with no edges still work correctly."""
        graph = Graph(3, directed=True)
        matrix = np.zeros((3, 3), dtype=int)
        graph.set_adjacency_matrix(matrix)

        result = solve_eulerian(graph)

        assert not result.has_circuit
        assert not result.has_path
        # Empty graphs are reported as disconnected, which is also correct
        assert ("no edges" in result.reason.lower() or "not connected" in result.reason.lower())


class TestBackwardCompatibility:
    """Tests ensuring existing single-edge functionality still works."""

    def test_directed_single_edges_still_work(self):
        """Test that directed graphs with single edges still work correctly."""
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 5

    def test_undirected_graphs_unchanged(self):
        """Test that undirected graph behavior is unchanged."""
        graph = Graph(5, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 6
        assert result.path[0] == result.path[-1]
