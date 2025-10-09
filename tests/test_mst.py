"""
Unit tests for MST (Minimum Spanning Tree) solver.

Tests cover Kruskal's algorithm, Union-Find data structure,
and various graph configurations.
"""

import pytest
import numpy as np
from graph import Graph
from mst_solver import MSTSolver, MSTResult, UnionFind, solve_mst


class TestUnionFind:
    """Tests for Union-Find data structure."""

    def test_union_find_initialization(self):
        """Test Union-Find initialization."""
        uf = UnionFind(5)
        assert len(uf.parent) == 5
        assert len(uf.rank) == 5

        # Initially, each element is its own parent
        for i in range(5):
            assert uf.find(i) == i

    def test_union_find_union(self):
        """Test union operation."""
        uf = UnionFind(5)

        # Union 0 and 1
        assert uf.union(0, 1) == True
        assert uf.find(0) == uf.find(1)

        # Union already connected - should return False
        assert uf.union(0, 1) == False

    def test_union_find_connected(self):
        """Test connectivity check."""
        uf = UnionFind(5)

        # Initially disconnected
        assert not uf.connected(0, 1)

        # After union, connected
        uf.union(0, 1)
        assert uf.connected(0, 1)

        # Transitive connectivity
        uf.union(1, 2)
        assert uf.connected(0, 2)

    def test_union_find_path_compression(self):
        """Test path compression optimization."""
        uf = UnionFind(5)

        # Create a chain: 0 -> 1 -> 2
        uf.union(0, 1)
        uf.union(1, 2)

        # Find should compress path
        root = uf.find(0)
        assert root == uf.find(2)

        # After compression, parent should point directly to root
        uf.find(0)  # Trigger compression
        assert uf.parent[0] == root

    def test_union_find_union_by_rank(self):
        """Test union by rank optimization."""
        uf = UnionFind(4)

        # Create trees of different sizes
        uf.union(0, 1)  # Rank 0 increases to 1
        uf.union(2, 3)  # Rank 2 increases to 1

        # Union two trees of equal rank
        result = uf.union(0, 2)
        assert result == True

        # All should be connected
        assert uf.connected(0, 3)
        assert uf.connected(1, 2)


class TestMSTBasic:
    """Tests for basic MST functionality."""

    def test_mst_simple_triangle(self):
        """Test MST on a simple triangle graph."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 0, weight=3)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 2  # Tree has V-1 edges
        assert result.total_weight == 3  # 1 + 2
        assert result.num_components == 1

        # Verify edges (should be 0-1 and 1-2)
        edge_set = set([(u, v) for u, v, w in result.mst_edges])
        assert (0, 1) in edge_set or (1, 0) in edge_set
        assert (1, 2) in edge_set or (2, 1) in edge_set

    def test_mst_square_graph(self):
        """Test MST on a square with diagonal."""
        graph = Graph(4, directed=False)
        # Square edges
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 3, weight=3)
        graph.add_edge(3, 0, weight=4)
        # Diagonal
        graph.add_edge(0, 2, weight=5)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 3
        assert result.total_weight == 6  # 1 + 2 + 3 (optimal)

    def test_mst_complete_graph_k4(self):
        """Test MST on complete graph K4."""
        graph = Graph(4, directed=False)

        # Create K4 with all edges weight 1
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j, weight=1)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 3
        assert result.total_weight == 3

    def test_mst_weighted_complete_graph(self):
        """Test MST on K5 with different weights."""
        graph = Graph(5, directed=False)

        # Add edges with increasing weights
        weight = 1
        for i in range(5):
            for j in range(i + 1, 5):
                graph.add_edge(i, j, weight=weight)
                weight += 1

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 4  # V-1 edges
        # Should pick the 4 smallest weights: 1, 2, 3, 4
        assert result.total_weight == 10


class TestMSTEdgeCases:
    """Tests for edge cases."""

    def test_mst_single_vertex(self):
        """Test MST on single vertex graph."""
        graph = Graph(1, directed=False)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 0
        assert result.total_weight == 0
        assert "Single vertex" in result.reason

    def test_mst_no_edges(self):
        """Test MST on graph with no edges."""
        graph = Graph(5, directed=False)

        result = solve_mst(graph)

        assert result.has_mst == False
        assert len(result.mst_edges) == 0
        assert result.num_components == 5

    def test_mst_two_vertices_one_edge(self):
        """Test MST on minimal connected graph."""
        graph = Graph(2, directed=False)
        graph.add_edge(0, 1, weight=5)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 1
        assert result.total_weight == 5
        assert result.mst_edges[0] == (0, 1, 5)

    def test_mst_equal_weights(self):
        """Test MST with all equal edge weights."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1, weight=10)
        graph.add_edge(1, 2, weight=10)
        graph.add_edge(2, 3, weight=10)
        graph.add_edge(3, 0, weight=10)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 3
        assert result.total_weight == 30


class TestMSTDisconnected:
    """Tests for disconnected graphs."""

    def test_mst_two_components(self):
        """Test MST on disconnected graph (2 components)."""
        graph = Graph(6, directed=False)

        # Component 1: Triangle {0, 1, 2}
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 0, weight=3)

        # Component 2: Triangle {3, 4, 5}
        graph.add_edge(3, 4, weight=4)
        graph.add_edge(4, 5, weight=5)
        graph.add_edge(5, 3, weight=6)

        result = solve_mst(graph)

        assert result.has_mst == False
        assert result.num_components == 2
        # Should have MSF (Minimum Spanning Forest) with 4 edges total
        assert len(result.mst_edges) == 4  # 2 trees, each with 2 edges
        assert result.total_weight == 12  # (1+2) + (4+5)

    def test_mst_three_components(self):
        """Test MST on graph with 3 disconnected components."""
        graph = Graph(7, directed=False)

        # Component 1: Edge {0, 1}
        graph.add_edge(0, 1, weight=1)

        # Component 2: Triangle {2, 3, 4}
        graph.add_edge(2, 3, weight=2)
        graph.add_edge(3, 4, weight=3)
        graph.add_edge(4, 2, weight=4)

        # Component 3: Edge {5, 6}
        graph.add_edge(5, 6, weight=5)

        result = solve_mst(graph)

        assert result.has_mst == False
        assert result.num_components == 3
        assert len(result.mst_edges) == 4  # 1 + 2 + 1 edges
        assert result.total_weight == 11  # 1 + (2+3) + 5

    def test_mst_isolated_vertices(self):
        """Test MST with isolated vertices."""
        graph = Graph(5, directed=False)

        # Only connect vertices 0, 1, 2
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        # Vertices 3 and 4 are isolated

        result = solve_mst(graph)

        assert result.has_mst == False
        assert result.num_components == 3  # {0,1,2}, {3}, {4}
        assert len(result.mst_edges) == 2


class TestMSTLargeGraphs:
    """Tests for larger graphs."""

    def test_mst_path_graph_10_vertices(self):
        """Test MST on path graph with 10 vertices."""
        graph = Graph(10, directed=False)

        # Create path: 0-1-2-3-4-5-6-7-8-9
        for i in range(9):
            graph.add_edge(i, i + 1, weight=i + 1)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 9
        # Total: 1+2+3+4+5+6+7+8+9 = 45
        assert result.total_weight == 45

    def test_mst_cycle_graph_8_vertices(self):
        """Test MST on cycle graph with 8 vertices."""
        graph = Graph(8, directed=False)

        # Create cycle with different weights
        for i in range(8):
            next_vertex = (i + 1) % 8
            graph.add_edge(i, next_vertex, weight=i + 1)

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 7  # Cycle minus one edge
        # Should skip the heaviest edge (weight 8)
        # Total: 1+2+3+4+5+6+7 = 28
        assert result.total_weight == 28

    def test_mst_complete_graph_k7(self):
        """Test MST on K7 with varied weights."""
        graph = Graph(7, directed=False)

        # Add edges with sequential weights
        weight = 1
        for i in range(7):
            for j in range(i + 1, 7):
                graph.add_edge(i, j, weight=weight)
                weight += 1

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 6  # V-1 edges
        # Should pick 6 smallest weights: 1+2+3+4+5+6 = 21
        assert result.total_weight == 21


class TestMSTProperties:
    """Tests for MST properties and validation."""

    def test_mst_acyclic(self):
        """Test that MST is acyclic (no cycles)."""
        graph = Graph(5, directed=False)

        # Create graph with cycle
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 3, weight=3)
        graph.add_edge(3, 4, weight=4)
        graph.add_edge(4, 0, weight=5)  # Creates cycle

        result = solve_mst(graph)

        # MST should have exactly V-1 edges (no cycles)
        assert len(result.mst_edges) == 4

        # Verify it's a tree using Union-Find
        uf = UnionFind(5)
        for u, v, w in result.mst_edges:
            # Adding edge should never create cycle
            assert uf.union(u, v) == True

    def test_mst_connected_result(self):
        """Test that MST connects all vertices (for connected graph)."""
        graph = Graph(6, directed=False)

        # Create connected graph
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 3, weight=3)
        graph.add_edge(3, 4, weight=4)
        graph.add_edge(4, 5, weight=5)
        graph.add_edge(5, 0, weight=6)

        result = solve_mst(graph)

        # Verify all vertices are in one component
        uf = UnionFind(6)
        for u, v, w in result.mst_edges:
            uf.union(u, v)

        # All vertices should be in same component
        root = uf.find(0)
        for i in range(1, 6):
            assert uf.find(i) == root

    def test_mst_minimum_weight(self):
        """Test that MST has minimum total weight."""
        graph = Graph(4, directed=False)

        # Create graph where greedy selection matters
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(0, 2, weight=10)  # Expensive shortcut
        graph.add_edge(1, 3, weight=10)  # Expensive shortcut

        result = solve_mst(graph)

        # Optimal MST: 0-1, 1-2, 2-3 with total weight 3
        assert result.total_weight == 3


class TestMSTIntegration:
    """Integration tests with Graph class."""

    def test_mst_with_graph_methods(self):
        """Test MST solver integrates correctly with Graph methods."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1, weight=5)
        graph.add_edge(1, 2, weight=3)
        graph.add_edge(2, 3, weight=7)
        graph.add_edge(3, 0, weight=2)

        solver = MSTSolver(graph)
        result = solver.solve()

        # Verify using graph methods
        assert graph.is_connected() == True
        assert graph.get_edge_count() == 4

        # MST should have 3 edges
        assert len(result.mst_edges) == 3

    def test_mst_solver_directed_graph_raises_error(self):
        """Test that MST solver rejects directed graphs."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)

        with pytest.raises(ValueError, match="only works with undirected graphs"):
            MSTSolver(graph)

    def test_mst_result_string_representation(self):
        """Test MSTResult string representation."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1, weight=5)
        graph.add_edge(1, 2, weight=3)

        result = solve_mst(graph)
        result_str = str(result)

        assert "MST exists" in result_str
        assert "total weight: 8" in result_str

    def test_mst_get_helper_methods(self):
        """Test MST solver helper methods."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1, weight=1)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 3, weight=3)

        solver = MSTSolver(graph)

        # Test get_mst_weight
        weight = solver.get_mst_weight()
        assert weight == 6

        # Test get_mst_edges
        edges = solver.get_mst_edges()
        assert len(edges) == 3
        assert all(isinstance(edge, tuple) and len(edge) == 3 for edge in edges)


class TestMSTStressTest:
    """Stress tests for performance."""

    def test_mst_large_complete_graph(self):
        """Test MST on larger complete graph."""
        # K10 has 45 edges
        graph = Graph(10, directed=False)

        weight = 1
        for i in range(10):
            for j in range(i + 1, 10):
                graph.add_edge(i, j, weight=weight)
                weight += 1

        result = solve_mst(graph)

        assert result.has_mst == True
        assert len(result.mst_edges) == 9  # V-1 edges
        # Should pick 9 smallest: 1+2+3+4+5+6+7+8+9 = 45
        assert result.total_weight == 45
