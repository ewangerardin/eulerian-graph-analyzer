"""
Unit tests for Graph class.

Tests cover graph creation, edge operations, degree calculations,
and connectivity checks.
"""

import pytest
import numpy as np
from graph import Graph


class TestGraphCreation:
    """Tests for graph initialization."""

    def test_graph_creation_undirected(self):
        """Test creating an undirected graph."""
        graph = Graph(5, directed=False)
        assert graph.num_vertices == 5
        assert not graph.directed
        assert graph.adjacency_matrix.shape == (5, 5)
        assert np.all(graph.adjacency_matrix == 0)

    def test_graph_creation_directed(self):
        """Test creating a directed graph."""
        graph = Graph(4, directed=True)
        assert graph.num_vertices == 4
        assert graph.directed
        assert graph.adjacency_matrix.shape == (4, 4)

    def test_graph_creation_invalid_vertices(self):
        """Test that creating graph with invalid vertices raises error."""
        with pytest.raises(ValueError, match="Number of vertices must be positive"):
            Graph(0, directed=False)

        with pytest.raises(ValueError, match="Number of vertices must be positive"):
            Graph(-5, directed=False)


class TestEdgeOperations:
    """Tests for adding and removing edges."""

    def test_add_edge_undirected(self):
        """Test adding edge in undirected graph adds both directions."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)

        assert graph.adjacency_matrix[0][1] == 1
        assert graph.adjacency_matrix[1][0] == 1  # Symmetric
        assert graph.has_edge(0, 1)
        assert graph.has_edge(1, 0)

    def test_add_edge_directed(self):
        """Test adding edge in directed graph adds one direction."""
        graph = Graph(4, directed=True)
        graph.add_edge(0, 1)

        assert graph.adjacency_matrix[0][1] == 1
        assert graph.adjacency_matrix[1][0] == 0  # Not symmetric
        assert graph.has_edge(0, 1)
        assert not graph.has_edge(1, 0)

    def test_add_edge_weighted(self):
        """Test adding weighted edges."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1, weight=5)

        assert graph.adjacency_matrix[0][1] == 5
        assert graph.adjacency_matrix[1][0] == 5

    def test_add_edge_invalid_weight(self):
        """Test that invalid edge weights raise errors."""
        graph = Graph(3, directed=False)

        with pytest.raises(ValueError, match="Edge weight must be positive"):
            graph.add_edge(0, 1, weight=0)

        with pytest.raises(ValueError, match="Edge weight must be positive"):
            graph.add_edge(0, 1, weight=-1)

    def test_add_edge_self_loop_undirected(self):
        """Test that self-loops are prevented in undirected graphs."""
        graph = Graph(3, directed=False)

        with pytest.raises(ValueError, match="Self-loops are not allowed"):
            graph.add_edge(0, 0)

    def test_add_edge_self_loop_directed(self):
        """Test that self-loops are allowed in directed graphs."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 0)  # Should not raise error

        assert graph.adjacency_matrix[0][0] == 1

    def test_remove_edge_undirected(self):
        """Test removing edge from undirected graph."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1)
        graph.remove_edge(0, 1)

        assert graph.adjacency_matrix[0][1] == 0
        assert graph.adjacency_matrix[1][0] == 0
        assert not graph.has_edge(0, 1)

    def test_remove_edge_directed(self):
        """Test removing edge from directed graph."""
        graph = Graph(3, directed=True)
        graph.add_edge(0, 1)
        graph.remove_edge(0, 1)

        assert graph.adjacency_matrix[0][1] == 0
        assert not graph.has_edge(0, 1)

    def test_invalid_vertex_handling(self):
        """Test that invalid vertex indices raise errors."""
        graph = Graph(3, directed=False)

        with pytest.raises(ValueError, match="Vertex index .* out of range"):
            graph.add_edge(-1, 0)

        with pytest.raises(ValueError, match="Vertex index .* out of range"):
            graph.add_edge(0, 5)

        with pytest.raises(ValueError, match="Vertex index .* out of range"):
            graph.has_edge(10, 0)


class TestDegreeCalculations:
    """Tests for degree calculation methods."""

    def test_degree_calculation_undirected(self):
        """Test degree calculation in undirected graph."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)

        assert graph.get_degree(0) == 2  # Connected to 1, 2
        assert graph.get_degree(1) == 2  # Connected to 0, 3
        assert graph.get_degree(2) == 1  # Connected to 0
        assert graph.get_degree(3) == 1  # Connected to 1

    def test_degree_calculation_directed(self):
        """Test out-degree calculation in directed graph."""
        graph = Graph(4, directed=True)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)

        assert graph.get_out_degree(0) == 2  # Edges to 1, 2
        assert graph.get_out_degree(1) == 1  # Edge to 2
        assert graph.get_out_degree(2) == 0  # No outgoing edges

    def test_in_degree_calculation(self):
        """Test in-degree calculation in directed graph."""
        graph = Graph(4, directed=True)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)

        assert graph.get_in_degree(0) == 0  # No incoming edges
        assert graph.get_in_degree(1) == 1  # Edge from 0
        assert graph.get_in_degree(2) == 2  # Edges from 0, 1


class TestConnectivity:
    """Tests for connectivity checking."""

    def test_connectivity_check_connected(self):
        """Test connectivity check for connected graph."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        assert graph.is_connected()

    def test_connectivity_check_disconnected(self):
        """Test connectivity check for disconnected graph."""
        graph = Graph(5, directed=False)
        # Two separate components: {0, 1} and {2, 3}
        graph.add_edge(0, 1)
        graph.add_edge(2, 3)

        assert not graph.is_connected()

    def test_connectivity_empty_graph(self):
        """Test connectivity for graph with no edges."""
        graph = Graph(3, directed=False)
        assert not graph.is_connected()

    def test_connectivity_single_vertex_no_edges(self):
        """Test connectivity for single vertex with no edges."""
        graph = Graph(1, directed=False)
        assert graph.is_connected()


class TestMatrixOperations:
    """Tests for adjacency matrix operations."""

    def test_get_adjacency_matrix_returns_copy(self):
        """Test that get_adjacency_matrix returns a copy."""
        graph = Graph(3, directed=False)
        graph.add_edge(0, 1)

        matrix = graph.get_adjacency_matrix()
        matrix[0][1] = 99  # Modify copy

        # Original should be unchanged
        assert graph.adjacency_matrix[0][1] == 1

    def test_set_adjacency_matrix_valid(self):
        """Test setting adjacency matrix with valid data."""
        graph = Graph(3, directed=False)
        new_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])

        graph.set_adjacency_matrix(new_matrix)
        assert np.array_equal(graph.adjacency_matrix, new_matrix)

    def test_set_adjacency_matrix_invalid_dimensions(self):
        """Test that invalid matrix dimensions raise error."""
        graph = Graph(3, directed=False)
        invalid_matrix = np.array([
            [0, 1],
            [1, 0]
        ])

        with pytest.raises(ValueError, match="Matrix dimensions .* don't match"):
            graph.set_adjacency_matrix(invalid_matrix)

    def test_set_adjacency_matrix_asymmetric_undirected(self):
        """Test that asymmetric matrix raises error for undirected graph."""
        graph = Graph(3, directed=False)
        asymmetric_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])

        with pytest.raises(ValueError, match="must be symmetric"):
            graph.set_adjacency_matrix(asymmetric_matrix)

    def test_matrix_symmetry_undirected(self):
        """Test that undirected graph maintains symmetric matrix."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        # Check symmetry
        assert np.array_equal(graph.adjacency_matrix, graph.adjacency_matrix.T)


class TestEdgeCount:
    """Tests for edge counting."""

    def test_get_edge_count_undirected(self):
        """Test edge count in undirected graph."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)

        assert graph.get_edge_count() == 3

    def test_get_edge_count_directed(self):
        """Test edge count in directed graph."""
        graph = Graph(4, directed=True)
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

        assert graph.get_edge_count() == 3

    def test_get_edge_count_empty(self):
        """Test edge count for empty graph."""
        graph = Graph(5, directed=False)
        assert graph.get_edge_count() == 0


class TestGraphRepresentation:
    """Tests for string representation."""

    def test_str_representation(self):
        """Test string representation of graph."""
        graph = Graph(2, directed=False)
        graph.add_edge(0, 1)

        str_repr = str(graph)
        assert "Undirected Graph" in str_repr
        assert "2 vertices" in str_repr

    def test_repr_representation(self):
        """Test repr of graph."""
        graph = Graph(3, directed=True)
        repr_str = repr(graph)

        assert "Graph" in repr_str
        assert "num_vertices=3" in repr_str
        assert "directed=True" in repr_str


class TestNeighbors:
    """Tests for getting neighbors."""

    def test_get_neighbors_undirected(self):
        """Test getting neighbors in undirected graph."""
        graph = Graph(4, directed=False)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)

        neighbors_0 = graph.get_neighbors(0)
        assert set(neighbors_0) == {1, 2}

        neighbors_1 = graph.get_neighbors(1)
        assert set(neighbors_1) == {0, 3}

    def test_get_neighbors_directed(self):
        """Test getting neighbors in directed graph (outgoing edges only)."""
        graph = Graph(4, directed=True)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(2, 0)

        neighbors_0 = graph.get_neighbors(0)
        assert set(neighbors_0) == {1, 2}

        neighbors_2 = graph.get_neighbors(2)
        assert set(neighbors_2) == {0}
