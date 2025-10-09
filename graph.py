"""
Graph data structure implementation using adjacency matrix representation.

This module provides a Graph class that supports both directed and undirected graphs
with efficient O(1) edge lookup time using an adjacency matrix.
"""

from typing import List, Set, Optional, Tuple
import numpy as np
from collections import deque


class Graph:
    """
    Graph implementation using adjacency matrix representation.

    Attributes:
        num_vertices (int): Number of vertices in the graph
        directed (bool): True if graph is directed, False if undirected
        adjacency_matrix (np.ndarray): 2D array representing edge connections
    """

    def __init__(self, num_vertices: int, directed: bool = False):
        """
        Initialize a graph with specified number of vertices.

        Args:
            num_vertices (int): Number of vertices in the graph (must be positive)
            directed (bool): Whether the graph is directed (default: False)

        Raises:
            ValueError: If num_vertices is not positive
        """
        if num_vertices <= 0:
            raise ValueError("Number of vertices must be positive")

        self.num_vertices = num_vertices
        self.directed = directed
        # Initialize adjacency matrix with zeros (no edges)
        self.adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    def add_edge(self, from_vertex: int, to_vertex: int, weight: int = 1) -> None:
        """
        Add an edge between two vertices.

        For undirected graphs, adds edges in both directions.
        Time complexity: O(1)

        Args:
            from_vertex (int): Starting vertex (0-indexed)
            to_vertex (int): Ending vertex (0-indexed)
            weight (int): Edge weight (default: 1)

        Raises:
            ValueError: If vertex indices are invalid or weight is non-positive
        """
        self._validate_vertex(from_vertex)
        self._validate_vertex(to_vertex)

        if weight <= 0:
            raise ValueError("Edge weight must be positive")

        # Prevent self-loops in undirected graphs
        if not self.directed and from_vertex == to_vertex:
            raise ValueError("Self-loops are not allowed in undirected graphs")

        # Add edge in adjacency matrix
        self.adjacency_matrix[from_vertex][to_vertex] = weight

        # For undirected graphs, add edge in both directions
        if not self.directed:
            self.adjacency_matrix[to_vertex][from_vertex] = weight

    def remove_edge(self, from_vertex: int, to_vertex: int) -> None:
        """
        Remove an edge between two vertices.

        For undirected graphs, removes edges in both directions.
        Time complexity: O(1)

        Args:
            from_vertex (int): Starting vertex (0-indexed)
            to_vertex (int): Ending vertex (0-indexed)

        Raises:
            ValueError: If vertex indices are invalid
        """
        self._validate_vertex(from_vertex)
        self._validate_vertex(to_vertex)

        self.adjacency_matrix[from_vertex][to_vertex] = 0

        if not self.directed:
            self.adjacency_matrix[to_vertex][from_vertex] = 0

    def has_edge(self, from_vertex: int, to_vertex: int) -> bool:
        """
        Check if an edge exists between two vertices.

        Time complexity: O(1)

        Args:
            from_vertex (int): Starting vertex (0-indexed)
            to_vertex (int): Ending vertex (0-indexed)

        Returns:
            bool: True if edge exists, False otherwise
        """
        self._validate_vertex(from_vertex)
        self._validate_vertex(to_vertex)

        return self.adjacency_matrix[from_vertex][to_vertex] > 0

    def get_neighbors(self, vertex: int) -> List[int]:
        """
        Get all neighbors of a vertex.

        Time complexity: O(V) where V is number of vertices

        Args:
            vertex (int): Vertex index (0-indexed)

        Returns:
            List[int]: List of neighboring vertex indices
        """
        self._validate_vertex(vertex)

        neighbors = []
        for i in range(self.num_vertices):
            if self.adjacency_matrix[vertex][i] > 0:
                neighbors.append(i)

        return neighbors

    def get_degree(self, vertex: int) -> int:
        """
        Get the degree of a vertex.

        For undirected graphs: total number of edges connected to vertex
        For directed graphs: out-degree (sum of outgoing edge multiplicities)

        Time complexity: O(V)

        Args:
            vertex (int): Vertex index (0-indexed)

        Returns:
            int: Degree of the vertex
        """
        self._validate_vertex(vertex)

        # For directed graphs with multi-edges, sum the row values
        # For undirected graphs, count non-zero entries (still binary)
        if self.directed:
            degree = np.sum(self.adjacency_matrix[vertex])
        else:
            degree = np.count_nonzero(self.adjacency_matrix[vertex])

        return degree

    def get_in_degree(self, vertex: int) -> int:
        """
        Get the in-degree of a vertex (for directed graphs).

        For directed graphs with multi-edges, sums the edge multiplicities.

        Time complexity: O(V)

        Args:
            vertex (int): Vertex index (0-indexed)

        Returns:
            int: In-degree of the vertex
        """
        self._validate_vertex(vertex)

        # For directed graphs with multi-edges, sum the column values
        if self.directed:
            in_degree = np.sum(self.adjacency_matrix[:, vertex])
        else:
            in_degree = np.count_nonzero(self.adjacency_matrix[:, vertex])

        return in_degree

    def get_out_degree(self, vertex: int) -> int:
        """
        Get the out-degree of a vertex (for directed graphs).

        Time complexity: O(V)

        Args:
            vertex (int): Vertex index (0-indexed)

        Returns:
            int: Out-degree of the vertex
        """
        return self.get_degree(vertex)

    def is_connected(self) -> bool:
        """
        Check if the graph is connected.

        For undirected graphs: all vertices are reachable from any vertex
        For directed graphs: underlying undirected graph is connected

        Uses DFS traversal. Time complexity: O(V^2) due to adjacency matrix

        Returns:
            bool: True if graph is connected, False otherwise
        """
        # Empty graph is considered disconnected
        if self.num_vertices == 0:
            return False

        # Find a vertex with non-zero degree to start DFS
        start_vertex = -1
        for i in range(self.num_vertices):
            if self.get_degree(i) > 0 or self.get_in_degree(i) > 0:
                start_vertex = i
                break

        # If no edges exist, graph is disconnected (unless it's a single vertex)
        if start_vertex == -1:
            return self.num_vertices == 1

        # Perform DFS to check connectivity
        visited = self._dfs_connected(start_vertex)

        # Check if all vertices with edges are visited
        for i in range(self.num_vertices):
            # Vertex should be visited if it has any edges
            has_edges = (self.get_degree(i) > 0 or self.get_in_degree(i) > 0)
            if has_edges and not visited[i]:
                return False

        return True

    def _dfs_connected(self, start_vertex: int) -> List[bool]:
        """
        Perform DFS to find all reachable vertices (treating graph as undirected).

        Args:
            start_vertex (int): Starting vertex for DFS

        Returns:
            List[bool]: Boolean array indicating which vertices were visited
        """
        visited = [False] * self.num_vertices
        stack = [start_vertex]

        while stack:
            vertex = stack.pop()

            if visited[vertex]:
                continue

            visited[vertex] = True

            # Add all neighbors (treat as undirected for connectivity check)
            for i in range(self.num_vertices):
                if not visited[i]:
                    # Check both directions for connectivity
                    if self.adjacency_matrix[vertex][i] > 0 or \
                       self.adjacency_matrix[i][vertex] > 0:
                        stack.append(i)

        return visited

    def get_edge_count(self) -> int:
        """
        Get the total number of edges in the graph.

        For directed graphs with multi-edges, counts each edge according to its multiplicity.

        Time complexity: O(V^2)

        Returns:
            int: Number of edges
        """
        if self.directed:
            # For directed graphs, sum all values (handles multi-edges)
            return int(np.sum(self.adjacency_matrix))
        else:
            # For undirected graphs, count upper triangle only (avoid double counting)
            return np.count_nonzero(np.triu(self.adjacency_matrix))

    def get_edge_weight(self, from_vertex: int, to_vertex: int) -> int:
        """
        Get the weight of an edge between two vertices.

        Time complexity: O(1)

        Args:
            from_vertex (int): Starting vertex (0-indexed)
            to_vertex (int): Ending vertex (0-indexed)

        Returns:
            int: Edge weight (0 if no edge exists)

        Raises:
            ValueError: If vertex indices are invalid
        """
        self._validate_vertex(from_vertex)
        self._validate_vertex(to_vertex)

        return int(self.adjacency_matrix[from_vertex][to_vertex])

    def _validate_vertex(self, vertex: int) -> None:
        """
        Validate that a vertex index is within valid range.

        Args:
            vertex (int): Vertex index to validate

        Raises:
            ValueError: If vertex index is out of range
        """
        if vertex < 0 or vertex >= self.num_vertices:
            raise ValueError(
                f"Vertex index {vertex} out of range [0, {self.num_vertices - 1}]"
            )

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get a copy of the adjacency matrix.

        Returns:
            np.ndarray: Copy of the adjacency matrix
        """
        return self.adjacency_matrix.copy()

    def set_adjacency_matrix(self, matrix: np.ndarray) -> None:
        """
        Set the adjacency matrix directly.

        For undirected graphs, only binary values (0 or 1) are allowed.
        For directed graphs, positive integers are allowed (for multi-edges).

        Args:
            matrix (np.ndarray): New adjacency matrix

        Raises:
            ValueError: If matrix dimensions don't match or matrix is invalid
        """
        if matrix.shape != (self.num_vertices, self.num_vertices):
            raise ValueError(
                f"Matrix dimensions {matrix.shape} don't match graph size "
                f"({self.num_vertices}, {self.num_vertices})"
            )

        # Validate matrix values
        if np.any(matrix < 0):
            raise ValueError("Matrix cannot contain negative values")

        # For undirected graphs, ensure binary values and symmetry
        if not self.directed:
            # Check for binary values (0 or 1 only)
            if np.any((matrix != 0) & (matrix != 1)):
                raise ValueError("Undirected graphs must use binary values (0 or 1) only")

            if not np.allclose(matrix, matrix.T):
                raise ValueError("Adjacency matrix must be symmetric for undirected graphs")

        self.adjacency_matrix = matrix.astype(int)

    def __str__(self) -> str:
        """
        String representation of the graph.

        Returns:
            str: String showing graph type and adjacency matrix
        """
        graph_type = "Directed" if self.directed else "Undirected"
        return f"{graph_type} Graph with {self.num_vertices} vertices:\n{self.adjacency_matrix}"

    def __repr__(self) -> str:
        """
        Representation of the graph for debugging.

        Returns:
            str: String representation
        """
        return f"Graph(num_vertices={self.num_vertices}, directed={self.directed})"
