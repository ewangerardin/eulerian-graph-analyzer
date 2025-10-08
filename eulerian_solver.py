"""
Eulerian path and circuit detection and solver using Hierholzer's algorithm.

This module provides functions to detect Eulerian paths/circuits and find the actual
route through the graph using an efficient stack-based implementation.
"""

from typing import List, Optional, Tuple, Dict
from collections import deque
import numpy as np
from graph import Graph


class EulerianResult:
    """
    Result container for Eulerian analysis.

    Attributes:
        has_circuit (bool): True if Eulerian circuit exists
        has_path (bool): True if Eulerian path exists
        path (List[int]): The Eulerian path/circuit if it exists, empty otherwise
        start_vertex (Optional[int]): Starting vertex for the path
        reason (str): Explanation of the result
    """

    def __init__(
        self,
        has_circuit: bool,
        has_path: bool,
        path: List[int],
        start_vertex: Optional[int] = None,
        reason: str = ""
    ):
        self.has_circuit = has_circuit
        self.has_path = has_path
        self.path = path
        self.start_vertex = start_vertex
        self.reason = reason

    def __str__(self) -> str:
        """String representation of the result."""
        if self.has_circuit:
            return f"Eulerian Circuit exists: {' -> '.join(map(str, self.path))}"
        elif self.has_path:
            return f"Eulerian Path exists (start: {self.start_vertex}): {' -> '.join(map(str, self.path))}"
        else:
            return f"No Eulerian path or circuit: {self.reason}"


class EulerianSolver:
    """
    Solver for finding Eulerian paths and circuits in graphs.

    Uses Hierholzer's algorithm for efficient path construction in O(E) time.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the solver with a graph.

        Args:
            graph (Graph): The graph to analyze
        """
        self.graph = graph
        # Cache for degree calculations
        self._degree_cache: Optional[Dict[int, int]] = None
        self._in_degree_cache: Optional[Dict[int, int]] = None
        self._out_degree_cache: Optional[Dict[int, int]] = None

    def analyze(self, circuit_only: bool = False) -> EulerianResult:
        """
        Analyze the graph for Eulerian properties and find the path/circuit.

        Time complexity: O(E) where E is the number of edges

        Args:
            circuit_only (bool): If True, only search for Eulerian circuits (path must return to start)

        Returns:
            EulerianResult: Complete analysis result with path if it exists
        """
        # Check if graph is connected
        if not self.graph.is_connected():
            return EulerianResult(
                has_circuit=False,
                has_path=False,
                path=[],
                reason="Graph is not connected"
            )

        # Check for edges
        if self.graph.get_edge_count() == 0:
            return EulerianResult(
                has_circuit=False,
                has_path=False,
                path=[],
                reason="Graph has no edges"
            )

        # Calculate degrees for all vertices
        self._compute_degree_cache()

        # Check Eulerian properties based on graph type
        if self.graph.directed:
            return self._analyze_directed(circuit_only)
        else:
            return self._analyze_undirected(circuit_only)

    def _analyze_undirected(self, circuit_only: bool = False) -> EulerianResult:
        """
        Analyze undirected graph for Eulerian properties.

        Eulerian Circuit: All vertices with edges have even degree
        Eulerian Path: Exactly 2 vertices have odd degree

        Args:
            circuit_only (bool): If True, only search for Eulerian circuits

        Returns:
            EulerianResult: Analysis result
        """
        odd_degree_vertices = []

        for v in range(self.graph.num_vertices):
            degree = self._degree_cache[v]
            if degree > 0 and degree % 2 == 1:
                odd_degree_vertices.append(v)

        if len(odd_degree_vertices) == 0:
            # Eulerian circuit exists
            # Start from any vertex with edges
            start_vertex = self._find_start_vertex()
            path = self._hierholzer_undirected(start_vertex)

            return EulerianResult(
                has_circuit=True,
                has_path=True,
                path=path,
                start_vertex=start_vertex,
                reason="All vertices have even degree - Eulerian circuit exists"
            )

        elif len(odd_degree_vertices) == 2:
            # Eulerian path exists (but not circuit)
            if circuit_only:
                # Circuit-only mode: reject paths that don't form circuits
                return EulerianResult(
                    has_circuit=False,
                    has_path=False,
                    path=[],
                    reason=f"Circuit-only mode: Graph has 2 vertices with odd degree ({odd_degree_vertices[0]}, {odd_degree_vertices[1]}) - only Eulerian path exists, not a circuit"
                )
            else:
                start_vertex = odd_degree_vertices[0]
                path = self._hierholzer_undirected(start_vertex)

                return EulerianResult(
                    has_circuit=False,
                    has_path=True,
                    path=path,
                    start_vertex=start_vertex,
                    reason=f"Exactly 2 vertices with odd degree ({odd_degree_vertices[0]}, {odd_degree_vertices[1]}) - Eulerian path exists"
                )

        else:
            return EulerianResult(
                has_circuit=False,
                has_path=False,
                path=[],
                reason=f"Graph has {len(odd_degree_vertices)} vertices with odd degree - no Eulerian path or circuit"
            )

    def _analyze_directed(self, circuit_only: bool = False) -> EulerianResult:
        """
        Analyze directed graph for Eulerian properties.

        Eulerian Circuit: in-degree = out-degree for all vertices
        Eulerian Path: Exactly one vertex with out-degree - in-degree = 1 (start)
                      and one vertex with in-degree - out-degree = 1 (end)
                      All others have in-degree = out-degree

        Args:
            circuit_only (bool): If True, only search for Eulerian circuits

        Returns:
            EulerianResult: Analysis result
        """
        start_vertices = []  # out-degree - in-degree = 1
        end_vertices = []    # in-degree - out-degree = 1
        imbalanced = []      # Vertices with other imbalances

        for v in range(self.graph.num_vertices):
            in_deg = self._in_degree_cache[v]
            out_deg = self._out_degree_cache[v]
            diff = out_deg - in_deg

            if diff == 1:
                start_vertices.append(v)
            elif diff == -1:
                end_vertices.append(v)
            elif diff != 0:
                imbalanced.append(v)

        # Check for Eulerian circuit
        if len(start_vertices) == 0 and len(end_vertices) == 0 and len(imbalanced) == 0:
            # All vertices have in-degree = out-degree
            start_vertex = self._find_start_vertex()
            path = self._hierholzer_directed(start_vertex)

            return EulerianResult(
                has_circuit=True,
                has_path=True,
                path=path,
                start_vertex=start_vertex,
                reason="All vertices have in-degree = out-degree - Eulerian circuit exists"
            )

        # Check for Eulerian path
        elif len(start_vertices) == 1 and len(end_vertices) == 1 and len(imbalanced) == 0:
            if circuit_only:
                # Circuit-only mode: reject paths that don't form circuits
                return EulerianResult(
                    has_circuit=False,
                    has_path=False,
                    path=[],
                    reason=f"Circuit-only mode: Graph has start vertex ({start_vertices[0]}) and end vertex ({end_vertices[0]}) - only Eulerian path exists, not a circuit"
                )
            else:
                start_vertex = start_vertices[0]
                path = self._hierholzer_directed(start_vertex)

                return EulerianResult(
                    has_circuit=False,
                    has_path=True,
                    path=path,
                    start_vertex=start_vertex,
                    reason=f"One start vertex ({start_vertex}) and one end vertex ({end_vertices[0]}) - Eulerian path exists"
                )

        else:
            return EulerianResult(
                has_circuit=False,
                has_path=False,
                path=[],
                reason=f"Degree imbalance: {len(start_vertices)} start vertices, {len(end_vertices)} end vertices, {len(imbalanced)} imbalanced - no Eulerian path or circuit"
            )

    def _hierholzer_undirected(self, start_vertex: int) -> List[int]:
        """
        Find Eulerian path/circuit using Hierholzer's algorithm for undirected graphs.

        Time complexity: O(E) where E is the number of edges

        Args:
            start_vertex (int): Starting vertex

        Returns:
            List[int]: The Eulerian path/circuit as a sequence of vertices
        """
        # Create a working copy of the adjacency matrix
        matrix = self.graph.get_adjacency_matrix()
        current_path = [start_vertex]
        circuit = []

        current_vertex = start_vertex

        while current_path:
            # Find if current vertex has any remaining edges
            has_edge = False

            for next_vertex in range(self.graph.num_vertices):
                if matrix[current_vertex][next_vertex] > 0:
                    has_edge = True

                    # Add next vertex to current path
                    current_path.append(next_vertex)

                    # Remove edge (both directions for undirected)
                    matrix[current_vertex][next_vertex] = 0
                    matrix[next_vertex][current_vertex] = 0

                    # Move to next vertex
                    current_vertex = next_vertex
                    break

            if not has_edge:
                # No more edges from current vertex, add to circuit
                circuit.append(current_vertex)

                # Backtrack
                if current_path:
                    current_path.pop()
                    if current_path:
                        current_vertex = current_path[-1]

        # Reverse circuit to get correct order
        circuit.reverse()

        return circuit

    def _hierholzer_directed(self, start_vertex: int) -> List[int]:
        """
        Find Eulerian path/circuit using Hierholzer's algorithm for directed graphs.

        Supports multi-edges: decrements edge multiplicity instead of removing completely.

        Time complexity: O(E) where E is the number of edges

        Args:
            start_vertex (int): Starting vertex

        Returns:
            List[int]: The Eulerian path/circuit as a sequence of vertices
        """
        # Create a working copy of the adjacency matrix
        matrix = self.graph.get_adjacency_matrix()
        current_path = [start_vertex]
        circuit = []

        current_vertex = start_vertex

        while current_path:
            # Find if current vertex has any remaining outgoing edges
            has_edge = False

            for next_vertex in range(self.graph.num_vertices):
                if matrix[current_vertex][next_vertex] > 0:
                    has_edge = True

                    # Add next vertex to current path
                    current_path.append(next_vertex)

                    # Decrement edge count (handles multi-edges)
                    matrix[current_vertex][next_vertex] -= 1

                    # Move to next vertex
                    current_vertex = next_vertex
                    break

            if not has_edge:
                # No more edges from current vertex, add to circuit
                circuit.append(current_vertex)

                # Backtrack
                if current_path:
                    current_path.pop()
                    if current_path:
                        current_vertex = current_path[-1]

        # Reverse circuit to get correct order
        circuit.reverse()

        return circuit

    def _compute_degree_cache(self) -> None:
        """
        Compute and cache degree information for all vertices.

        Time complexity: O(V^2) where V is the number of vertices
        """
        self._degree_cache = {}
        self._in_degree_cache = {}
        self._out_degree_cache = {}

        for v in range(self.graph.num_vertices):
            self._degree_cache[v] = self.graph.get_degree(v)
            self._in_degree_cache[v] = self.graph.get_in_degree(v)
            self._out_degree_cache[v] = self.graph.get_out_degree(v)

    def _find_start_vertex(self) -> int:
        """
        Find a suitable starting vertex (any vertex with edges).

        Returns:
            int: Starting vertex index
        """
        for v in range(self.graph.num_vertices):
            if self._degree_cache[v] > 0:
                return v
        return 0

    def has_eulerian_circuit(self) -> bool:
        """
        Check if the graph has an Eulerian circuit.

        Returns:
            bool: True if Eulerian circuit exists
        """
        result = self.analyze()
        return result.has_circuit

    def has_eulerian_path(self) -> bool:
        """
        Check if the graph has an Eulerian path.

        Returns:
            bool: True if Eulerian path exists (includes circuits)
        """
        result = self.analyze()
        return result.has_path

    def find_eulerian_path(self) -> Optional[List[int]]:
        """
        Find the Eulerian path/circuit if it exists.

        Returns:
            Optional[List[int]]: The path as a list of vertices, or None if no path exists
        """
        result = self.analyze()
        return result.path if result.has_path else None


def solve_eulerian(graph: Graph) -> EulerianResult:
    """
    Convenience function to analyze a graph for Eulerian properties.

    Args:
        graph (Graph): The graph to analyze

    Returns:
        EulerianResult: Complete analysis result
    """
    solver = EulerianSolver(graph)
    return solver.analyze()
