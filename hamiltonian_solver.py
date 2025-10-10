"""
Hamiltonian Path and Circuit solver with exact and heuristic algorithms.

This module provides algorithms for finding Hamiltonian paths and circuits:
- Exact: Backtracking with pruning (O(n!))
- Heuristics: Theorem-based detection (Ore's, Dirac's), TSP reduction
"""

from typing import List, Optional, Set, Tuple
import numpy as np
from graph import Graph
import time


class HamiltonianResult:
    """
    Result container for Hamiltonian path/circuit analysis.

    Attributes:
        has_path (bool): True if Hamiltonian path exists
        has_circuit (bool): True if Hamiltonian circuit exists
        path (List[int]): Hamiltonian path if found (empty if not found)
        algorithm_used (str): Name of algorithm used
        reason (str): Explanation of the result
        timeout (bool): True if search timed out
    """

    def __init__(
        self,
        has_path: bool,
        has_circuit: bool,
        path: List[int],
        algorithm_used: str,
        reason: str = "",
        timeout: bool = False
    ):
        self.has_path = has_path
        self.has_circuit = has_circuit
        self.path = path
        self.algorithm_used = algorithm_used
        self.reason = reason
        self.timeout = timeout

    def __str__(self) -> str:
        """String representation of the result."""
        if self.has_circuit:
            path_str = " -> ".join(map(str, self.path)) + f" -> {self.path[0]}"
            return (f"Hamiltonian circuit found using {self.algorithm_used}\n"
                   f"Path: {path_str}\n"
                   f"Reason: {self.reason}")
        elif self.has_path:
            path_str = " -> ".join(map(str, self.path))
            return (f"Hamiltonian path found using {self.algorithm_used}\n"
                   f"Path: {path_str}\n"
                   f"Reason: {self.reason}")
        else:
            timeout_str = " (timed out)" if self.timeout else ""
            return f"No Hamiltonian path/circuit{timeout_str}: {self.reason}"


class HamiltonianSolver:
    """
    Solver for Hamiltonian path and circuit problems.

    Uses multiple strategies:
    1. Theorem-based fast checks (Ore's, Dirac's)
    2. Backtracking with pruning
    3. TSP reduction (all edges weight 1)
    """

    def __init__(self, graph: Graph, timeout: float = 5.0):
        """
        Initialize the Hamiltonian solver.

        Args:
            graph (Graph): The graph to analyze (directed or undirected)
            timeout (float): Maximum time in seconds for backtracking (default: 5.0)
        """
        self.graph = graph
        self.timeout = timeout
        self.start_time = 0

    def solve(self, circuit_only: bool = False, use_tsp_reduction: bool = False) -> HamiltonianResult:
        """
        Find Hamiltonian path or circuit.

        Args:
            circuit_only (bool): If True, only search for circuits (not paths)
            use_tsp_reduction (bool): If True, use TSP reduction for complete graphs

        Returns:
            HamiltonianResult: Analysis result
        """
        # Check for empty or trivial graphs
        if self.graph.num_vertices == 0:
            return HamiltonianResult(
                has_path=False,
                has_circuit=False,
                path=[],
                algorithm_used="trivial",
                reason="Graph has no vertices"
            )

        if self.graph.num_vertices == 1:
            return HamiltonianResult(
                has_path=True,
                has_circuit=True,
                path=[0],
                algorithm_used="trivial",
                reason="Single vertex graph"
            )

        if self.graph.num_vertices == 2:
            if self.graph.has_edge(0, 1):
                if self.graph.has_edge(1, 0) or not self.graph.directed:
                    return HamiltonianResult(
                        has_path=True,
                        has_circuit=True,
                        path=[0, 1],
                        algorithm_used="trivial",
                        reason="Two vertices with bidirectional edge"
                    )
                else:
                    return HamiltonianResult(
                        has_path=True,
                        has_circuit=False,
                        path=[0, 1],
                        algorithm_used="trivial",
                        reason="Two vertices with unidirectional edge"
                    )
            else:
                return HamiltonianResult(
                    has_path=False,
                    has_circuit=False,
                    path=[],
                    algorithm_used="trivial",
                    reason="Two vertices not connected"
                )

        # Try fast theorem-based checks for undirected graphs
        if not self.graph.directed:
            theorem_result = self._check_theorems()
            if theorem_result is not None:
                return theorem_result

        # Try TSP reduction for complete graphs
        if use_tsp_reduction and self._is_complete():
            return self._solve_via_tsp()

        # Use backtracking search
        return self._solve_backtracking(circuit_only)

    def _check_theorems(self) -> Optional[HamiltonianResult]:
        """
        Check if graph satisfies sufficient conditions for Hamiltonian circuit.

        Checks:
        - Dirac's theorem: If deg(v) ≥ n/2 for all v, then Hamiltonian circuit exists
        - Ore's theorem: If deg(u) + deg(v) ≥ n for all non-adjacent u,v, then circuit exists

        Returns:
            Optional[HamiltonianResult]: Result if theorem applies, None otherwise
        """
        n = self.graph.num_vertices

        # Dirac's theorem
        min_degree = min(self.graph.get_degree(v) for v in range(n))
        if min_degree >= n / 2:
            # Theorem guarantees circuit exists - use backtracking to find it
            result = self._solve_backtracking(circuit_only=True)
            if result.has_circuit:
                result.reason = f"Dirac's theorem satisfied (min degree {min_degree} ≥ {n/2})"
            return result

        # Ore's theorem
        ore_satisfied = True
        for i in range(n):
            for j in range(i + 1, n):
                if not self.graph.has_edge(i, j):
                    deg_sum = self.graph.get_degree(i) + self.graph.get_degree(j)
                    if deg_sum < n:
                        ore_satisfied = False
                        break
            if not ore_satisfied:
                break

        if ore_satisfied:
            result = self._solve_backtracking(circuit_only=True)
            if result.has_circuit:
                result.reason = "Ore's theorem satisfied (deg(u)+deg(v) ≥ n for all non-adjacent u,v)"
            return result

        return None

    def _is_complete(self) -> bool:
        """
        Check if graph is complete (all vertices connected).

        Returns:
            bool: True if complete graph
        """
        n = self.graph.num_vertices

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.graph.directed:
                        if not self.graph.has_edge(i, j):
                            return False
                    else:
                        if not self.graph.has_edge(i, j):
                            return False

        return True

    def _solve_via_tsp(self) -> HamiltonianResult:
        """
        Solve Hamiltonian using TSP reduction.

        For complete graphs, Hamiltonian circuit exists iff TSP tour exists.
        All edges have weight 1, so any tour is valid.

        Returns:
            HamiltonianResult: Result from TSP reduction
        """
        try:
            from tsp_solver import solve_tsp

            # Create TSP problem with all edges weight 1
            # (already set in original graph)
            tsp_result = solve_tsp(self.graph, algorithm="exact")

            if tsp_result.has_tour:
                # Remove duplicate end vertex from TSP tour
                path = tsp_result.tour_path[:-1]
                return HamiltonianResult(
                    has_path=True,
                    has_circuit=True,
                    path=path,
                    algorithm_used="TSP reduction",
                    reason="Complete graph - solved via TSP"
                )
            else:
                return HamiltonianResult(
                    has_path=False,
                    has_circuit=False,
                    path=[],
                    algorithm_used="TSP reduction",
                    reason="TSP found no tour"
                )

        except ImportError:
            # Fall back to backtracking if TSP solver not available
            return self._solve_backtracking(circuit_only=False)

    def _solve_backtracking(self, circuit_only: bool = False) -> HamiltonianResult:
        """
        Solve using backtracking with pruning.

        Args:
            circuit_only (bool): Only search for circuits

        Returns:
            HamiltonianResult: Result from backtracking search
        """
        self.start_time = time.time()
        n = self.graph.num_vertices

        # Try starting from each vertex for path search
        if not circuit_only:
            for start in range(n):
                if self._timed_out():
                    return HamiltonianResult(
                        has_path=False,
                        has_circuit=False,
                        path=[],
                        algorithm_used="backtracking (timed out)",
                        reason=f"Search timed out after {self.timeout}s",
                        timeout=True
                    )

                path = self._backtrack_path(start, [start], set([start]))
                if path is not None:
                    # Check if it's also a circuit
                    is_circuit = self._can_return_to_start(path)
                    return HamiltonianResult(
                        has_path=True,
                        has_circuit=is_circuit,
                        path=path,
                        algorithm_used="backtracking",
                        reason="Found via exhaustive backtracking search"
                    )

        # Try circuit search (only need to check from vertex 0 due to symmetry)
        path = self._backtrack_circuit(0, [0], set([0]))
        if path is not None:
            return HamiltonianResult(
                has_path=True,
                has_circuit=True,
                path=path,
                algorithm_used="backtracking",
                reason="Found via exhaustive backtracking search"
            )

        # No Hamiltonian path or circuit found
        timeout_msg = " (may exist but not found within timeout)" if self._timed_out() else ""
        return HamiltonianResult(
            has_path=False,
            has_circuit=False,
            path=[],
            algorithm_used="backtracking",
            reason=f"No Hamiltonian path/circuit found{timeout_msg}",
            timeout=self._timed_out()
        )

    def _backtrack_path(self, current: int, path: List[int], visited: Set[int]) -> Optional[List[int]]:
        """
        Backtracking search for Hamiltonian path.

        Args:
            current (int): Current vertex
            path (List[int]): Current path
            visited (Set[int]): Set of visited vertices

        Returns:
            Optional[List[int]]: Hamiltonian path if found, None otherwise
        """
        if self._timed_out():
            return None

        # Base case: visited all vertices
        if len(path) == self.graph.num_vertices:
            return path

        # Try each neighbor
        neighbors = self._get_valid_neighbors(current, visited)

        for next_vertex in neighbors:
            if next_vertex not in visited:
                # Make move
                path.append(next_vertex)
                visited.add(next_vertex)

                # Recurse
                result = self._backtrack_path(next_vertex, path, visited)
                if result is not None:
                    return result

                # Undo move
                path.pop()
                visited.remove(next_vertex)

        return None

    def _backtrack_circuit(self, current: int, path: List[int], visited: Set[int]) -> Optional[List[int]]:
        """
        Backtracking search for Hamiltonian circuit.

        Args:
            current (int): Current vertex
            path (List[int]): Current path
            visited (Set[int]): Set of visited vertices

        Returns:
            Optional[List[int]]: Hamiltonian circuit if found, None otherwise
        """
        if self._timed_out():
            return None

        # Base case: visited all vertices
        if len(path) == self.graph.num_vertices:
            # Check if we can return to start
            if self._has_edge(current, path[0]):
                return path
            else:
                return None

        # Try each neighbor
        neighbors = self._get_valid_neighbors(current, visited)

        for next_vertex in neighbors:
            if next_vertex not in visited:
                # Make move
                path.append(next_vertex)
                visited.add(next_vertex)

                # Recurse
                result = self._backtrack_circuit(next_vertex, path, visited)
                if result is not None:
                    return result

                # Undo move
                path.pop()
                visited.remove(next_vertex)

        return None

    def _get_valid_neighbors(self, vertex: int, visited: Set[int]) -> List[int]:
        """
        Get list of valid unvisited neighbors.

        Args:
            vertex (int): Current vertex
            visited (Set[int]): Set of visited vertices

        Returns:
            List[int]: List of unvisited neighbors
        """
        neighbors = []

        for v in range(self.graph.num_vertices):
            if v not in visited and self._has_edge(vertex, v):
                neighbors.append(v)

        # Sort by degree (heuristic: visit low-degree vertices first)
        neighbors.sort(key=lambda v: self.graph.get_degree(v))

        return neighbors

    def _has_edge(self, u: int, v: int) -> bool:
        """
        Check if edge exists from u to v.

        Args:
            u (int): Source vertex
            v (int): Destination vertex

        Returns:
            bool: True if edge exists
        """
        return self.graph.has_edge(u, v)

    def _can_return_to_start(self, path: List[int]) -> bool:
        """
        Check if last vertex in path can return to first vertex.

        Args:
            path (List[int]): The path

        Returns:
            bool: True if can form circuit
        """
        if len(path) < 2:
            return False

        return self._has_edge(path[-1], path[0])

    def _timed_out(self) -> bool:
        """
        Check if search has exceeded timeout.

        Returns:
            bool: True if timed out
        """
        return time.time() - self.start_time > self.timeout


def solve_hamiltonian(
    graph: Graph,
    circuit_only: bool = False,
    use_tsp_reduction: bool = False,
    timeout: float = 5.0
) -> HamiltonianResult:
    """
    Convenience function to find Hamiltonian path/circuit.

    Args:
        graph (Graph): The graph to analyze
        circuit_only (bool): Only search for circuits (default: False)
        use_tsp_reduction (bool): Use TSP reduction for complete graphs (default: False)
        timeout (float): Maximum search time in seconds (default: 5.0)

    Returns:
        HamiltonianResult: Analysis result
    """
    solver = HamiltonianSolver(graph, timeout=timeout)
    return solver.solve(circuit_only=circuit_only, use_tsp_reduction=use_tsp_reduction)


def has_hamiltonian_path(graph: Graph, timeout: float = 5.0) -> bool:
    """
    Check if graph has a Hamiltonian path.

    Args:
        graph (Graph): The graph to check
        timeout (float): Maximum search time in seconds

    Returns:
        bool: True if Hamiltonian path exists
    """
    result = solve_hamiltonian(graph, timeout=timeout)
    return result.has_path


def has_hamiltonian_circuit(graph: Graph, timeout: float = 5.0) -> bool:
    """
    Check if graph has a Hamiltonian circuit.

    Args:
        graph (Graph): The graph to check
        timeout (float): Maximum search time in seconds

    Returns:
        bool: True if Hamiltonian circuit exists
    """
    result = solve_hamiltonian(graph, circuit_only=True, timeout=timeout)
    return result.has_circuit
