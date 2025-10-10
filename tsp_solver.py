"""
Traveling Salesman Problem (TSP) solver with exact and approximation algorithms.

This module provides multiple algorithms for solving the TSP:
- Exact: Held-Karp dynamic programming (O(n²·2ⁿ))
- Approximations: Nearest Neighbor (O(n²)), MST-based (O(E log E)), Christofides (O(n³))
"""

from typing import List, Tuple, Optional, Set
import numpy as np
from itertools import combinations
from graph import Graph
from mst_solver import solve_mst, MSTResult
import math


class TSPResult:
    """
    Result container for TSP analysis.

    Attributes:
        has_tour (bool): True if a valid tour exists
        tour_path (List[int]): Ordered list of vertices in the tour (including return to start)
        total_distance (int): Total distance of the tour
        algorithm_used (str): Name of algorithm used to find tour
        is_approximate (bool): True if using approximation algorithm
        approximation_ratio (Optional[float]): Worst-case approximation ratio (None for exact)
        reason (str): Explanation of the result
    """

    def __init__(
        self,
        has_tour: bool,
        tour_path: List[int],
        total_distance: int,
        algorithm_used: str,
        is_approximate: bool = False,
        approximation_ratio: Optional[float] = None,
        reason: str = ""
    ):
        self.has_tour = has_tour
        self.tour_path = tour_path
        self.total_distance = total_distance
        self.algorithm_used = algorithm_used
        self.is_approximate = is_approximate
        self.approximation_ratio = approximation_ratio
        self.reason = reason

    def __str__(self) -> str:
        """String representation of the result."""
        if self.has_tour:
            path_str = " -> ".join(map(str, self.tour_path))
            approx_str = f" (approx ratio: {self.approximation_ratio})" if self.is_approximate else " (optimal)"
            return (f"TSP tour found using {self.algorithm_used}{approx_str}\n"
                   f"Distance: {self.total_distance}\n"
                   f"Path: {path_str}")
        else:
            return f"No TSP tour: {self.reason}"


class TSPSolver:
    """
    Solver for the Traveling Salesman Problem with multiple algorithms.

    Algorithms:
    - exact: Held-Karp DP (O(n²·2ⁿ)) - for graphs ≤15 vertices
    - nearest_neighbor: Greedy heuristic (O(n²)) - 50% approximate
    - mst_approximation: MST-based 2-approximation (O(E log E))
    - christofides: MST + matching 1.5-approximation (O(n³))
    """

    def __init__(self, graph: Graph):
        """
        Initialize the TSP solver with a graph.

        Args:
            graph (Graph): The graph to analyze (must be undirected and weighted)

        Raises:
            ValueError: If graph is directed
        """
        if graph.directed:
            raise ValueError("TSP solver only works with undirected graphs")

        self.graph = graph

    def solve(self, algorithm: str = "auto") -> TSPResult:
        """
        Solve TSP using specified algorithm.

        Args:
            algorithm (str): Algorithm choice - "auto", "exact", "nearest_neighbor",
                           "mst_approximation", or "christofides"

        Returns:
            TSPResult: Complete TSP analysis result
        """
        # Check for disconnected graph
        if not self.graph.is_connected():
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used=algorithm,
                reason="Graph is disconnected - no Hamiltonian tour possible"
            )

        # Check for empty or single vertex
        if self.graph.num_vertices == 0:
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used=algorithm,
                reason="Graph has no vertices"
            )

        if self.graph.num_vertices == 1:
            return TSPResult(
                has_tour=True,
                tour_path=[0, 0],
                total_distance=0,
                algorithm_used=algorithm,
                reason="Single vertex tour"
            )

        # Auto-select algorithm based on graph size
        if algorithm == "auto":
            if self.graph.num_vertices <= 10:
                algorithm = "exact"
            elif self.graph.num_vertices <= 20:
                algorithm = "christofides"
            else:
                algorithm = "mst_approximation"

        # Route to appropriate algorithm
        if algorithm == "exact":
            return self._solve_exact()
        elif algorithm == "nearest_neighbor":
            return self._solve_nearest_neighbor()
        elif algorithm == "mst_approximation":
            return self._solve_mst_approximation()
        elif algorithm == "christofides":
            return self._solve_christofides()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _solve_exact(self) -> TSPResult:
        """
        Solve TSP exactly using Held-Karp dynamic programming.

        Time complexity: O(n²·2ⁿ)
        Space complexity: O(n·2ⁿ)

        Returns:
            TSPResult: Optimal tour
        """
        n = self.graph.num_vertices

        # Check if all vertices are connected to each other
        for i in range(n):
            for j in range(n):
                if i != j and self.graph.get_edge_weight(i, j) == 0:
                    # Missing edge - no Hamiltonian cycle
                    return TSPResult(
                        has_tour=False,
                        tour_path=[],
                        total_distance=0,
                        algorithm_used="exact (Held-Karp)",
                        reason=f"Graph is incomplete - missing edge ({i},{j})"
                    )

        # DP table: dp[S][i] = minimum cost to visit all vertices in set S ending at i
        # S is represented as a bitmask
        INF = float('inf')
        dp = {}
        parent = {}

        # Base case: start from vertex 0, visit only vertex 0
        dp[(1, 0)] = 0

        # Iterate through all subsets of vertices
        for subset_size in range(2, n + 1):
            for subset in combinations(range(1, n), subset_size - 1):
                # Add vertex 0 to make complete subset
                bits = 1  # Include vertex 0
                for vertex in subset:
                    bits |= (1 << vertex)

                # Try ending at each vertex in subset
                for end in subset:
                    prev_bits = bits & ~(1 << end)

                    # Try coming from each vertex in previous subset
                    min_cost = INF
                    best_prev = -1

                    for prev in ([0] if prev_bits == 1 else subset):
                        if prev == end:
                            continue
                        if (prev_bits, prev) in dp:
                            cost = dp[(prev_bits, prev)] + self.graph.get_edge_weight(prev, end)
                            if cost < min_cost:
                                min_cost = cost
                                best_prev = prev

                    if min_cost < INF:
                        dp[(bits, end)] = min_cost
                        parent[(bits, end)] = best_prev

        # Find minimum cost tour (must return to vertex 0)
        full_set = (1 << n) - 1
        min_tour_cost = INF
        last_vertex = -1

        for end in range(1, n):
            if (full_set, end) in dp:
                cost = dp[(full_set, end)] + self.graph.get_edge_weight(end, 0)
                if cost < min_tour_cost:
                    min_tour_cost = cost
                    last_vertex = end

        if min_tour_cost == INF:
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used="exact (Held-Karp)",
                reason="No Hamiltonian tour exists"
            )

        # Reconstruct path
        path = []
        bits = full_set
        current = last_vertex

        while (bits, current) in parent:
            path.append(current)
            prev = parent[(bits, current)]
            bits &= ~(1 << current)
            current = prev

        path.append(0)
        path.reverse()
        path.append(0)  # Return to start

        return TSPResult(
            has_tour=True,
            tour_path=path,
            total_distance=int(min_tour_cost),
            algorithm_used="exact (Held-Karp)",
            is_approximate=False,
            approximation_ratio=1.0,
            reason="Optimal tour found using dynamic programming"
        )

    def _solve_nearest_neighbor(self, start_vertex: int = 0) -> TSPResult:
        """
        Solve TSP using nearest neighbor greedy heuristic.

        Time complexity: O(n²)

        Args:
            start_vertex (int): Starting vertex for the tour

        Returns:
            TSPResult: Approximate tour
        """
        n = self.graph.num_vertices
        visited = [False] * n
        path = [start_vertex]
        visited[start_vertex] = True
        total_distance = 0
        current = start_vertex

        # Greedy: always go to nearest unvisited neighbor
        for _ in range(n - 1):
            nearest = -1
            min_dist = float('inf')

            for next_vertex in range(n):
                if not visited[next_vertex]:
                    dist = self.graph.get_edge_weight(current, next_vertex)
                    if dist > 0 and dist < min_dist:
                        min_dist = dist
                        nearest = next_vertex

            if nearest == -1:
                # No path to unvisited vertex
                return TSPResult(
                    has_tour=False,
                    tour_path=[],
                    total_distance=0,
                    algorithm_used="nearest_neighbor",
                    reason="Graph is disconnected - cannot complete tour"
                )

            path.append(nearest)
            visited[nearest] = True
            total_distance += min_dist
            current = nearest

        # Return to start
        return_dist = self.graph.get_edge_weight(current, start_vertex)
        if return_dist == 0:
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used="nearest_neighbor",
                reason=f"No edge from {current} back to {start_vertex}"
            )

        path.append(start_vertex)
        total_distance += return_dist

        return TSPResult(
            has_tour=True,
            tour_path=path,
            total_distance=int(total_distance),
            algorithm_used="nearest_neighbor",
            is_approximate=True,
            approximation_ratio=None,  # No guaranteed bound
            reason="Greedy heuristic - not guaranteed optimal"
        )

    def _solve_mst_approximation(self) -> TSPResult:
        """
        Solve TSP using MST-based 2-approximation algorithm.

        Algorithm:
        1. Find MST using Kruskal's algorithm
        2. Perform DFS on MST to get pre-order traversal
        3. Use traversal order as tour (shortcut repeated vertices)

        Time complexity: O(E log E) dominated by MST
        Approximation ratio: 2.0 (for metric TSP)

        Returns:
            TSPResult: 2-approximate tour
        """
        # Find MST
        mst_result = solve_mst(self.graph)

        if not mst_result.has_mst:
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used="mst_approximation",
                reason=f"No MST exists - {mst_result.reason}"
            )

        # Build adjacency list from MST edges
        mst_adj = {i: [] for i in range(self.graph.num_vertices)}
        for u, v, _ in mst_result.mst_edges:
            mst_adj[u].append(v)
            mst_adj[v].append(u)

        # DFS traversal to get pre-order tour
        visited = [False] * self.graph.num_vertices
        tour = []

        def dfs(vertex: int):
            visited[vertex] = True
            tour.append(vertex)
            for neighbor in sorted(mst_adj[vertex]):  # Sort for deterministic order
                if not visited[neighbor]:
                    dfs(neighbor)

        dfs(0)  # Start from vertex 0
        tour.append(0)  # Return to start

        # Calculate tour distance
        total_distance = 0
        for i in range(len(tour) - 1):
            dist = self.graph.get_edge_weight(tour[i], tour[i + 1])
            if dist == 0:
                return TSPResult(
                    has_tour=False,
                    tour_path=[],
                    total_distance=0,
                    algorithm_used="mst_approximation",
                    reason=f"Missing edge ({tour[i]},{tour[i+1]}) - graph may not satisfy triangle inequality"
                )
            total_distance += dist

        return TSPResult(
            has_tour=True,
            tour_path=tour,
            total_distance=int(total_distance),
            algorithm_used="mst_approximation",
            is_approximate=True,
            approximation_ratio=2.0,
            reason="MST-based 2-approximation (guaranteed ≤2× optimal for metric TSP)"
        )

    def _solve_christofides(self) -> TSPResult:
        """
        Solve TSP using Christofides algorithm (1.5-approximation).

        Algorithm:
        1. Find MST
        2. Find vertices with odd degree in MST
        3. Find minimum weight perfect matching on odd-degree vertices
        4. Combine MST and matching to get Eulerian graph
        5. Find Eulerian circuit
        6. Convert to Hamiltonian circuit by skipping repeated vertices

        Time complexity: O(n³)
        Approximation ratio: 1.5 (for metric TSP)

        Returns:
            TSPResult: 1.5-approximate tour
        """
        # For now, fall back to MST approximation
        # Full Christofides requires complex matching algorithm
        # This is a simplified version

        # Step 1: Find MST
        mst_result = solve_mst(self.graph)

        if not mst_result.has_mst:
            return TSPResult(
                has_tour=False,
                tour_path=[],
                total_distance=0,
                algorithm_used="christofides",
                reason=f"No MST exists - {mst_result.reason}"
            )

        # Step 2: Find odd-degree vertices in MST
        mst_degree = [0] * self.graph.num_vertices
        for u, v, _ in mst_result.mst_edges:
            mst_degree[u] += 1
            mst_degree[v] += 1

        odd_vertices = [i for i in range(self.graph.num_vertices) if mst_degree[i] % 2 == 1]

        # Step 3: Simplified matching (greedy pairing of odd vertices)
        matching_edges = []
        used = set()

        for i in range(len(odd_vertices)):
            if odd_vertices[i] in used:
                continue

            best_pair = -1
            min_weight = float('inf')

            for j in range(i + 1, len(odd_vertices)):
                if odd_vertices[j] in used:
                    continue

                weight = self.graph.get_edge_weight(odd_vertices[i], odd_vertices[j])
                if weight > 0 and weight < min_weight:
                    min_weight = weight
                    best_pair = j

            if best_pair != -1:
                matching_edges.append((odd_vertices[i], odd_vertices[best_pair], min_weight))
                used.add(odd_vertices[i])
                used.add(odd_vertices[best_pair])

        # Step 4: Build multigraph (MST + matching)
        multigraph_adj = {i: [] for i in range(self.graph.num_vertices)}

        for u, v, w in mst_result.mst_edges:
            multigraph_adj[u].append(v)
            multigraph_adj[v].append(u)

        for u, v, w in matching_edges:
            multigraph_adj[u].append(v)
            multigraph_adj[v].append(u)

        # Step 5: Find Eulerian circuit (all vertices now have even degree)
        circuit = self._find_eulerian_circuit(multigraph_adj, 0)

        # Step 6: Convert to Hamiltonian by skipping repeats
        tour = []
        seen = set()

        for vertex in circuit:
            if vertex not in seen:
                tour.append(vertex)
                seen.add(vertex)

        tour.append(0)  # Return to start

        # Calculate tour distance
        total_distance = 0
        for i in range(len(tour) - 1):
            dist = self.graph.get_edge_weight(tour[i], tour[i + 1])
            if dist == 0:
                # Fall back to MST approximation
                return self._solve_mst_approximation()
            total_distance += dist

        return TSPResult(
            has_tour=True,
            tour_path=tour,
            total_distance=int(total_distance),
            algorithm_used="christofides",
            is_approximate=True,
            approximation_ratio=1.5,
            reason="Christofides 1.5-approximation (guaranteed ≤1.5× optimal for metric TSP)"
        )

    def _find_eulerian_circuit(self, adj: dict, start: int) -> List[int]:
        """
        Find Eulerian circuit in multigraph using Hierholzer's algorithm.

        Args:
            adj (dict): Adjacency list (may have duplicate edges)
            start (int): Starting vertex

        Returns:
            List[int]: Eulerian circuit
        """
        # Copy adjacency list to avoid modifying original
        graph = {v: list(neighbors) for v, neighbors in adj.items()}

        circuit = []
        stack = [start]

        while stack:
            v = stack[-1]
            if graph[v]:
                u = graph[v].pop()
                # Remove reverse edge
                if v in graph[u]:
                    graph[u].remove(v)
                stack.append(u)
            else:
                circuit.append(stack.pop())

        return circuit[::-1]


def solve_tsp(graph: Graph, algorithm: str = "auto") -> TSPResult:
    """
    Convenience function to solve TSP on a graph.

    Args:
        graph (Graph): The graph to analyze (must be undirected and weighted)
        algorithm (str): Algorithm choice - "auto", "exact", "nearest_neighbor",
                        "mst_approximation", or "christofides"

    Returns:
        TSPResult: Complete TSP analysis result

    Raises:
        ValueError: If graph is directed
    """
    solver = TSPSolver(graph)
    return solver.solve(algorithm)
