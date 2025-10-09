"""
Chinese Postman Problem (CPP) solver.

This module provides functions to solve the Chinese Postman Problem - finding the
shortest closed walk that visits every edge at least once in a weighted graph.

For graphs that are not Eulerian, the algorithm finds the minimum-cost set of edges
to duplicate to make the graph Eulerian, then finds an Eulerian circuit.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver, EulerianResult
import itertools


@dataclass
class ChinesePostmanResult:
    """
    Result container for Chinese Postman Problem solution.

    Attributes:
        has_solution (bool): True if a solution exists
        optimal_route (Optional[List[int]]): The optimal route visiting all edges
        total_cost (float): Total cost of the optimal route
        added_edges (List[Tuple[int, int, float]]): Edges that were duplicated
        original_cost (float): Cost of original edges
        reason (str): Explanation of the result
    """
    has_solution: bool
    optimal_route: Optional[List[int]]
    total_cost: float
    added_edges: List[Tuple[int, int, float]]
    original_cost: float
    reason: str

    def __str__(self) -> str:
        """String representation of the result."""
        if self.has_solution:
            route_str = ' -> '.join(map(str, self.optimal_route)) if self.optimal_route else "N/A"
            added_str = f", added {len(self.added_edges)} edges" if self.added_edges else ""
            return f"CPP Solution (cost: {self.total_cost}){added_str}: {route_str}"
        else:
            return f"No CPP solution: {self.reason}"


class ChinesePostmanSolver:
    """
    Solver for the Chinese Postman Problem.

    The Chinese Postman Problem finds the shortest closed walk that visits every edge
    at least once. For Eulerian graphs, this is simply the Eulerian circuit. For
    non-Eulerian graphs, we find the minimum-cost set of edges to duplicate.

    Algorithm:
    1. Check if graph is connected (required)
    2. Check if graph is Eulerian (if yes, return Eulerian circuit)
    3. Find all odd-degree vertices (undirected) or unbalanced vertices (directed)
    4. Compute shortest paths between all pairs using Floyd-Warshall
    5. Find minimum-cost perfect matching on odd/unbalanced vertices
    6. Augment graph with matching edges
    7. Find Eulerian circuit on augmented graph

    Complexity: O(V^3) for Floyd-Warshall + O(n! / (2^(n/2) * (n/2)!)) for matching
                where n = number of odd-degree vertices (typically small)
    """

    def __init__(self, graph: Graph):
        """
        Initialize the solver with a graph.

        Args:
            graph (Graph): The weighted graph to analyze
        """
        self.graph = graph

    def solve(self) -> ChinesePostmanResult:
        """
        Solve the Chinese Postman Problem.

        Returns:
            ChinesePostmanResult: Complete solution with optimal route and cost
        """
        # 1. Check connectivity
        if not self.graph.is_connected():
            return ChinesePostmanResult(
                has_solution=False,
                optimal_route=None,
                total_cost=0.0,
                added_edges=[],
                original_cost=0.0,
                reason="Graph is not connected"
            )

        # Check for edges
        if self.graph.get_edge_count() == 0:
            return ChinesePostmanResult(
                has_solution=False,
                optimal_route=None,
                total_cost=0.0,
                added_edges=[],
                original_cost=0.0,
                reason="Graph has no edges"
            )

        # Calculate original cost (sum of all edge weights)
        original_cost = self._calculate_graph_cost()

        # 2. Check if already Eulerian
        eulerian_solver = EulerianSolver(self.graph)
        eulerian_result = eulerian_solver.analyze()

        if eulerian_result.has_circuit:
            # Graph is already Eulerian - optimal route is the Eulerian circuit
            return ChinesePostmanResult(
                has_solution=True,
                optimal_route=eulerian_result.path,
                total_cost=original_cost,
                added_edges=[],
                original_cost=original_cost,
                reason="Graph is already Eulerian - no edges need to be duplicated"
            )

        # 3. Find odd-degree vertices (undirected only for now)
        if self.graph.directed:
            return self._solve_directed()
        else:
            return self._solve_undirected(original_cost)

    def _solve_undirected(self, original_cost: float) -> ChinesePostmanResult:
        """
        Solve CPP for undirected graphs.

        Args:
            original_cost (float): Total cost of original edges

        Returns:
            ChinesePostmanResult: Solution for undirected graph
        """
        odd_vertices = self._find_odd_vertices()

        if len(odd_vertices) == 0:
            # Should have been caught by Eulerian check, but handle anyway
            eulerian_solver = EulerianSolver(self.graph)
            eulerian_result = eulerian_solver.analyze()
            return ChinesePostmanResult(
                has_solution=True,
                optimal_route=eulerian_result.path,
                total_cost=original_cost,
                added_edges=[],
                original_cost=original_cost,
                reason="Graph is Eulerian"
            )

        # 4. Compute shortest paths using Floyd-Warshall
        distances, next_vertex = self._floyd_warshall()

        # Check if all odd vertices are reachable from each other
        for i, v1 in enumerate(odd_vertices):
            for v2 in odd_vertices[i+1:]:
                if distances[v1][v2] == float('inf'):
                    return ChinesePostmanResult(
                        has_solution=False,
                        optimal_route=None,
                        total_cost=0.0,
                        added_edges=[],
                        original_cost=original_cost,
                        reason=f"No path between odd-degree vertices {v1} and {v2}"
                    )

        # 5. Find minimum-cost perfect matching
        matching = self._min_cost_matching(odd_vertices, distances)

        # 6. Augment graph with matching edges (add actual path edges)
        augmented_graph = self._create_augmented_graph(matching, distances, next_vertex)

        # 7. Find Eulerian circuit on augmented graph
        # CPP should always produce a circuit (closed walk)
        eulerian_solver = EulerianSolver(augmented_graph)
        eulerian_result = eulerian_solver.analyze(circuit_only=True)

        if not eulerian_result.has_circuit:
            return ChinesePostmanResult(
                has_solution=False,
                optimal_route=None,
                total_cost=0.0,
                added_edges=[],
                original_cost=original_cost,
                reason=f"Failed to create Eulerian circuit after augmentation: {eulerian_result.reason}"
            )

        # Calculate added edges and total cost
        added_edges = []
        added_cost = 0.0
        for u, v in matching:
            # Find the shortest path cost
            cost = distances[u][v]
            added_edges.append((u, v, cost))
            added_cost += cost

        total_cost = original_cost + added_cost

        return ChinesePostmanResult(
            has_solution=True,
            optimal_route=eulerian_result.path,
            total_cost=total_cost,
            added_edges=added_edges,
            original_cost=original_cost,
            reason=f"Found optimal solution by duplicating {len(added_edges)} edge(s)"
        )

    def _solve_directed(self) -> ChinesePostmanResult:
        """
        Solve CPP for directed graphs.

        For directed graphs, we need to balance in-degree and out-degree for all vertices.

        Returns:
            ChinesePostmanResult: Solution for directed graph
        """
        # Calculate original cost
        original_cost = self._calculate_graph_cost()

        # Find vertices with degree imbalance
        imbalanced = self._find_imbalanced_vertices()

        if len(imbalanced['excess']) == 0 and len(imbalanced['deficit']) == 0:
            # Graph is already Eulerian
            eulerian_solver = EulerianSolver(self.graph)
            eulerian_result = eulerian_solver.analyze()
            return ChinesePostmanResult(
                has_solution=True,
                optimal_route=eulerian_result.path,
                total_cost=original_cost,
                added_edges=[],
                original_cost=original_cost,
                reason="Directed graph is already Eulerian"
            )

        # Compute shortest paths
        distances, next_vertex = self._floyd_warshall()

        # Find minimum-cost flow to balance vertices
        matching = self._min_cost_flow_matching(imbalanced, distances)

        if matching is None:
            return ChinesePostmanResult(
                has_solution=False,
                optimal_route=None,
                total_cost=0.0,
                added_edges=[],
                original_cost=original_cost,
                reason="Cannot balance directed graph - no valid augmentation found"
            )

        # Augment graph
        augmented_graph = self._create_augmented_graph(matching, distances, next_vertex)

        # Find Eulerian circuit
        # CPP should always produce a circuit (closed walk)
        eulerian_solver = EulerianSolver(augmented_graph)
        eulerian_result = eulerian_solver.analyze(circuit_only=True)

        if not eulerian_result.has_circuit:
            return ChinesePostmanResult(
                has_solution=False,
                optimal_route=None,
                total_cost=0.0,
                added_edges=[],
                original_cost=original_cost,
                reason=f"Failed to create Eulerian circuit in directed graph: {eulerian_result.reason}"
            )

        # Calculate total cost
        added_cost = sum(distances[u][v] for u, v in matching)
        total_cost = original_cost + added_cost

        added_edges = [(u, v, distances[u][v]) for u, v in matching]

        return ChinesePostmanResult(
            has_solution=True,
            optimal_route=eulerian_result.path,
            total_cost=total_cost,
            added_edges=added_edges,
            original_cost=original_cost,
            reason=f"Found optimal solution for directed graph by adding {len(added_edges)} edge(s)"
        )

    def _floyd_warshall(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute all-pairs shortest paths using Floyd-Warshall algorithm.

        Time complexity: O(V^3)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, next vertex in path)
        """
        n = self.graph.num_vertices
        # Initialize distance matrix
        dist = np.full((n, n), float('inf'))
        # Next vertex in shortest path (for path reconstruction)
        next_vertex = np.full((n, n), -1, dtype=int)

        # Distance from vertex to itself is 0
        for i in range(n):
            dist[i][i] = 0
            next_vertex[i][i] = i

        # Set initial distances based on edges
        for i in range(n):
            for j in range(n):
                if i != j and self.graph.adjacency_matrix[i][j] > 0:
                    # Use edge weight
                    dist[i][j] = self.graph.adjacency_matrix[i][j]
                    next_vertex[i][j] = j

        # For undirected graphs, ensure symmetry
        if not self.graph.directed:
            for i in range(n):
                for j in range(i+1, n):
                    if dist[i][j] != float('inf'):
                        dist[j][i] = dist[i][j]
                        next_vertex[j][i] = i

        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_vertex[i][j] = next_vertex[i][k]

        return dist, next_vertex

    def _reconstruct_path(self, start: int, end: int, next_vertex: np.ndarray) -> List[int]:
        """
        Reconstruct shortest path between two vertices.

        Args:
            start (int): Start vertex
            end (int): End vertex
            next_vertex (np.ndarray): Next vertex matrix from Floyd-Warshall

        Returns:
            List[int]: Path from start to end (excluding start, including end)
        """
        if next_vertex[start][end] == -1:
            return []

        path = []
        current = start
        while current != end:
            current = next_vertex[current][end]
            path.append(current)

        return path

    def _find_odd_vertices(self) -> List[int]:
        """
        Find all vertices with odd degree (for undirected graphs).

        Returns:
            List[int]: List of vertices with odd degree
        """
        odd_vertices = []
        for v in range(self.graph.num_vertices):
            if self.graph.get_degree(v) % 2 == 1:
                odd_vertices.append(v)
        return odd_vertices

    def _find_imbalanced_vertices(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find vertices with degree imbalance (for directed graphs).

        For CPP, we care about edge EXISTENCE, not weights.
        So we count each edge as 1, regardless of weight.

        Returns:
            Dict with 'excess' (out > in) and 'deficit' (in > out) vertices
            Each entry is (vertex, imbalance_amount)
        """
        excess = []  # out-degree > in-degree
        deficit = []  # in-degree > out-degree

        for v in range(self.graph.num_vertices):
            # Count edges (not weights)
            in_deg = 0
            out_deg = 0

            for j in range(self.graph.num_vertices):
                if self.graph.adjacency_matrix[j][v] > 0:
                    in_deg += 1
                if self.graph.adjacency_matrix[v][j] > 0:
                    out_deg += 1

            diff = out_deg - in_deg

            if diff > 0:
                excess.append((v, diff))
            elif diff < 0:
                deficit.append((v, -diff))

        return {'excess': excess, 'deficit': deficit}

    def _min_cost_matching(self, odd_vertices: List[int], distances: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find minimum-cost perfect matching on odd-degree vertices.

        Uses brute force for small graphs. For graphs with many odd vertices,
        this could be optimized with Blossom algorithm.

        Args:
            odd_vertices (List[int]): List of odd-degree vertices
            distances (np.ndarray): All-pairs shortest distances

        Returns:
            List[Tuple[int, int]]: List of matched vertex pairs
        """
        n = len(odd_vertices)

        if n == 0:
            return []

        if n % 2 != 0:
            raise ValueError("Number of odd-degree vertices must be even")

        # For small n (≤10), use brute force
        if n <= 10:
            return self._brute_force_matching(odd_vertices, distances)
        else:
            # For larger graphs, use greedy approximation
            return self._greedy_matching(odd_vertices, distances)

    def _brute_force_matching(self, odd_vertices: List[int], distances: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find optimal matching using brute force enumeration.

        Args:
            odd_vertices (List[int]): List of odd-degree vertices
            distances (np.ndarray): All-pairs shortest distances

        Returns:
            List[Tuple[int, int]]: Optimal matching
        """
        n = len(odd_vertices)
        if n == 0:
            return []
        if n == 2:
            return [(odd_vertices[0], odd_vertices[1])]

        # Generate all possible perfect matchings
        min_cost = float('inf')
        best_matching = []

        # Use recursive partitioning to generate matchings
        def generate_matchings(remaining: List[int]) -> List[List[Tuple[int, int]]]:
            if len(remaining) == 0:
                return [[]]
            if len(remaining) == 2:
                return [[(remaining[0], remaining[1])]]

            matchings = []
            first = remaining[0]
            rest = remaining[1:]

            for i, partner in enumerate(rest):
                # Match first with partner
                pair = (first, partner)
                # Recursively match the rest
                remaining_vertices = rest[:i] + rest[i+1:]
                sub_matchings = generate_matchings(remaining_vertices)
                for sub_matching in sub_matchings:
                    matchings.append([pair] + sub_matching)

            return matchings

        all_matchings = generate_matchings(odd_vertices)

        for matching in all_matchings:
            cost = sum(distances[u][v] for u, v in matching)
            if cost < min_cost:
                min_cost = cost
                best_matching = matching

        return best_matching

    def _greedy_matching(self, odd_vertices: List[int], distances: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find approximate matching using greedy algorithm.

        Args:
            odd_vertices (List[int]): List of odd-degree vertices
            distances (np.ndarray): All-pairs shortest distances

        Returns:
            List[Tuple[int, int]]: Approximate matching
        """
        remaining = set(odd_vertices)
        matching = []

        while len(remaining) > 0:
            # Find the pair with minimum distance
            min_dist = float('inf')
            best_pair = None

            vertices_list = list(remaining)
            for i, u in enumerate(vertices_list):
                for v in vertices_list[i+1:]:
                    if distances[u][v] < min_dist:
                        min_dist = distances[u][v]
                        best_pair = (u, v)

            if best_pair:
                matching.append(best_pair)
                remaining.remove(best_pair[0])
                remaining.remove(best_pair[1])
            else:
                break

        return matching

    def _min_cost_flow_matching(self, imbalanced: Dict[str, List[Tuple[int, int]]],
                                 distances: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Find minimum-cost flow to balance directed graph.

        Uses greedy matching for simplicity. For production, consider min-cost flow algorithms.

        Args:
            imbalanced (Dict): Excess and deficit vertices with amounts
            distances (np.ndarray): All-pairs shortest distances

        Returns:
            Optional[List[Tuple[int, int]]]: Matching pairs or None if impossible
        """
        excess = imbalanced['excess'][:]  # (vertex, amount)
        deficit = imbalanced['deficit'][:]  # (vertex, amount)

        matching = []

        # Greedy approach: repeatedly match excess to deficit
        while excess and deficit:
            # Find minimum-cost edge from any excess to any deficit
            min_cost = float('inf')
            best_match = None

            for i, (e_vertex, e_amount) in enumerate(excess):
                for j, (d_vertex, d_amount) in enumerate(deficit):
                    cost = distances[e_vertex][d_vertex]
                    if cost < min_cost and cost != float('inf'):
                        min_cost = cost
                        best_match = (i, j, e_vertex, d_vertex)

            if best_match is None:
                # No valid matching found
                return None

            i, j, e_vertex, d_vertex = best_match
            e_amount = excess[i][1]
            d_amount = deficit[j][1]

            # Add edge(s) to matching
            flow = min(e_amount, d_amount)
            for _ in range(flow):
                matching.append((e_vertex, d_vertex))

            # Update balances
            excess[i] = (e_vertex, e_amount - flow)
            deficit[j] = (d_vertex, d_amount - flow)

            # Remove depleted vertices
            if excess[i][1] == 0:
                excess.pop(i)
            if deficit[j][1] == 0:
                deficit.pop(j)

        # Check if balanced
        if excess or deficit:
            return None

        return matching

    def _create_augmented_graph(self, matching: List[Tuple[int, int]],
                                distances: np.ndarray,
                                next_vertex: np.ndarray) -> Graph:
        """
        Create augmented graph by adding edges along shortest paths for matching pairs.

        For undirected graphs, we convert to directed (with edges in both directions) to support
        multi-edges needed for CPP. For directed graphs, we add directed edges along shortest path.

        Args:
            matching (List[Tuple[int, int]]): Pairs of vertices to connect
            distances (np.ndarray): Shortest distances (for weights)
            next_vertex (np.ndarray): Next vertex matrix for path reconstruction

        Returns:
            Graph: Augmented graph with added edges (directed, with multi-edges)
        """
        # Create a directed graph to support multi-edges
        # Even for originally undirected graphs, we use directed representation
        augmented = Graph(self.graph.num_vertices, directed=True)

        # Copy original edges (bidirectional for undirected graphs)
        # Important: Use edge COUNT (1) not weight for the augmented graph
        # The Eulerian solver interprets matrix values as edge counts for directed graphs
        if self.graph.directed:
            # Convert weights to counts: any edge with weight > 0 becomes count of 1
            for i in range(self.graph.num_vertices):
                for j in range(self.graph.num_vertices):
                    if self.graph.adjacency_matrix[i][j] > 0:
                        augmented.adjacency_matrix[i][j] = 1
        else:
            # Convert undirected to directed by adding edges in both directions
            for i in range(self.graph.num_vertices):
                for j in range(self.graph.num_vertices):
                    if self.graph.adjacency_matrix[i][j] > 0:
                        augmented.adjacency_matrix[i][j] = 1
                        augmented.adjacency_matrix[j][i] = 1

        # Add edges along shortest paths for each matching pair
        for u, v in matching:
            # Reconstruct the shortest path from u to v
            path = [u] + self._reconstruct_path(u, v, next_vertex)

            # Add all edges along this path
            for i in range(len(path) - 1):
                from_v = path[i]
                to_v = path[i + 1]

                # Increment edge count in both directions for undirected
                augmented.adjacency_matrix[from_v][to_v] += 1
                if not self.graph.directed:
                    augmented.adjacency_matrix[to_v][from_v] += 1

        return augmented

    def _calculate_graph_cost(self) -> float:
        """
        Calculate total cost of all edges in the graph.

        Returns:
            float: Sum of all edge weights
        """
        total = 0.0
        if self.graph.directed:
            # Sum all edges
            total = float(np.sum(self.graph.adjacency_matrix))
        else:
            # Sum upper triangle only (avoid double counting)
            for i in range(self.graph.num_vertices):
                for j in range(i+1, self.graph.num_vertices):
                    total += self.graph.adjacency_matrix[i][j]

        return total


def solve_chinese_postman(graph: Graph) -> ChinesePostmanResult:
    """
    Convenience function to solve the Chinese Postman Problem.

    Args:
        graph (Graph): The weighted graph to analyze

    Returns:
        ChinesePostmanResult: Complete solution with optimal route
    """
    solver = ChinesePostmanSolver(graph)
    return solver.solve()
