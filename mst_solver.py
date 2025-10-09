"""
Minimum Spanning Tree (MST) solver using Kruskal's algorithm.

This module provides functions to find the minimum spanning tree of a weighted
undirected graph using Kruskal's algorithm with Union-Find data structure.
"""

from typing import List, Tuple, Optional
import numpy as np
from graph import Graph


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure with path compression and union by rank.

    Provides efficient operations for tracking connected components:
    - find(): O(α(n)) amortized where α is inverse Ackermann function
    - union(): O(α(n)) amortized

    Attributes:
        parent (List[int]): Parent pointers for each element
        rank (List[int]): Rank (approximate depth) for union by rank optimization
    """

    def __init__(self, size: int):
        """
        Initialize Union-Find structure with n singleton sets.

        Args:
            size (int): Number of elements (vertices)
        """
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        """
        Find the root of the set containing x with path compression.

        Path compression: make every node on the path point directly to root.

        Args:
            x (int): Element to find

        Returns:
            int: Root of the set containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union the sets containing x and y using union by rank.

        Union by rank: attach smaller tree under root of larger tree.

        Args:
            x (int): Element in first set
            y (int): Element in second set

        Returns:
            bool: True if sets were merged (were different), False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank: attach smaller tree under larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same set.

        Args:
            x (int): First element
            y (int): Second element

        Returns:
            bool: True if x and y are connected
        """
        return self.find(x) == self.find(y)


class MSTResult:
    """
    Result container for MST analysis.

    Attributes:
        has_mst (bool): True if MST exists (graph is connected)
        mst_edges (List[Tuple[int, int, int]]): List of (u, v, weight) edges in MST
        total_weight (int): Total weight of the MST
        num_components (int): Number of connected components (1 if MST exists)
        reason (str): Explanation of the result
    """

    def __init__(
        self,
        has_mst: bool,
        mst_edges: List[Tuple[int, int, int]],
        total_weight: int,
        num_components: int = 1,
        reason: str = ""
    ):
        self.has_mst = has_mst
        self.mst_edges = mst_edges
        self.total_weight = total_weight
        self.num_components = num_components
        self.reason = reason

    def __str__(self) -> str:
        """String representation of the result."""
        if self.has_mst:
            edges_str = ", ".join([f"({u},{v}):{w}" for u, v, w in self.mst_edges])
            return f"MST exists: {len(self.mst_edges)} edges, total weight: {self.total_weight}\nEdges: {edges_str}"
        else:
            return f"No MST (graph has {self.num_components} components): {self.reason}"


class MSTSolver:
    """
    Solver for finding Minimum Spanning Trees using Kruskal's algorithm.

    Time complexity: O(E log E) where E is the number of edges
    Space complexity: O(V + E) where V is the number of vertices
    """

    def __init__(self, graph: Graph):
        """
        Initialize the MST solver with a graph.

        Args:
            graph (Graph): The graph to analyze (must be undirected)

        Raises:
            ValueError: If graph is directed
        """
        if graph.directed:
            raise ValueError("MST solver only works with undirected graphs")

        self.graph = graph

    def solve(self) -> MSTResult:
        """
        Find the Minimum Spanning Tree using Kruskal's algorithm.

        Algorithm:
        1. Sort all edges by weight
        2. Initialize Union-Find structure
        3. Iterate through sorted edges:
           - If edge connects different components, add to MST
           - Union the components
        4. Continue until V-1 edges added or all edges processed

        Time complexity: O(E log E) dominated by sorting

        Returns:
            MSTResult: Complete MST analysis result
        """
        # Check for empty graph
        if self.graph.num_vertices == 0:
            return MSTResult(
                has_mst=False,
                mst_edges=[],
                total_weight=0,
                num_components=0,
                reason="Graph has no vertices"
            )

        # Check for single vertex
        if self.graph.num_vertices == 1:
            return MSTResult(
                has_mst=True,
                mst_edges=[],
                total_weight=0,
                num_components=1,
                reason="Single vertex graph - MST is empty"
            )

        # Collect all edges with weights
        edges = self._get_all_edges()

        # Check if graph has any edges
        if len(edges) == 0:
            # Count number of vertices as components
            return MSTResult(
                has_mst=False,
                mst_edges=[],
                total_weight=0,
                num_components=self.graph.num_vertices,
                reason=f"Graph has no edges - {self.graph.num_vertices} disconnected components"
            )

        # Sort edges by weight (Kruskal's algorithm step 1)
        edges.sort(key=lambda edge: edge[2])

        # Initialize Union-Find
        uf = UnionFind(self.graph.num_vertices)

        # Build MST
        mst_edges = []
        total_weight = 0

        for u, v, weight in edges:
            # If adding this edge doesn't create a cycle
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight

                # Stop when we have V-1 edges (complete MST)
                if len(mst_edges) == self.graph.num_vertices - 1:
                    break

        # Check if we have a complete MST (connected graph)
        if len(mst_edges) == self.graph.num_vertices - 1:
            return MSTResult(
                has_mst=True,
                mst_edges=mst_edges,
                total_weight=total_weight,
                num_components=1,
                reason="Graph is connected - MST found using Kruskal's algorithm"
            )
        else:
            # Graph is disconnected - return minimum spanning forest
            num_components = self._count_components(uf)
            return MSTResult(
                has_mst=False,
                mst_edges=mst_edges,
                total_weight=total_weight,
                num_components=num_components,
                reason=f"Graph is disconnected - {num_components} components (minimum spanning forest)"
            )

    def _get_all_edges(self) -> List[Tuple[int, int, int]]:
        """
        Extract all edges from the graph with their weights.

        For undirected graphs, only include each edge once (i < j).

        Returns:
            List[Tuple[int, int, int]]: List of (u, v, weight) tuples
        """
        edges = []

        for i in range(self.graph.num_vertices):
            for j in range(i + 1, self.graph.num_vertices):  # Only upper triangle
                weight = self.graph.adjacency_matrix[i][j]
                if weight > 0:
                    edges.append((i, j, int(weight)))

        return edges

    def _count_components(self, uf: UnionFind) -> int:
        """
        Count the number of connected components using Union-Find.

        Args:
            uf (UnionFind): Union-Find structure

        Returns:
            int: Number of connected components
        """
        roots = set()
        for i in range(self.graph.num_vertices):
            roots.add(uf.find(i))
        return len(roots)

    def get_mst_weight(self) -> Optional[int]:
        """
        Get the total weight of the MST if it exists.

        Returns:
            Optional[int]: Total weight or None if no MST exists
        """
        result = self.solve()
        return result.total_weight if result.has_mst else None

    def get_mst_edges(self) -> Optional[List[Tuple[int, int, int]]]:
        """
        Get the list of edges in the MST if it exists.

        Returns:
            Optional[List[Tuple[int, int, int]]]: List of edges or None if no MST exists
        """
        result = self.solve()
        return result.mst_edges if result.has_mst else None


def solve_mst(graph: Graph) -> MSTResult:
    """
    Convenience function to find MST of a graph.

    Args:
        graph (Graph): The graph to analyze (must be undirected)

    Returns:
        MSTResult: Complete MST analysis result

    Raises:
        ValueError: If graph is directed
    """
    solver = MSTSolver(graph)
    return solver.solve()
