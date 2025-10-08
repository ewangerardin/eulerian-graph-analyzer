"""
Integration tests for the Eulerian Graph Analyzer.

Tests end-to-end workflows combining graph creation, analysis,
and verification of results.
"""

import pytest
import numpy as np
from graph import Graph
from eulerian_solver import EulerianSolver, solve_eulerian


class TestEndToEndWorkflows:
    """Tests for complete workflows."""

    def test_end_to_end_circuit_workflow(self):
        """Test complete workflow for Eulerian circuit detection."""
        # Step 1: Create graph
        num_vertices = 5
        graph = Graph(num_vertices, directed=False)

        # Step 2: Add edges to form a pentagon
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        # Step 3: Verify graph properties
        assert graph.num_vertices == 5
        assert graph.get_edge_count() == 5
        assert graph.is_connected()

        # Step 4: Check all vertices have even degree
        for v in range(num_vertices):
            degree = graph.get_degree(v)
            assert degree == 2  # Pentagon: all vertices have degree 2

        # Step 5: Analyze for Eulerian properties
        result = solve_eulerian(graph)

        # Step 6: Verify results
        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 6  # 5 edges + return to start
        assert result.path[0] == result.path[-1]  # Starts and ends at same vertex

        # Step 7: Verify path validity
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            assert graph.has_edge(u, v) or graph.has_edge(v, u)

    def test_end_to_end_path_workflow(self):
        """Test complete workflow for Eulerian path detection."""
        # Step 1: Create graph
        graph = Graph(5, directed=False)

        # Step 2: Add edges (path with 2 odd-degree vertices)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4)]
        for u, v in edges:
            graph.add_edge(u, v)

        # Step 3: Verify graph properties
        assert graph.is_connected()
        assert graph.get_edge_count() == 5

        # Step 4: Count odd-degree vertices
        odd_degree_count = 0
        odd_vertices = []
        for v in range(5):
            degree = graph.get_degree(v)
            if degree % 2 == 1:
                odd_degree_count += 1
                odd_vertices.append(v)

        assert odd_degree_count == 2  # Should have exactly 2 odd vertices

        # Step 5: Analyze for Eulerian properties
        result = solve_eulerian(graph)

        # Step 6: Verify results
        assert not result.has_circuit  # Path but not circuit
        assert result.has_path
        assert len(result.path) == 6  # 5 edges + 1

        # Step 7: Verify path starts at odd-degree vertex
        assert result.start_vertex in odd_vertices

        # Step 8: Verify path ends at other odd-degree vertex
        end_vertex = result.path[-1]
        assert end_vertex in odd_vertices
        assert end_vertex != result.start_vertex

    def test_end_to_end_no_eulerian_workflow(self):
        """Test workflow for graph with no Eulerian path."""
        # Step 1: Create complete graph K4
        graph = Graph(4, directed=False)

        # Step 2: Add all edges (complete graph)
        for i in range(4):
            for j in range(i + 1, 4):
                graph.add_edge(i, j)

        # Step 3: Verify graph properties
        assert graph.is_connected()
        assert graph.get_edge_count() == 6  # K4 has 6 edges

        # Step 4: Count odd-degree vertices
        odd_degree_count = 0
        for v in range(4):
            degree = graph.get_degree(v)
            if degree % 2 == 1:
                odd_degree_count += 1

        assert odd_degree_count == 4  # All vertices have degree 3 (odd)

        # Step 5: Analyze for Eulerian properties
        result = solve_eulerian(graph)

        # Step 6: Verify no Eulerian path exists
        assert not result.has_circuit
        assert not result.has_path
        assert len(result.path) == 0
        assert "odd degree" in result.reason.lower()


class TestDirectedGraphWorkflows:
    """Integration tests for directed graphs."""

    def test_directed_circuit_workflow(self):
        """Test workflow for directed Eulerian circuit."""
        # Create directed cycle
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        # Verify balanced degrees
        for v in range(4):
            in_deg = graph.get_in_degree(v)
            out_deg = graph.get_out_degree(v)
            assert in_deg == out_deg == 1

        # Analyze
        result = solve_eulerian(graph)

        # Verify circuit
        assert result.has_circuit
        assert result.has_path
        assert len(result.path) == 5

        # Verify path follows directed edges
        for i in range(len(result.path) - 1):
            u, v = result.path[i], result.path[i + 1]
            assert graph.has_edge(u, v)  # Must follow direction

    def test_directed_path_workflow(self):
        """Test workflow for directed Eulerian path."""
        # Create directed path
        graph = Graph(4, directed=True)
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        # Verify degree properties
        assert graph.get_out_degree(0) - graph.get_in_degree(0) == 1  # Start
        assert graph.get_in_degree(3) - graph.get_out_degree(3) == 1  # End

        # Analyze
        result = solve_eulerian(graph)

        # Verify path
        assert not result.has_circuit
        assert result.has_path
        assert result.start_vertex == 0
        assert result.path[-1] == 3


class TestMatrixBasedWorkflow:
    """Tests using adjacency matrix directly."""

    def test_matrix_based_circuit_creation(self):
        """Test creating graph from adjacency matrix."""
        # Create graph
        graph = Graph(4, directed=False)

        # Define adjacency matrix for square graph
        matrix = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

        # Set matrix
        graph.set_adjacency_matrix(matrix)

        # Verify matrix was set correctly
        assert np.array_equal(graph.adjacency_matrix, matrix)

        # Analyze
        result = solve_eulerian(graph)

        # Should have Eulerian circuit
        assert result.has_circuit

    def test_matrix_modification_workflow(self):
        """Test modifying graph through matrix and analyzing."""
        # Create empty graph
        graph = Graph(3, directed=False)

        # Get matrix, modify it, and set it back
        matrix = graph.get_adjacency_matrix()
        matrix[0][1] = matrix[1][0] = 1
        matrix[1][2] = matrix[2][1] = 1
        matrix[2][0] = matrix[0][2] = 1

        graph.set_adjacency_matrix(matrix)

        # Analyze triangle graph
        result = solve_eulerian(graph)

        # Triangle has all vertices with degree 2 (even)
        assert result.has_circuit


class TestErrorHandlingWorkflows:
    """Tests for error handling in workflows."""

    def test_workflow_with_invalid_graph(self):
        """Test workflow handles invalid graph gracefully."""
        # Create graph with invalid construction
        with pytest.raises(ValueError):
            graph = Graph(0, directed=False)

    def test_workflow_with_disconnected_graph(self):
        """Test workflow detects disconnected graph."""
        graph = Graph(4, directed=False)
        # Create two separate edges
        graph.add_edge(0, 1)
        graph.add_edge(2, 3)

        # Verify disconnected
        assert not graph.is_connected()

        # Analyze
        result = solve_eulerian(graph)

        # Should detect disconnection
        assert not result.has_path
        assert "not connected" in result.reason.lower()

    def test_workflow_with_invalid_matrix(self):
        """Test workflow handles invalid matrix."""
        graph = Graph(3, directed=False)

        # Try to set asymmetric matrix on undirected graph
        asymmetric = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])

        with pytest.raises(ValueError, match="must be symmetric"):
            graph.set_adjacency_matrix(asymmetric)


class TestGUIInitialization:
    """Tests for GUI initialization (without actually running GUI)."""

    def test_gui_imports_successfully(self):
        """Test that GUI module can be imported."""
        try:
            import gui
            assert hasattr(gui, 'EulerianGUI')
            assert hasattr(gui, 'run_gui')
        except ImportError as e:
            pytest.fail(f"GUI module import failed: {e}")

    def test_gui_class_exists(self):
        """Test that EulerianGUI class is defined."""
        from gui import EulerianGUI
        assert EulerianGUI is not None

    def test_main_module_integration(self):
        """Test that main module integrates all components."""
        try:
            import main
            assert hasattr(main, 'run_all_tests')
            assert hasattr(main, 'main')
        except ImportError as e:
            pytest.fail(f"Main module import failed: {e}")


class TestCircuitOnlyWorkflow:
    """Integration tests for circuit-only mode."""

    def test_circuit_only_workflow_accepts_circuit(self):
        """Test circuit-only mode accepts valid circuit."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert result.has_circuit
        assert result.has_path
        assert result.path[0] == result.path[-1]

    def test_circuit_only_workflow_rejects_path(self):
        """Test circuit-only mode rejects path that doesn't form circuit."""
        graph = Graph(4, directed=False)
        edges = [(0, 1), (1, 2), (2, 3)]
        for u, v in edges:
            graph.add_edge(u, v)

        solver = EulerianSolver(graph)
        result = solver.analyze(circuit_only=True)

        assert not result.has_circuit
        assert not result.has_path
        assert "circuit-only mode" in result.reason.lower()


class TestLargeGraphWorkflow:
    """Tests for larger graphs."""

    def test_large_circuit_graph(self):
        """Test workflow with larger circuit graph."""
        # Create cycle with 10 vertices
        graph = Graph(10, directed=False)
        for i in range(10):
            graph.add_edge(i, (i + 1) % 10)

        result = solve_eulerian(graph)

        assert result.has_circuit
        assert len(result.path) == 11  # 10 edges + return

    def test_complex_directed_graph(self):
        """Test workflow with complex directed graph."""
        # Create more complex directed graph with circuit
        graph = Graph(6, directed=True)
        edges = [
            (0, 1), (1, 2), (2, 0),  # Inner triangle
            (0, 3), (3, 4), (4, 5), (5, 0)  # Outer path
        ]
        for u, v in edges:
            graph.add_edge(u, v)

        # Check degree balance
        balanced = True
        for v in range(6):
            if graph.get_in_degree(v) != graph.get_out_degree(v):
                balanced = False
                break

        result = solve_eulerian(graph)

        if balanced:
            assert result.has_circuit
        else:
            # May have path but not circuit
            pass  # Depends on exact structure
