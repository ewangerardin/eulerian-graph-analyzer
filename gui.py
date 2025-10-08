"""
Graphical User Interface for Eulerian Graph Analysis.

This module provides a modern tkinter-based GUI with graph visualization
using matplotlib and networkx for analyzing Eulerian paths and circuits.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
from typing import Optional, List
from graph import Graph
from eulerian_solver import EulerianSolver, EulerianResult


class EulerianGUI:
    """
    Main GUI application for Eulerian graph analysis.

    Provides an intuitive interface for creating graphs, analyzing Eulerian properties,
    and visualizing the results with highlighted paths.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the GUI application.

        Args:
            root (tk.Tk): Root tkinter window
        """
        self.root = root
        self.root.title("Eulerian Graph Analyzer")
        self.root.geometry("1200x800")

        # Graph state
        self.graph: Optional[Graph] = None
        self.result: Optional[EulerianResult] = None
        self.num_vertices = 5  # Default
        self.is_directed = False
        self.circuit_only = False  # New: circuit-only mode

        # Matrix entry widgets
        self.matrix_entries = []

        # Setup GUI components
        self._setup_styles()
        self._create_widgets()
        self._create_matrix_grid(self.num_vertices)

    def _setup_styles(self) -> None:
        """Configure ttk styles for modern appearance."""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure button styles
        style.configure('Action.TButton', font=('Arial', 10, 'bold'), padding=5)
        style.configure('Example.TButton', font=('Arial', 9), padding=3)

    def _create_widgets(self) -> None:
        """Create and layout all GUI widgets."""
        # Main container with two panels
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for input and controls
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)

        # Right panel for visualization and results
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)

        # Setup left panel
        self._create_control_section(left_panel)
        self._create_matrix_section(left_panel)
        self._create_action_buttons(left_panel)

        # Setup right panel
        self._create_visualization_section(right_panel)
        self._create_results_section(right_panel)

    def _create_control_section(self, parent: ttk.Frame) -> None:
        """Create the control section with graph settings."""
        control_frame = ttk.LabelFrame(parent, text="Graph Settings", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Number of vertices
        ttk.Label(control_frame, text="Number of Vertices:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.vertices_var = tk.IntVar(value=self.num_vertices)
        vertices_spinbox = ttk.Spinbox(
            control_frame,
            from_=2,
            to=10,
            textvariable=self.vertices_var,
            width=10,
            command=self._on_vertices_changed
        )
        vertices_spinbox.grid(row=0, column=1, sticky=tk.W, pady=2)

        # Graph type
        ttk.Label(control_frame, text="Graph Type:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.directed_var = tk.BooleanVar(value=False)

        radio_frame = ttk.Frame(control_frame)
        radio_frame.grid(row=1, column=1, sticky=tk.W, pady=2)

        ttk.Radiobutton(
            radio_frame,
            text="Undirected",
            variable=self.directed_var,
            value=False,
            command=self._on_graph_type_changed
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            radio_frame,
            text="Directed",
            variable=self.directed_var,
            value=True,
            command=self._on_graph_type_changed
        ).pack(side=tk.LEFT, padx=10)

        # Path/Circuit mode selection
        ttk.Label(control_frame, text="Search Mode:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.circuit_only_var = tk.BooleanVar(value=False)

        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=2, column=1, sticky=tk.W, pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="Path or Circuit",
            variable=self.circuit_only_var,
            value=False,
            command=self._on_mode_changed
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            mode_frame,
            text="Circuit Only",
            variable=self.circuit_only_var,
            value=True,
            command=self._on_mode_changed
        ).pack(side=tk.LEFT, padx=10)

        # Add tooltip/help text
        mode_help = ttk.Label(
            control_frame,
            text="(Circuit: path must return to start)",
            font=('Arial', 8),
            foreground='gray'
        )
        mode_help.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

    def _create_matrix_section(self, parent: ttk.Frame) -> None:
        """Create the adjacency matrix input section."""
        matrix_frame = ttk.LabelFrame(parent, text="Adjacency Matrix", padding=10)
        matrix_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollable canvas for matrix
        canvas = tk.Canvas(matrix_frame, height=300)
        scrollbar = ttk.Scrollbar(matrix_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.matrix_container = ttk.Frame(canvas)

        self.matrix_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.matrix_container, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_action_buttons(self, parent: ttk.Frame) -> None:
        """Create action buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # Primary actions
        ttk.Button(
            button_frame,
            text="Analyze Graph",
            style='Action.TButton',
            command=self._analyze_graph
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            button_frame,
            text="Clear Matrix",
            command=self._clear_matrix
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            button_frame,
            text="Export Results",
            command=self._export_results
        ).pack(fill=tk.X, pady=2)

        # Example graphs
        examples_frame = ttk.LabelFrame(button_frame, text="Load Example", padding=5)
        examples_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            examples_frame,
            text="Eulerian Circuit",
            style='Example.TButton',
            command=lambda: self._load_example("circuit")
        ).pack(fill=tk.X, pady=1)

        ttk.Button(
            examples_frame,
            text="Eulerian Path",
            style='Example.TButton',
            command=lambda: self._load_example("path")
        ).pack(fill=tk.X, pady=1)

        ttk.Button(
            examples_frame,
            text="No Eulerian Path",
            style='Example.TButton',
            command=lambda: self._load_example("none")
        ).pack(fill=tk.X, pady=1)

    def _create_visualization_section(self, parent: ttk.Frame) -> None:
        """Create the graph visualization section."""
        viz_frame = ttk.LabelFrame(parent, text="Graph Visualization", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Create or load a graph to visualize',
                     ha='center', va='center', fontsize=12, color='gray')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.axis('off')

    def _create_results_section(self, parent: ttk.Frame) -> None:
        """Create the results display section."""
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding=10)
        results_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=10,
            wrap=tk.WORD,
            font=('Courier', 10)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "No analysis performed yet.\n\n")
        self.results_text.insert(tk.END, "Click 'Analyze Graph' to check for Eulerian properties.")
        self.results_text.config(state=tk.DISABLED)

    def _create_matrix_grid(self, size: int) -> None:
        """
        Create a grid of entry widgets for the adjacency matrix.

        Args:
            size (int): Number of vertices (matrix will be size x size)
        """
        # Clear existing entries
        for widget in self.matrix_container.winfo_children():
            widget.destroy()
        self.matrix_entries = []

        # Create header row
        ttk.Label(self.matrix_container, text="", width=3).grid(row=0, column=0)
        for j in range(size):
            ttk.Label(
                self.matrix_container,
                text=str(j),
                font=('Arial', 9, 'bold'),
                width=4
            ).grid(row=0, column=j + 1)

        # Create matrix entries
        for i in range(size):
            # Row header
            ttk.Label(
                self.matrix_container,
                text=str(i),
                font=('Arial', 9, 'bold'),
                width=3
            ).grid(row=i + 1, column=0)

            row_entries = []
            for j in range(size):
                entry = ttk.Entry(self.matrix_container, width=4, justify=tk.CENTER)
                entry.insert(0, "0")
                entry.grid(row=i + 1, column=j + 1, padx=1, pady=1)

                # Disable diagonal entries for undirected graphs
                if i == j and not self.is_directed:
                    entry.config(state=tk.DISABLED)

                row_entries.append(entry)

            self.matrix_entries.append(row_entries)

    def _on_vertices_changed(self) -> None:
        """Handle change in number of vertices."""
        self.num_vertices = self.vertices_var.get()
        self._create_matrix_grid(self.num_vertices)

    def _on_graph_type_changed(self) -> None:
        """Handle change in graph type (directed/undirected)."""
        self.is_directed = self.directed_var.get()
        self._create_matrix_grid(self.num_vertices)

    def _on_mode_changed(self) -> None:
        """Handle change in search mode (path or circuit-only)."""
        self.circuit_only = self.circuit_only_var.get()

    def _clear_matrix(self) -> None:
        """Clear all matrix entries."""
        for i in range(len(self.matrix_entries)):
            for j in range(len(self.matrix_entries[i])):
                if self.matrix_entries[i][j]['state'] != tk.DISABLED:
                    self.matrix_entries[i][j].delete(0, tk.END)
                    self.matrix_entries[i][j].insert(0, "0")

        # Clear visualization and results
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Create or load a graph to visualize',
                     ha='center', va='center', fontsize=12, color='gray')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.axis('off')
        self.canvas.draw()

        self._update_results("Matrix cleared. Enter new graph data.")

    def _get_matrix_from_entries(self) -> Optional[np.ndarray]:
        """
        Extract adjacency matrix from entry widgets.

        Returns:
            Optional[np.ndarray]: Adjacency matrix or None if invalid input
        """
        try:
            matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=int)

            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if self.matrix_entries[i][j]['state'] != tk.DISABLED:
                        value = self.matrix_entries[i][j].get().strip()
                        if value:
                            matrix[i][j] = int(value)

                            # For undirected graphs, ensure symmetry
                            if not self.is_directed and i != j:
                                matrix[j][i] = matrix[i][j]
                                self.matrix_entries[j][i].delete(0, tk.END)
                                self.matrix_entries[j][i].insert(0, str(matrix[i][j]))

            return matrix

        except ValueError as e:
            messagebox.showerror("Invalid Input", "Please enter valid integer values in the matrix.")
            return None

    def _analyze_graph(self) -> None:
        """Analyze the graph for Eulerian properties and visualize results."""
        # Get matrix from entries
        matrix = self._get_matrix_from_entries()
        if matrix is None:
            return

        # Create graph
        try:
            self.graph = Graph(self.num_vertices, directed=self.is_directed)
            self.graph.set_adjacency_matrix(matrix)

            # Analyze graph with circuit-only mode
            solver = EulerianSolver(self.graph)
            self.result = solver.analyze(circuit_only=self.circuit_only)

            # Display results
            self._display_results()

            # Visualize graph
            self._visualize_graph()

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error analyzing graph: {str(e)}")

    def _display_results(self) -> None:
        """Display analysis results in the text widget."""
        if not self.result:
            return

        output = []
        output.append("=" * 60)
        output.append("EULERIAN GRAPH ANALYSIS RESULTS")
        output.append("=" * 60)
        output.append("")

        # Graph information
        graph_type = "Directed" if self.graph.directed else "Undirected"
        search_mode = "Circuit Only" if self.circuit_only else "Path or Circuit"
        output.append(f"Graph Type: {graph_type}")
        output.append(f"Search Mode: {search_mode}")
        output.append(f"Vertices: {self.graph.num_vertices}")
        output.append(f"Edges: {self.graph.get_edge_count()}")
        output.append(f"Connected: {'Yes' if self.graph.is_connected() else 'No'}")
        output.append("")

        # Degree information
        output.append("Vertex Degrees:")
        for v in range(self.graph.num_vertices):
            if self.graph.directed:
                in_deg = self.graph.get_in_degree(v)
                out_deg = self.graph.get_out_degree(v)
                output.append(f"  Vertex {v}: in-degree={in_deg}, out-degree={out_deg}")
            else:
                deg = self.graph.get_degree(v)
                output.append(f"  Vertex {v}: degree={deg}")
        output.append("")

        # Eulerian properties
        output.append("Eulerian Properties:")
        output.append(f"  Has Eulerian Circuit: {'Yes' if self.result.has_circuit else 'No'}")
        output.append(f"  Has Eulerian Path: {'Yes' if self.result.has_path else 'No'}")
        output.append(f"  Reason: {self.result.reason}")
        output.append("")

        # Path/Circuit
        if self.result.has_path:
            path_type = "Circuit" if self.result.has_circuit else "Path"
            output.append(f"Eulerian {path_type}:")
            output.append(f"  Start Vertex: {self.result.start_vertex}")
            output.append(f"  Length: {len(self.result.path)} vertices")
            output.append(f"  Route: {' -> '.join(map(str, self.result.path))}")

            # Verify circuit condition if in circuit mode
            if self.circuit_only and self.result.has_circuit:
                if len(self.result.path) > 0 and self.result.path[0] == self.result.path[-1]:
                    output.append(f"  Circuit Verified: Path returns to start vertex {self.result.start_vertex}")
        else:
            if self.circuit_only:
                output.append("No Eulerian circuit exists in circuit-only mode.")
            else:
                output.append("No Eulerian path or circuit exists.")

        output.append("")
        output.append("=" * 60)

        self._update_results("\n".join(output))

    def _update_results(self, text: str) -> None:
        """
        Update the results text widget.

        Args:
            text (str): Text to display
        """
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def _visualize_graph(self) -> None:
        """Visualize the graph using networkx and matplotlib."""
        if not self.graph:
            return

        self.ax.clear()

        # Create networkx graph
        if self.graph.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        # Add edges
        for i in range(self.graph.num_vertices):
            for j in range(self.graph.num_vertices):
                if self.graph.adjacency_matrix[i][j] > 0:
                    G.add_edge(i, j)

        # Layout
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = {}

        # Draw graph
        if self.result and self.result.has_path:
            # Highlight Eulerian path
            self._draw_graph_with_path(G, pos)
        else:
            # Draw normal graph
            nx.draw(
                G,
                pos,
                ax=self.ax,
                with_labels=True,
                node_color='lightblue',
                node_size=800,
                font_size=12,
                font_weight='bold',
                arrows=self.graph.directed,
                arrowsize=20,
                edge_color='gray',
                width=2
            )

        self.ax.set_title(
            f"{'Directed' if self.graph.directed else 'Undirected'} Graph Visualization",
            fontsize=14,
            fontweight='bold'
        )
        self.ax.axis('off')
        self.canvas.draw()

    def _draw_graph_with_path(self, G: nx.Graph, pos: dict) -> None:
        """
        Draw graph with Eulerian path highlighted.

        Args:
            G (nx.Graph): NetworkX graph
            pos (dict): Node positions
        """
        # Create edge list from path
        path_edges = []
        if self.result and len(self.result.path) > 1:
            for i in range(len(self.result.path) - 1):
                path_edges.append((self.result.path[i], self.result.path[i + 1]))

        # All edges
        all_edges = list(G.edges())

        # Draw non-path edges in gray
        non_path_edges = [edge for edge in all_edges if edge not in path_edges]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=self.ax,
            node_color='lightgreen',
            node_size=800
        )

        # Draw non-path edges
        if non_path_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=self.ax,
                edgelist=non_path_edges,
                edge_color='lightgray',
                width=1,
                arrows=self.graph.directed,
                arrowsize=15
            )

        # Draw path edges with gradient colors
        if path_edges:
            # Create color gradient from blue to red
            colors = plt.cm.rainbow(np.linspace(0, 1, len(path_edges)))

            for idx, edge in enumerate(path_edges):
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=self.ax,
                    edgelist=[edge],
                    edge_color=[colors[idx]],
                    width=3,
                    arrows=self.graph.directed,
                    arrowsize=20
                )

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            ax=self.ax,
            font_size=12,
            font_weight='bold'
        )

        # Add path type to title
        path_type = "Eulerian Circuit" if self.result.has_circuit else "Eulerian Path"
        self.ax.set_title(
            f"{path_type} Visualization (Rainbow colors show path sequence)",
            fontsize=14,
            fontweight='bold'
        )

    def _load_example(self, example_type: str) -> None:
        """
        Load example graphs.

        Args:
            example_type (str): Type of example ('circuit', 'path', or 'none')
        """
        if example_type == "circuit":
            # Undirected graph with Eulerian circuit (all even degrees)
            self.vertices_var.set(5)
            self.directed_var.set(False)
            self.num_vertices = 5
            self.is_directed = False
            self._create_matrix_grid(5)

            # Pentagon graph (cycle of 5 vertices)
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
            for i, j in edges:
                self.matrix_entries[i][j].delete(0, tk.END)
                self.matrix_entries[i][j].insert(0, "1")
                self.matrix_entries[j][i].delete(0, tk.END)
                self.matrix_entries[j][i].insert(0, "1")

        elif example_type == "path":
            # Undirected graph with Eulerian path (exactly 2 odd degrees)
            self.vertices_var.set(5)
            self.directed_var.set(False)
            self.num_vertices = 5
            self.is_directed = False
            self._create_matrix_grid(5)

            # Path graph: 0-1-2-3-4 with extra edges
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
            for i, j in edges:
                self.matrix_entries[i][j].delete(0, tk.END)
                self.matrix_entries[i][j].insert(0, "1")
                self.matrix_entries[j][i].delete(0, tk.END)
                self.matrix_entries[j][i].insert(0, "1")

        elif example_type == "none":
            # Graph with no Eulerian path (more than 2 odd degrees)
            self.vertices_var.set(4)
            self.directed_var.set(False)
            self.num_vertices = 4
            self.is_directed = False
            self._create_matrix_grid(4)

            # Complete graph K4 (all vertices have degree 3 - odd)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        self.matrix_entries[i][j].delete(0, tk.END)
                        self.matrix_entries[i][j].insert(0, "1")

    def _export_results(self) -> None:
        """Export analysis results to a text file."""
        if not self.result:
            messagebox.showwarning("No Results", "Please analyze a graph first.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")


def run_gui() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = EulerianGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
