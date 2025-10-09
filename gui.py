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
from chinese_postman import ChinesePostmanSolver, ChinesePostmanResult
from mst_solver import MSTSolver, MSTResult


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
        self.root.title("🔄 Eulerian Graph Analyzer")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)

        # Graph state
        self.graph: Optional[Graph] = None
        self.result: Optional[EulerianResult] = None
        self.cpp_result: Optional[ChinesePostmanResult] = None
        self.mst_result: Optional[MSTResult] = None
        self.num_vertices = 5  # Default
        self.is_directed = False
        self.circuit_only = False  # New: circuit-only mode
        self.analysis_mode = "eulerian"  # "eulerian", "cpp", or "mst"

        # Matrix entry widgets
        self.matrix_entries = []

        # Setup GUI components
        self._setup_styles()
        self._create_widgets()
        self._create_matrix_grid(self.num_vertices)

    def _setup_styles(self) -> None:
        """Configure ultra-modern styles with dark theme and gradients."""
        style = ttk.Style()
        style.theme_use('clam')

        # Ultra-modern color palette - Dark theme
        self.colors = {
            'bg_dark': '#0f172a',        # Deep navy background
            'bg_card': '#1e293b',        # Card background
            'bg_light': '#334155',       # Light accent
            'primary': '#3b82f6',        # Bright blue
            'primary_hover': '#2563eb',  # Blue hover
            'accent': '#8b5cf6',         # Purple accent
            'success': '#10b981',        # Green
            'warning': '#f59e0b',        # Orange
            'danger': '#ef4444',         # Red
            'text': '#f1f5f9',           # Light text
            'text_muted': '#94a3b8',     # Muted text
            'border': '#475569'          # Border color
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg_dark'])

        # Ultra-modern main action button - gradient effect
        style.configure('Action.TButton',
                       font=('Segoe UI', 12, 'bold'),
                       padding=(25, 12),
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       relief='flat')
        style.map('Action.TButton',
                 background=[('active', self.colors['primary_hover']),
                           ('pressed', '#1d4ed8')])

        # Modern secondary buttons with subtle style
        style.configure('Secondary.TButton',
                       font=('Segoe UI', 10),
                       padding=(18, 10),
                       background=self.colors['bg_light'],
                       foreground=self.colors['text'],
                       borderwidth=1,
                       relief='flat')
        style.map('Secondary.TButton',
                 background=[('active', '#475569')])

        # Sleek example buttons
        style.configure('Example.TButton',
                       font=('Segoe UI', 9),
                       padding=(12, 6),
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       borderwidth=1)
        style.map('Example.TButton',
                 background=[('active', self.colors['bg_light'])])

        # Modern frames with dark theme
        style.configure('Dark.TFrame',
                       background=self.colors['bg_dark'])

        style.configure('Card.TFrame',
                       background=self.colors['bg_card'],
                       relief='flat')

        # Enhanced label frames with modern look
        style.configure('Card.TLabelframe',
                       background=self.colors['bg_card'],
                       borderwidth=2,
                       relief='flat')
        style.configure('Card.TLabelframe.Label',
                       font=('Segoe UI', 11, 'bold'),
                       foreground=self.colors['primary'],
                       background=self.colors['bg_card'])

        # Labels with dark theme
        style.configure('TLabel',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'])

        style.configure('Muted.TLabel',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text_muted'])

        # Radiobuttons with modern styling
        style.configure('TRadiobutton',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))

        # Spinbox styling
        style.configure('TSpinbox',
                       fieldbackground=self.colors['bg_light'],
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       borderwidth=0)

    def _create_widgets(self) -> None:
        """Create and layout all GUI widgets with modern dark theme."""
        # Main container with dark theme
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left panel for input and controls - dark themed
        left_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_panel.configure(width=450)

        # Right panel for visualization and results - dark themed
        right_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Setup left panel
        self._create_control_section(left_panel)
        self._create_matrix_section(left_panel)
        self._create_action_buttons(left_panel)

        # Setup right panel (now includes both visualization and results)
        self._create_visualization_section(right_panel)

    def _create_control_section(self, parent: ttk.Frame) -> None:
        """Create the control section with graph settings."""
        control_frame = ttk.LabelFrame(parent, text="⚙️  Graph Configuration",
                                      style='Card.TLabelframe', padding=15)
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

        # Analysis type selection
        ttk.Label(control_frame, text="Analysis Type:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.analysis_mode_var = tk.StringVar(value="eulerian")

        analysis_frame = ttk.Frame(control_frame)
        analysis_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=2)

        ttk.Radiobutton(
            analysis_frame,
            text="Eulerian",
            variable=self.analysis_mode_var,
            value="eulerian",
            command=self._on_analysis_mode_changed
        ).pack(side=tk.LEFT)

        ttk.Radiobutton(
            analysis_frame,
            text="Chinese Postman",
            variable=self.analysis_mode_var,
            value="cpp",
            command=self._on_analysis_mode_changed
        ).pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(
            analysis_frame,
            text="MST",
            variable=self.analysis_mode_var,
            value="mst",
            command=self._on_analysis_mode_changed
        ).pack(side=tk.LEFT, padx=10)

        # Add analysis type tooltip/help text
        analysis_help = ttk.Label(
            control_frame,
            text="(CPP: shortest route | MST: minimum spanning tree)",
            font=('Arial', 8),
            foreground='gray'
        )
        analysis_help.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

    def _create_matrix_section(self, parent: ttk.Frame) -> None:
        """Create the adjacency matrix input section."""
        matrix_frame = ttk.LabelFrame(parent, text="📊  Adjacency Matrix",
                                     style='Card.TLabelframe', padding=15)
        matrix_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Modern help text with dark theme
        help_frame = tk.Frame(matrix_frame, bg=self.colors['bg_light'],
                             highlightbackground=self.colors['border'],
                             highlightthickness=1)
        help_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        help_text = tk.Label(
            help_frame,
            text="💡 Matrix Input Guide:\n• Undirected: use 0 or 1 only\n• Directed: use integers ≥0 (values >1 = multi-edges)",
            font=('Segoe UI', 9),
            foreground=self.colors['text'],
            background=self.colors['bg_light'],
            justify=tk.LEFT,
            padx=12,
            pady=10
        )
        help_text.pack(anchor=tk.W)

        # Scrollable canvas for matrix with dark theme
        canvas = tk.Canvas(matrix_frame, height=320,
                          bg=self.colors['bg_card'],
                          highlightthickness=0)
        scrollbar = ttk.Scrollbar(matrix_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.matrix_container = tk.Frame(canvas, bg=self.colors['bg_card'])

        self.matrix_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.matrix_container, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _create_action_buttons(self, parent: ttk.Frame) -> None:
        """Create action buttons section with improved styling."""
        button_frame = ttk.Frame(parent, padding=5)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # Primary action - Analyze (most prominent)
        ttk.Button(
            button_frame,
            text="🔍  Analyze Graph",
            style='Action.TButton',
            command=self._analyze_graph
        ).pack(fill=tk.X, pady=(0, 10))

        # Secondary actions
        ttk.Button(
            button_frame,
            text="🗑️  Clear Matrix",
            style='Secondary.TButton',
            command=self._clear_matrix
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Button(
            button_frame,
            text="💾  Export Results",
            style='Secondary.TButton',
            command=self._export_results
        ).pack(fill=tk.X, pady=(0, 8))

        # Example graphs section
        examples_frame = ttk.LabelFrame(button_frame, text="📚  Load Example",
                                       style='Card.TLabelframe', padding=10)
        examples_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            examples_frame,
            text="✓  Eulerian Circuit",
            style='Example.TButton',
            command=lambda: self._load_example("circuit")
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            examples_frame,
            text="→  Eulerian Path",
            style='Example.TButton',
            command=lambda: self._load_example("path")
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            examples_frame,
            text="✗  No Eulerian Path",
            style='Example.TButton',
            command=lambda: self._load_example("none")
        ).pack(fill=tk.X, pady=2)

    def _create_visualization_section(self, parent: ttk.Frame) -> None:
        """Create the graph visualization and results sections with resizable paned window."""
        # Create PanedWindow for resizable visualization and results
        paned_window = tk.PanedWindow(parent, orient=tk.VERTICAL,
                                     bg=self.colors['bg_dark'],
                                     sashwidth=8,
                                     sashrelief=tk.RAISED,
                                     bd=0)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top pane: Visualization
        viz_frame = ttk.LabelFrame(paned_window, text="🎨  Graph Visualization",
                                  style='Card.TLabelframe', padding=15)

        # Matplotlib figure with ultra-modern dark theme
        self.fig = Figure(figsize=(10, 7), dpi=100,
                         facecolor=self.colors['bg_card'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['bg_dark'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial empty plot with modern dark theme message
        self.ax.text(0.5, 0.5, '🎨 Create or load a graph to visualize',
                     ha='center', va='center', fontsize=16,
                     color=self.colors['text_muted'],
                     style='italic', weight='light')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.axis('off')

        # Bottom pane: Results
        results_frame = ttk.LabelFrame(paned_window, text="📋  Analysis Results",
                                      style='Card.TLabelframe', padding=15)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=12,
            wrap=tk.WORD,
            font=('Consolas', 10, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            relief=tk.FLAT,
            padx=15,
            pady=15,
            insertbackground=self.colors['primary'],
            selectbackground=self.colors['primary'],
            selectforeground='white'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "📝 No analysis performed yet.\n\n", 'muted')
        self.results_text.insert(tk.END, "Click '🔍 Analyze Graph' to check for Eulerian properties.", 'muted')

        # Configure tags for colored output
        self.results_text.tag_config('muted', foreground=self.colors['text_muted'])
        self.results_text.tag_config('primary', foreground=self.colors['primary'],
                                    font=('Consolas', 10, 'bold'))
        self.results_text.tag_config('success', foreground=self.colors['success'],
                                    font=('Consolas', 10, 'bold'))
        self.results_text.tag_config('warning', foreground=self.colors['warning'])
        self.results_text.tag_config('error', foreground=self.colors['danger'])

        self.results_text.config(state=tk.DISABLED)

        # Add both panes to PanedWindow
        paned_window.add(viz_frame, minsize=300)
        paned_window.add(results_frame, minsize=150)

        # Set initial sash position (60% for viz, 40% for results)
        self.root.update_idletasks()
        paned_window.sash_place(0, 0, 400)

    def _create_results_section(self, parent: ttk.Frame) -> None:
        """Create the results display section - now handled in _create_visualization_section."""
        # This method is no longer used but kept for compatibility
        pass

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

        # Create header row with modern dark styling
        header_label = tk.Label(self.matrix_container, text="", width=3,
                               bg=self.colors['bg_card'])
        header_label.grid(row=0, column=0, padx=2, pady=2)

        for j in range(size):
            header = tk.Label(
                self.matrix_container,
                text=str(j),
                font=('Segoe UI', 10, 'bold'),
                width=5,
                bg=self.colors['bg_light'],
                fg=self.colors['primary'],
                pady=8
            )
            header.grid(row=0, column=j + 1, padx=1, pady=2)

        # Create matrix entries with ultra-modern styling
        for i in range(size):
            # Row header
            row_header = tk.Label(
                self.matrix_container,
                text=str(i),
                font=('Segoe UI', 10, 'bold'),
                width=3,
                bg=self.colors['bg_light'],
                fg=self.colors['primary'],
                padx=8
            )
            row_header.grid(row=i + 1, column=0, padx=2, pady=1)

            row_entries = []
            for j in range(size):
                # Modern styled entry with dark theme
                entry = tk.Entry(self.matrix_container, width=5,
                               justify=tk.CENTER,
                               font=('Segoe UI', 11, 'bold'),
                               bg=self.colors['bg_light'],
                               fg=self.colors['text'],
                               insertbackground=self.colors['primary'],
                               relief=tk.FLAT,
                               borderwidth=2,
                               highlightbackground=self.colors['border'],
                               highlightcolor=self.colors['primary'],
                               highlightthickness=1)
                entry.insert(0, "0")
                entry.grid(row=i + 1, column=j + 1, padx=1, pady=1)

                # Disable diagonal entries for undirected graphs
                if i == j and not self.is_directed:
                    entry.config(state=tk.DISABLED,
                               disabledbackground=self.colors['bg_card'],
                               disabledforeground=self.colors['text_muted'])
                else:
                    # Add focus effects
                    entry.bind('<FocusIn>', lambda e, ent=entry: ent.config(
                        highlightthickness=2))
                    entry.bind('<FocusOut>', lambda e, ent=entry: ent.config(
                        highlightthickness=1))

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

    def _on_analysis_mode_changed(self) -> None:
        """Handle change in analysis mode (Eulerian or CPP)."""
        self.analysis_mode = self.analysis_mode_var.get()

    def _clear_matrix(self) -> None:
        """Clear all matrix entries."""
        for i in range(len(self.matrix_entries)):
            for j in range(len(self.matrix_entries[i])):
                if self.matrix_entries[i][j]['state'] != tk.DISABLED:
                    self.matrix_entries[i][j].delete(0, tk.END)
                    self.matrix_entries[i][j].insert(0, "0")

        # Clear visualization with dark theme
        self.ax.clear()
        self.ax.text(0.5, 0.5, '🎨 Create or load a graph to visualize',
                     ha='center', va='center', fontsize=16,
                     color=self.colors['text_muted'],
                     style='italic', weight='light')
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.ax.axis('off')
        self.ax.set_facecolor(self.colors['bg_dark'])
        self.fig.set_facecolor(self.colors['bg_card'])
        self.canvas.draw()

        self._update_results("✓ Matrix cleared. Enter new graph data.")

    def _get_matrix_from_entries(self) -> Optional[np.ndarray]:
        """
        Extract adjacency matrix from entry widgets.

        Validates:
        - Undirected graphs: only binary values (0 or 1) allowed
        - Directed graphs: positive integers allowed (for multi-edges)

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
                            int_value = int(value)

                            # Validate based on graph type
                            if not self.is_directed:
                                # Undirected graphs: only binary values allowed
                                if int_value not in [0, 1]:
                                    messagebox.showerror(
                                        "Invalid Input",
                                        f"Undirected graphs must use binary values (0 or 1) only.\n"
                                        f"Found value {int_value} at position ({i}, {j})."
                                    )
                                    return None
                            else:
                                # Directed graphs: positive integers allowed
                                if int_value < 0:
                                    messagebox.showerror(
                                        "Invalid Input",
                                        f"Matrix values must be non-negative.\n"
                                        f"Found value {int_value} at position ({i}, {j})."
                                    )
                                    return None

                            matrix[i][j] = int_value

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
        """Analyze the graph for Eulerian properties, CPP, or MST and visualize results."""
        # Get matrix from entries
        matrix = self._get_matrix_from_entries()
        if matrix is None:
            return

        # Create graph
        try:
            self.graph = Graph(self.num_vertices, directed=self.is_directed)
            self.graph.set_adjacency_matrix(matrix)

            # Analyze based on selected mode
            if self.analysis_mode == "mst":
                # MST analysis - only for undirected graphs
                if self.is_directed:
                    messagebox.showerror("Invalid Graph Type",
                                       "MST analysis only works with undirected graphs.\n"
                                       "Please set Graph Type to 'Undirected' and try again.")
                    return
                mst_solver = MSTSolver(self.graph)
                self.mst_result = mst_solver.solve()
                self.result = None  # Clear Eulerian result
                self.cpp_result = None  # Clear CPP result
            elif self.analysis_mode == "cpp":
                # Chinese Postman Problem analysis
                cpp_solver = ChinesePostmanSolver(self.graph)
                self.cpp_result = cpp_solver.solve()
                self.result = None  # Clear Eulerian result
                self.mst_result = None  # Clear MST result
            else:
                # Eulerian analysis with circuit-only mode
                solver = EulerianSolver(self.graph)
                self.result = solver.analyze(circuit_only=self.circuit_only)
                self.cpp_result = None  # Clear CPP result
                self.mst_result = None  # Clear MST result

            # Display results
            self._display_results()

            # Visualize graph
            self._visualize_graph()

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error analyzing graph: {str(e)}")

    def _display_results(self) -> None:
        """Display analysis results in the text widget."""
        if self.mst_result:
            self._display_mst_results()
        elif self.cpp_result:
            self._display_cpp_results()
        elif self.result:
            self._display_eulerian_results()

    def _display_eulerian_results(self) -> None:
        """Display Eulerian analysis results."""
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

    def _display_cpp_results(self) -> None:
        """Display Chinese Postman Problem results."""
        output = []
        output.append("=" * 60)
        output.append("CHINESE POSTMAN PROBLEM RESULTS")
        output.append("=" * 60)
        output.append("")

        # Graph information
        graph_type = "Directed" if self.graph.directed else "Undirected"
        output.append(f"Graph Type: {graph_type}")
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
                parity = "even" if deg % 2 == 0 else "odd"
                output.append(f"  Vertex {v}: degree={deg} ({parity})")
        output.append("")

        # CPP Solution
        output.append("Chinese Postman Solution:")
        output.append(f"  Has Solution: {'Yes' if self.cpp_result.has_solution else 'No'}")
        output.append(f"  Reason: {self.cpp_result.reason}")
        output.append("")

        if self.cpp_result.has_solution:
            output.append("Cost Analysis:")
            output.append(f"  Original Graph Cost: {self.cpp_result.original_cost}")
            if self.cpp_result.added_edges:
                added_cost = self.cpp_result.total_cost - self.cpp_result.original_cost
                output.append(f"  Added Edges Cost: {added_cost}")
                output.append(f"  Total Route Cost: {self.cpp_result.total_cost}")
            else:
                output.append(f"  Total Route Cost: {self.cpp_result.total_cost}")
                output.append("  (No edges needed to be duplicated - graph is Eulerian)")
            output.append("")

            if self.cpp_result.added_edges:
                output.append("Edges to Duplicate:")
                for u, v, cost in self.cpp_result.added_edges:
                    output.append(f"  Edge ({u}, {v}): cost = {cost}")
                output.append("")

            if self.cpp_result.optimal_route:
                output.append("Optimal Route:")
                output.append(f"  Length: {len(self.cpp_result.optimal_route)} vertices")
                output.append(f"  Route: {' -> '.join(map(str, self.cpp_result.optimal_route))}")
        else:
            output.append("No solution found for Chinese Postman Problem.")

        output.append("")
        output.append("=" * 60)

        self._update_results("\n".join(output))

    def _display_mst_results(self) -> None:
        """Display Minimum Spanning Tree results."""
        output = []
        output.append("=" * 60)
        output.append("MINIMUM SPANNING TREE (MST) RESULTS")
        output.append("=" * 60)
        output.append("")

        # Graph information
        graph_type = "Directed" if self.graph.directed else "Undirected"
        output.append(f"Graph Type: {graph_type}")
        output.append(f"Vertices: {self.graph.num_vertices}")
        output.append(f"Edges: {self.graph.get_edge_count()}")
        output.append(f"Connected: {'Yes' if self.graph.is_connected() else 'No'}")
        output.append("")

        # Degree information
        output.append("Vertex Degrees:")
        for v in range(self.graph.num_vertices):
            deg = self.graph.get_degree(v)
            parity = "even" if deg % 2 == 0 else "odd"
            output.append(f"  Vertex {v}: degree={deg} ({parity})")
        output.append("")

        # Check Eulerian properties for information
        solver = EulerianSolver(self.graph)
        eulerian_result = solver.analyze(circuit_only=self.circuit_only)

        output.append("Eulerian Properties (for reference):")
        output.append(f"  Has Eulerian Circuit: {'Yes' if eulerian_result.has_circuit else 'No'}")
        output.append(f"  Has Eulerian Path: {'Yes' if eulerian_result.has_path else 'No'}")
        output.append("")

        # MST Solution
        output.append("Minimum Spanning Tree:")
        output.append(f"  Has MST: {'Yes' if self.mst_result.has_mst else 'No'}")
        output.append(f"  Reason: {self.mst_result.reason}")
        output.append("")

        if self.mst_result.has_mst or len(self.mst_result.mst_edges) > 0:
            output.append("MST Details:")
            output.append(f"  Number of Edges in MST: {len(self.mst_result.mst_edges)}")
            output.append(f"  Total Weight: {self.mst_result.total_weight}")
            output.append(f"  Number of Components: {self.mst_result.num_components}")
            output.append("")

            if self.mst_result.mst_edges:
                output.append("MST Edges:")
                for u, v, weight in self.mst_result.mst_edges:
                    output.append(f"  Edge ({u}, {v}): weight = {weight}")
                output.append("")

                # Calculate if MST is Eulerian
                if self.mst_result.has_mst:
                    # Check if MST itself has Eulerian properties
                    mst_graph = Graph(self.graph.num_vertices, directed=False)
                    for u, v, w in self.mst_result.mst_edges:
                        mst_graph.add_edge(u, v, weight=w)

                    mst_solver = EulerianSolver(mst_graph)
                    mst_eulerian = mst_solver.analyze()

                    output.append("MST Eulerian Properties:")
                    output.append(f"  MST has Eulerian Circuit: {'Yes' if mst_eulerian.has_circuit else 'No'}")
                    output.append(f"  MST has Eulerian Path: {'Yes' if mst_eulerian.has_path else 'No'}")
        else:
            output.append("No MST found (graph is disconnected or has no edges).")

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
        if self.mst_result and (self.mst_result.has_mst or len(self.mst_result.mst_edges) > 0):
            # Highlight MST edges
            self._draw_graph_with_mst(G, pos)
        elif self.cpp_result and self.cpp_result.has_solution:
            # Highlight CPP route with added edges
            self._draw_graph_with_cpp(G, pos)
        elif self.result and self.result.has_path:
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
            fontsize=18,
            fontweight='bold',
            color=self.colors['primary'],
            pad=25
        )
        self.ax.axis('off')
        self.ax.set_facecolor(self.colors['bg_dark'])
        self.fig.set_facecolor(self.colors['bg_card'])
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

        # Add path type to title with modern styling
        path_type = "Eulerian Circuit" if self.result.has_circuit else "Eulerian Path"
        self.ax.set_title(
            f"✨ {path_type} Found! 🌈",
            fontsize=18,
            fontweight='bold',
            color=self.colors['success'],
            pad=25
        )
        self.ax.set_facecolor(self.colors['bg_dark'])
        self.fig.set_facecolor(self.colors['bg_card'])

    def _draw_graph_with_cpp(self, G: nx.Graph, pos: dict) -> None:
        """
        Draw graph with CPP solution highlighted.

        Args:
            G (nx.Graph): NetworkX graph
            pos (dict): Node positions
        """
        # Create edge list from optimal route
        route_edges = []
        if self.cpp_result and self.cpp_result.optimal_route and len(self.cpp_result.optimal_route) > 1:
            for i in range(len(self.cpp_result.optimal_route) - 1):
                route_edges.append((self.cpp_result.optimal_route[i], self.cpp_result.optimal_route[i + 1]))

        # Create set of added edges (edges that were duplicated)
        added_edge_set = set()
        if self.cpp_result and self.cpp_result.added_edges:
            for u, v, cost in self.cpp_result.added_edges:
                added_edge_set.add((u, v))
                if not self.graph.directed:
                    added_edge_set.add((v, u))

        # All edges
        all_edges = list(G.edges())

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=self.ax,
            node_color='lightcoral',
            node_size=800
        )

        # Draw original edges (not in route) in gray
        non_route_edges = [edge for edge in all_edges if edge not in route_edges]
        if non_route_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=self.ax,
                edgelist=non_route_edges,
                edge_color='lightgray',
                width=1,
                arrows=self.graph.directed,
                arrowsize=15
            )

        # Draw route edges with gradient colors
        # Highlight duplicated edges differently
        if route_edges:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(route_edges)))

            for idx, edge in enumerate(route_edges):
                # Check if this edge was added (duplicated)
                is_added = edge in added_edge_set

                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=self.ax,
                    edgelist=[edge],
                    edge_color=[colors[idx]],
                    width=4 if is_added else 3,  # Thicker for duplicated edges
                    style='dashed' if is_added else 'solid',  # Dashed for duplicated
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

        # Add edge weights as labels
        edge_labels = {}
        for i in range(self.graph.num_vertices):
            for j in range(self.graph.num_vertices):
                if self.graph.adjacency_matrix[i][j] > 0:
                    edge_labels[(i, j)] = str(self.graph.adjacency_matrix[i][j])

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            ax=self.ax,
            font_size=8
        )

        # Add CPP info to title
        cost_str = f"Cost: {self.cpp_result.total_cost}"
        added_str = f" ({len(self.cpp_result.added_edges)} edges duplicated)" if self.cpp_result.added_edges else ""
        self.ax.set_title(
            f"📮 Chinese Postman Solution - {cost_str}{added_str}",
            fontsize=16,
            fontweight='bold',
            color=self.colors['warning'],
            pad=25
        )
        self.ax.set_facecolor(self.colors['bg_dark'])
        self.fig.set_facecolor(self.colors['bg_card'])

    def _draw_graph_with_mst(self, G: nx.Graph, pos: dict) -> None:
        """
        Draw graph with MST edges highlighted.

        Args:
            G (nx.Graph): NetworkX graph
            pos (dict): Node positions
        """
        # Create set of MST edges
        mst_edge_set = set()
        if self.mst_result and self.mst_result.mst_edges:
            for u, v, weight in self.mst_result.mst_edges:
                mst_edge_set.add((u, v))
                mst_edge_set.add((v, u))  # Add reverse for undirected

        # All edges
        all_edges = list(G.edges())

        # Separate MST and non-MST edges
        mst_edges = [edge for edge in all_edges if edge in mst_edge_set]
        non_mst_edges = [edge for edge in all_edges if edge not in mst_edge_set]

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=self.ax,
            node_color='#90EE90',  # Light green
            node_size=800
        )

        # Draw non-MST edges in light gray
        if non_mst_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=self.ax,
                edgelist=non_mst_edges,
                edge_color='lightgray',
                width=1,
                style='dotted',
                arrows=False
            )

        # Draw MST edges in green with thicker lines
        if mst_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=self.ax,
                edgelist=mst_edges,
                edge_color='#00AA00',  # Dark green
                width=4,
                arrows=False
            )

        # Draw labels
        nx.draw_networkx_labels(
            G,
            pos,
            ax=self.ax,
            font_size=12,
            font_weight='bold'
        )

        # Add edge weights as labels
        edge_labels = {}
        for i in range(self.graph.num_vertices):
            for j in range(i + 1, self.graph.num_vertices):
                if self.graph.adjacency_matrix[i][j] > 0:
                    edge_labels[(i, j)] = str(self.graph.adjacency_matrix[i][j])

        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            ax=self.ax,
            font_size=9,
            font_color='#00AA00' if mst_edges else 'gray'
        )

        # Add MST info to title
        status = "Found" if self.mst_result.has_mst else "Minimum Spanning Forest"
        weight_str = f"Total Weight: {self.mst_result.total_weight}"
        self.ax.set_title(
            f"🌲 Minimum Spanning Tree {status} - {weight_str}",
            fontsize=16,
            fontweight='bold',
            color=self.colors['success'],
            pad=25
        )
        self.ax.set_facecolor(self.colors['bg_dark'])
        self.fig.set_facecolor(self.colors['bg_card'])

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
    print("Initializing GUI application...")
    print("Creating main window...")

    try:
        root = tk.Tk()
        print("Root window created successfully")

        print("Initializing GUI components...")
        app = EulerianGUI(root)
        print("GUI components initialized successfully")

        # Force window to appear on top and center it
        print("Configuring window appearance...")
        root.update_idletasks()

        # Center the window on screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = 1200
        window_height = 800
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Force window to appear on top
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)
        root.focus_force()

        print("")
        print("=" * 60)
        print("GUI WINDOW OPENED SUCCESSFULLY!")
        print("=" * 60)
        print("If you don't see the window:")
        print("  1. Check your taskbar for 'Eulerian Graph Analyzer'")
        print("  2. Try Alt+Tab to switch to the window")
        print("  3. The window is centered on your screen")
        print("=" * 60)
        print("")
        print("Starting main event loop (window will stay open)...")

        root.mainloop()

        print("GUI window closed by user.")

    except Exception as e:
        print("")
        print("=" * 60)
        print("ERROR: Failed to launch GUI!")
        print("=" * 60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("")
        print("This may be caused by:")
        print("  1. Missing tkinter installation")
        print("  2. Display/graphics driver issues")
        print("  3. Missing matplotlib backend")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_gui()
