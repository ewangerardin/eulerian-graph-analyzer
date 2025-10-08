# Eulerian Graph Analyzer

A complete Python application for analyzing Eulerian paths and circuits in graphs using Hierholzer's algorithm with a modern GUI interface.

## Features

- **Graph Representation**: Adjacency matrix implementation with O(1) edge lookup
- **Eulerian Detection**: Efficient detection of Eulerian paths and circuits
- **Hierholzer's Algorithm**: O(E) time complexity for finding routes
- **Interactive GUI**: Modern tkinter interface with graph visualization
- **Support for Both**: Directed and undirected graphs
- **Visual Highlighting**: Rainbow-colored path visualization
- **Export Results**: Save analysis results to text files

## Project Structure

```
.
├── graph.py              # Graph class with adjacency matrix
├── eulerian_solver.py    # Eulerian path/circuit detection
├── gui.py                # GUI with visualization
├── main.py               # Entry point and test cases
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- NetworkX

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/Mac**: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launch GUI Application

```bash
python main.py
```

The GUI provides:
- **Matrix Input Grid**: Manual entry of adjacency matrix
- **Graph Type Selection**: Radio buttons for directed/undirected
- **Analyze Button**: Check for Eulerian properties
- **Results Panel**: Displays analysis results and path
- **Graph Visualization**: Visual representation with highlighted paths
- **Load Example Buttons**: Pre-configured example graphs
- **Export Function**: Save results to file

### Run Test Cases

```bash
python main.py test
```

This runs 6 comprehensive test cases:
1. Undirected graph with Eulerian circuit (Pentagon)
2. Undirected graph with Eulerian path only
3. Graph with no Eulerian path (Complete K4)
4. Directed graph with Eulerian circuit
5. Directed graph with Eulerian path only
6. Disconnected graph

### Command-Line Help

```bash
python main.py --help
```

## Algorithm Details

### Graph Class (`graph.py`)

- **Adjacency Matrix**: N×N matrix for N vertices
- **O(1) Edge Lookup**: Constant time to check if edge exists
- **Degree Calculation**: Cached for efficiency
- **Connectivity Check**: BFS/DFS traversal in O(V²) time

### Eulerian Solver (`eulerian_solver.py`)

**Eulerian Circuit (Undirected)**:
- All vertices with edges have even degree
- Graph is connected

**Eulerian Path (Undirected)**:
- Exactly 2 vertices have odd degree
- Graph is connected

**Eulerian Circuit (Directed)**:
- All vertices have in-degree = out-degree
- Graph is weakly connected

**Eulerian Path (Directed)**:
- Exactly one vertex with out-degree - in-degree = 1 (start)
- Exactly one vertex with in-degree - out-degree = 1 (end)
- All other vertices have in-degree = out-degree

**Hierholzer's Algorithm**:
1. Start from appropriate vertex
2. Follow edges, removing them as you go
3. When stuck, backtrack and record vertices
4. Reverse the recorded sequence for final path
5. Time complexity: O(E) where E = number of edges

## Examples

### Example 1: Pentagon (Eulerian Circuit)

```python
from graph import Graph
from eulerian_solver import solve_eulerian

# Create pentagon graph
graph = Graph(5, directed=False)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
for u, v in edges:
    graph.add_edge(u, v)

# Analyze
result = solve_eulerian(graph)
print(result)
# Output: Eulerian Circuit exists: 0 -> 1 -> 2 -> 3 -> 4 -> 0
```

### Example 2: Path Graph (Eulerian Path)

```python
# Create path with exactly 2 odd-degree vertices
graph = Graph(5, directed=False)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
for u, v in edges:
    graph.add_edge(u, v)

result = solve_eulerian(graph)
print(result)
# Output: Eulerian Path exists (start: 0): 0 -> 1 -> 2 -> 3 -> 1 -> 3 -> 4
```

### Example 3: Using GUI

1. Launch GUI: `python main.py`
2. Set number of vertices (e.g., 5)
3. Select graph type (Undirected/Directed)
4. Enter adjacency matrix values
5. Click "Analyze Graph"
6. View results and visualization
7. Optionally export results to file

Or use the "Load Example" buttons for quick demonstrations.

## GUI Features

### Input Section
- **Vertices Spinner**: Set graph size (2-10 vertices)
- **Radio Buttons**: Choose directed or undirected
- **Matrix Grid**: Enter edge weights (1 for edge, 0 for no edge)
- Auto-symmetry for undirected graphs

### Analysis Results
- Graph properties (type, vertices, edges, connectivity)
- Degree information for each vertex
- Eulerian properties (circuit/path existence)
- Complete path sequence if it exists

### Visualization
- **Spring Layout**: Automatic node positioning
- **Rainbow Coloring**: Path edges colored in sequence
- **Arrow Support**: For directed graphs
- **Highlighted Path**: Visual route indication

### Example Graphs
- **Eulerian Circuit**: Pentagon graph (all even degrees)
- **Eulerian Path**: Path graph (2 odd-degree vertices)
- **No Eulerian**: Complete K4 (4 odd-degree vertices)

## Code Quality

- **Type Hints**: All functions fully typed
- **Comprehensive Docstrings**: Detailed documentation
- **Error Handling**: Validation for invalid inputs
- **Optimizations**: Degree caching, efficient algorithms
- **Test Coverage**: 6 test cases covering all scenarios

## Performance

- **Graph Creation**: O(V²) for initialization
- **Add/Remove Edge**: O(1)
- **Connectivity Check**: O(V²) using BFS
- **Eulerian Detection**: O(V²) for degree calculation
- **Path Finding**: O(E) using Hierholzer's algorithm
- **Total Analysis**: O(V² + E) = O(V²) for dense graphs

## Limitations

- Matrix size limited to 10×10 in GUI (can be increased in code)
- Undirected graphs don't support self-loops (by design)
- Visualization quality decreases for very large graphs
- Memory usage is O(V²) due to adjacency matrix

## Future Enhancements

- Import/export graph from file (JSON, GraphML)
- Adjacency list option for sparse graphs
- Step-by-step algorithm visualization
- Support for weighted graph analysis
- Hamiltonian path detection
- Graph coloring algorithms

## License

This project is provided as-is for educational purposes.

## Author

Created with Claude Code - AI-powered development assistant

## References

- Hierholzer's Algorithm: Original paper (1873)
- Graph Theory: Introduction to Graph Theory by Douglas B. West
- NetworkX Documentation: https://networkx.org/
- Matplotlib Documentation: https://matplotlib.org/
