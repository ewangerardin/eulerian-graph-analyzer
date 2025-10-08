# Eulerian Graph Analyzer - Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python verify_fix.py
   ```

## Running the Application

### GUI Mode (Recommended)

```bash
python main.py
```

**What to expect:**
- Detailed initialization messages will appear in the console
- A GUI window will open, centered on your screen
- The window title will be "Eulerian Graph Analyzer"
- If you don't see the window, check your taskbar or use Alt+Tab

**GUI Features:**
- Interactive adjacency matrix input
- Directed/Undirected graph selection
- Circuit-only mode toggle
- Visual graph representation with NetworkX
- Rainbow-colored path highlighting
- Example graph templates
- Export results to text file

### Test Mode (Command Line)

```bash
python main.py test
```

This runs 6 comprehensive test cases demonstrating:
1. Eulerian circuit in undirected graph
2. Eulerian path without circuit
3. Graph with no Eulerian path
4. Eulerian circuit in directed graph
5. Eulerian path in directed graph
6. Disconnected graph handling

### Full Test Suite

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Expected results:
- 77 tests total (100% pass rate)
- Coverage: 98%+ on core modules

## Troubleshooting

### GUI doesn't open

1. **Check the console output** - it will show exactly where the initialization failed
2. **Verify dependencies:**
   ```bash
   python verify_fix.py
   ```
3. **Check your taskbar** - the window may be minimized
4. **Use Alt+Tab** - the window may be behind other windows

### Import errors

If you see import errors, reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Display issues

The GUI window is set to 1200x800 pixels. If your screen is smaller:
1. The window will still open but may be partially off-screen
2. You can resize it by dragging the edges
3. Or modify the size in gui.py line 37

## System Requirements

- **Python:** 3.8 or higher (tested with 3.11.9)
- **Operating System:** Windows, Linux, or macOS
- **Display:** Minimum 1280x900 recommended for full GUI experience
- **Required packages:**
  - tkinter (usually included with Python)
  - numpy >= 1.24.0
  - matplotlib >= 3.7.0
  - networkx >= 3.0
  - pytest >= 7.4.0 (for testing)
  - pytest-cov >= 4.1.0 (for coverage)

## Features Overview

### Graph Analysis
- **Algorithm:** Hierholzer's algorithm (O(E) complexity)
- **Graph types:** Directed and undirected
- **Special features:** Circuit-only mode (rejects non-returning paths)
- **Validation:** Connectivity checking, degree analysis, matrix validation

### GUI Components
- **Left Panel:**
  - Graph settings (vertices, type, mode)
  - Adjacency matrix input
  - Action buttons
  - Example graph loader

- **Right Panel:**
  - Visual graph representation
  - Analysis results display
  - Path/circuit highlighting

### Results Display
- Vertex degree information
- Eulerian property detection
- Complete path/circuit route
- Verification messages
- Export capability

## Example Usage

### Example 1: Pentagon Circuit
1. Launch GUI: `python main.py`
2. Click "Load Example" → "Eulerian Circuit"
3. Click "Analyze Graph"
4. See the rainbow-colored circuit path

### Example 2: Custom Graph
1. Launch GUI: `python main.py`
2. Set "Number of Vertices" to desired value (2-10)
3. Select graph type (Directed/Undirected)
4. Enter 1s in matrix cells to create edges
5. Click "Analyze Graph"
6. View results and visualization

### Example 3: Circuit-Only Mode
1. Launch GUI: `python main.py`
2. Set "Search Mode" to "Circuit Only"
3. Create or load a graph
4. Click "Analyze Graph"
5. Only circuits (paths returning to start) will be found

## File Structure

```
PyCharmMiscProject/
├── graph.py                 # Graph data structure (98% coverage)
├── eulerian_solver.py       # Hierholzer's algorithm (99% coverage)
├── gui.py                   # GUI implementation
├── main.py                  # Entry point (GUI + tests)
├── tests/
│   ├── test_graph.py        # 28 graph tests
│   ├── test_eulerian_solver.py  # 28 solver tests
│   └── test_integration.py  # 21 integration tests
├── requirements.txt         # Dependencies
├── verify_fix.py           # Installation verification
├── test_gui_simple.py      # Simple GUI test
├── GUI_FIX_NOTES.md        # Technical fix details
└── QUICK_START.md          # This file
```

## Performance Characteristics

- **Graph Creation:** O(V²)
- **Add/Remove Edge:** O(1)
- **Degree Calculation:** O(V) with caching
- **Connectivity Check:** O(V²) via DFS
- **Eulerian Detection:** O(V² + E)
- **Path Construction:** O(E) via Hierholzer's algorithm

## Support

If you encounter issues:

1. **Run verification:**
   ```bash
   python verify_fix.py
   ```

2. **Check console output** - detailed error messages are provided

3. **Test with simple GUI:**
   ```bash
   python test_gui_simple.py
   ```

4. **Review logs** in GUI_FIX_NOTES.md for technical details

## Success Indicators

✅ When everything is working correctly:

1. Running `python verify_fix.py` shows all checks passed
2. Running `python main.py` shows:
   ```
   GUI WINDOW OPENED SUCCESSFULLY!
   Starting main event loop (window will stay open)...
   ```
3. The GUI window appears on screen
4. The window stays open until you close it
5. All interactions work smoothly

## Version Information

- **Application:** Eulerian Graph Analyzer v1.0
- **Test Suite:** 77 tests, 100% pass rate
- **Coverage:** 98%+ on core modules
- **Last Updated:** 2025-10-08
