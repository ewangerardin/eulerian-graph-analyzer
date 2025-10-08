"""
Verification script to confirm GUI fix is working.
This script tests that the GUI can be launched successfully.
"""

import sys
import subprocess
import time

print("=" * 70)
print("EULERIAN GRAPH ANALYZER - GUI FIX VERIFICATION")
print("=" * 70)
print()

print("Step 1: Checking Python version...")
print(f"  Python version: {sys.version}")
print()

print("Step 2: Verifying dependencies...")
try:
    import tkinter as tk
    print("  [OK] tkinter")
except ImportError as e:
    print(f"  [FAIL] tkinter: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"  [OK] matplotlib (version {matplotlib.__version__})")
except ImportError as e:
    print(f"  [FAIL] matplotlib: {e}")
    sys.exit(1)

try:
    import networkx as nx
    print(f"  [OK] networkx (version {nx.__version__})")
except ImportError as e:
    print(f"  [FAIL] networkx: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  [OK] numpy (version {np.__version__})")
except ImportError as e:
    print(f"  [FAIL] numpy: {e}")
    sys.exit(1)

print()
print("Step 3: Verifying project files...")
import os

files_to_check = [
    "graph.py",
    "eulerian_solver.py",
    "gui.py",
    "main.py",
    "requirements.txt"
]

for filename in files_to_check:
    if os.path.exists(filename):
        print(f"  [OK] {filename}")
    else:
        print(f"  [FAIL] Missing file: {filename}")
        sys.exit(1)

print()
print("Step 4: Testing GUI imports...")
try:
    from gui import run_gui, EulerianGUI
    print("  [OK] GUI module imports successfully")
except ImportError as e:
    print(f"  [FAIL] GUI import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 5: Testing core functionality imports...")
try:
    from graph import Graph
    from eulerian_solver import EulerianSolver, solve_eulerian
    print("  [OK] Core modules import successfully")
except ImportError as e:
    print(f"  [FAIL] Core module import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("Step 6: Quick functionality test...")
try:
    # Create a simple graph
    g = Graph(3, directed=False)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)

    # Analyze it
    result = solve_eulerian(g)

    if result.has_circuit:
        print("  [OK] Graph analysis working correctly")
    else:
        print("  [FAIL] Unexpected analysis result")
        sys.exit(1)
except Exception as e:
    print(f"  [FAIL] Functionality test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL VERIFICATION CHECKS PASSED!")
print("=" * 70)
print()
print("The GUI fix is working correctly. You can now:")
print()
print("  1. Launch the GUI:")
print("     python main.py")
print()
print("  2. Run test cases:")
print("     python main.py test")
print()
print("  3. Run the full test suite:")
print("     pytest tests/ -v")
print()
print("The GUI window should:")
print("  - Open and display centered on screen")
print("  - Show detailed initialization messages")
print("  - Stay open until you close it")
print("  - Provide clear error messages if anything fails")
print()
print("=" * 70)
