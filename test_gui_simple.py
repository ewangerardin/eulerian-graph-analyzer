"""Simple test to check if tkinter and matplotlib are working."""

import sys
print("Python version:", sys.version)
print()

print("Testing tkinter import...")
try:
    import tkinter as tk
    print("[OK] tkinter imported successfully")
except Exception as e:
    print(f"[FAIL] tkinter import failed: {e}")
    sys.exit(1)

print()
print("Testing matplotlib import...")
try:
    import matplotlib
    print(f"[OK] matplotlib imported successfully (version {matplotlib.__version__})")
    print(f"  Backend: {matplotlib.get_backend()}")
except Exception as e:
    print(f"[FAIL] matplotlib import failed: {e}")
    sys.exit(1)

print()
print("Testing matplotlib backend for tkinter...")
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    print("[OK] matplotlib tkinter backend imported successfully")
except Exception as e:
    print(f"[FAIL] matplotlib backend import failed: {e}")
    sys.exit(1)

print()
print("Testing networkx import...")
try:
    import networkx as nx
    print(f"[OK] networkx imported successfully (version {nx.__version__})")
except Exception as e:
    print(f"[FAIL] networkx import failed: {e}")
    sys.exit(1)

print()
print("Testing numpy import...")
try:
    import numpy as np
    print(f"[OK] numpy imported successfully (version {np.__version__})")
except Exception as e:
    print(f"[FAIL] numpy import failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("Creating simple tkinter window to test GUI...")
print("=" * 60)
print()

try:
    root = tk.Tk()
    root.title("Test Window")
    root.geometry("400x300")

    label = tk.Label(root, text="If you see this window, tkinter is working!",
                     font=("Arial", 14), pady=20)
    label.pack()

    button = tk.Button(root, text="Close Window", command=root.destroy,
                       font=("Arial", 12), padx=20, pady=10)
    button.pack()

    print("[OK] Test window created successfully")
    print()
    print("Window should be visible on your screen.")
    print("If you see it, click 'Close Window' button.")
    print("If you don't see it, check your taskbar or use Alt+Tab.")
    print()
    print("Starting mainloop()...")

    root.mainloop()

    print()
    print("[OK] Window closed successfully - tkinter is working!")

except Exception as e:
    print()
    print(f"[FAIL] Error creating window: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
