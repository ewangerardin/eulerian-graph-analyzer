"""Quick test to check GUI launching."""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')  # Ensure TkAgg backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def test_gui():
    print("Creating window...")
    root = tk.Tk()
    root.title("GUI Test")
    root.geometry("800x600")

    print("Adding label...")
    label = ttk.Label(root, text="If you see this window, tkinter works!", font=('Arial', 16))
    label.pack(pady=20)

    print("Adding matplotlib figure...")
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_title("If you see this plot, matplotlib works!")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    print("Window should be visible now!")
    print("Close the window to exit.")
    root.mainloop()

if __name__ == "__main__":
    test_gui()
