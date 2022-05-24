# Imports
import tkinter as tk
from train import train_network
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DEFAULT_EPOCHS = 100

# Callback for button
def accept_and_train():
    # Try to read EPOCHS from Entry as int
    try:
        EPOCHS = int(epochs_entry.get())
    except:
        EPOCHS = DEFAULT_EPOCHS
        epochs_entry.delete(0, 100)
        epochs_entry.insert(0, str(DEFAULT_EPOCHS))
        ui.update()
    # Clear old plot
    figure.clear()
    # Run training - pass the EPOCHS, and get metrics as result
    rmse, r2 = train_network(EPOCHS)
    # Update label with metrics
    metrics_label.config(text=f"RMSE = {rmse:.3f}, R^2  = {r2:.3f}")
    # Update figure in canvas (figure was "filled" with data in train_network function)
    figure_canvas.get_tk_widget().pack(expand=True)


# Main window
ui = tk.Tk()
ui.title("ANN train")

# An empty figure, and a canvas for it
figure = plt.figure()
figure_canvas = FigureCanvasTkAgg(figure, ui)

# Label and input (Entry) for EPOCHS
tk.Label(ui, text="EPOCHS = ").pack(side=tk.LEFT)
epochs_entry = tk.Entry(ui, width=6)
epochs_entry.insert(tk.END, str(DEFAULT_EPOCHS))
epochs_entry.pack(side=tk.LEFT)

# Button
train_button = tk.Button(ui, text="Train", width=8, command=accept_and_train)
train_button.pack(side=tk.LEFT)

# Label for metrics
metrics_label = tk.Label(ui, text="", font=30)
metrics_label.pack()

# Start UI
ui.mainloop()
