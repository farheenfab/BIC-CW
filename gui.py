import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
import pandas as pd
from annpso import PSO
import threading

no_layers = None
n_nodes = []
temp_node_entries = []
functions = []
temp_funcs = []

def browse():
   global f_path
   f_path = filedialog.askopenfilename(
       initialdir="/",
       title="Select File", 
       filetypes=(("Text files", "*.txt"), ("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")))
   
   if f_path:
        file_explorer.configure(text="File Selected : " + f_path)

def on_value_change(*args):
    value = alpha_w.get()
    print(value)

def on_value_change_1(*args):
    value1 = err_crit.get()
    print(value1)

def next_screen1():
    # Clear the existing elements
    for widget in gui.winfo_children():
        widget.grid_forget()

    create_nodes()

def create_nodes():
    global node_entries
    node_entries = []

    num_layers = int(no_layers.get())
    for i in range(num_layers+1):
        if i != num_layers:
            tk.Label(gui, text=f'Enter number of nodes for Layer {i+1}').grid(row=i, column=0)
        else:
            tk.Label(gui, text='Enter number of nodes for Output Layer').grid(row=i, column=0)

        node_entry = tk.Spinbox(gui, from_=1, to=100000000)
        node_entry.grid(row=i, column=1)
        node_entries.append(node_entry)

    next_button = tk.Button(gui, text='Next', command=next_screen2)
    next_button.grid(row=num_layers+1, column=1, padx=(3, 220))

def next_screen2():
    global n_nodes
    n_nodes = [int(entry.get()) for entry in node_entries]  # Get values from Entry widgets
    print(n_nodes)

    # Clear the existing elements
    for widget in gui.winfo_children():
        widget.grid_forget()
    
    num_layers = int(no_layers.get())

    global temp_funcs
    temp_funcs = []
    current_row = 0

    for i in range(num_layers+1):
        layer_label = f"Layer {i + 1}" if i != num_layers else "Output Layer"
        activation_var = StringVar()
        
        if layer_label != "Output Layer":
            tk.Label(gui, text=f"Select activation function for {layer_label}").grid(row=current_row, padx=(0, 20))
            current_row += 1
            tk.Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").grid(row=current_row, padx=(0, 50))
            tk.Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").grid(row=current_row, padx=(220, 10))
            tk.Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").grid(row=current_row, padx = (450,5))
            current_row += 1

        else:
            tk.Label(gui, text=f"Select activation function for {layer_label}").grid(row=current_row, padx=(0, 20))
            current_row += 1
            tk.Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").grid(row=current_row, padx=(0,50))
            tk.Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").grid(row=current_row, padx=(220, 10))
            tk.Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").grid(row=current_row, padx = (450,5))

        temp_funcs.append(activation_var)

    next_button = tk.Button(gui, text='Next', command=next_screen3)
    next_button.grid(row=i+9, column=1, padx=(3, 220))

def next_screen3():
    global functions
    functions = [entry.get() for entry in temp_funcs]  # Get values from Entry widgets
    print(functions)

    for widget in gui.winfo_children():
        widget.grid_forget()

    progress_bar = ttk.Progressbar(gui, orient="horizontal", length=300, mode="determinate")
    progress_bar.grid(column=0, row=0, pady=20, padx=10)

    # Start PSO in a separate thread
    pso_thread = threading.Thread(target=run_pso, args=(progress_bar,))
    pso_thread.start()

def run_pso(progress_bar):
    # Extract parameters from the GUI inputs
    # Assuming you have a way to extract X, y (input data and labels) from the GUI
    X, y = get_data_from_gui()

    # Run PSO and update progress bar
    # The PSO function needs to be adapted to update the progress bar
    best_params = PSO(X, y, int(no_layers.get()) + 1, n_nodes, functions, int(epochs.get()), int(pop_size.get()), 
                      float(alpha_w.get()), int(beta.get()), int(gamma.get()), int(delta.get()), 
                      float(err_crit.get()), int(num_informants.get()), "MSE", progress_callback=update_progress, progress_bar=progress_bar)

    # Update the GUI with results after PSO completion
    result_label = tk.Label(gui, text=f"PSO completed! Best params: {best_params}")
    result_label.grid(column=0, row=1)

def update_progress(progress_bar, value):
    progress_bar['value'] = value * 100
    gui.update_idletasks()  # Update the GUI to reflect the change

def get_data_from_gui():
    # Implement this function to extract X, y (input data and labels) from the GUI
    # Example: X, y = ... 
    if header.get() == 1:
        data = pd.read_csv(f_path, header=0)
    else:
        data = pd.read_csv(f_path, header=None)

    # Extract features (X) and labels (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    print(data.head())
    return X, y

# Create the main window
gui = tk.Tk()
gui.title('ANN-PSO GUI')

file_explorer = tk.Label(gui, text="Select dataset file",font=("bold"))
file_explorer.grid(row=0, column=0, sticky="w")

button=tk.Button(gui, text="Browse Dataset", command=browse)
button.grid(row=0, column=1, padx=10)
print(file_explorer)

tk.Label(gui, text='Does the data contain a header?').grid(row=1)
header = IntVar()
Checkbutton(gui, variable=header).grid(row=1, column=1, sticky=W)
tk.Label(gui, text='Please keep it unchecked for CW dataset', fg='blue').grid(row=1, column=1, padx=30)

# Set up the Spinbox for the number of layers
i = 2
tk.Label(gui, text='Enter number of hidden layers').grid(row=i)
no_layers = tk.Spinbox(gui, from_=1, to=100)
no_layers.grid(row=i, column=1)

tk.Label(gui, text='Enter number of epochs').grid(row=i+1)
epochs = tk.Spinbox(gui, from_=1, to=100000)
epochs.grid(row=i+1, column=1)

tk.Label(gui, text='Enter population size').grid(row=i+2)
pop_size = tk.Spinbox(gui, from_=1, to=100000000)
pop_size.grid(row=i+2, column=1)
    
tk.Label(gui, text='Enter alpha weight value').grid(row=i+3)
alpha_w = tk.Spinbox(gui, from_=0.0, to=10, increment=0.1, format="%0.1f", command = on_value_change)
alpha_w.grid(row=i+3, column=1)

tk.Label(gui, text='Enter beta value').grid(row=i+4)
beta = tk.Spinbox(gui, from_=1, to=100)
beta.grid(row=i+4, column=1)

tk.Label(gui, text='Enter gamma value').grid(row=i+5)
gamma = tk.Spinbox(gui, from_=1, to=100)
gamma.grid(row=i+5, column=1)

tk.Label(gui, text='Enter delta value').grid(row=i+6)
delta = tk.Spinbox(gui, from_=1, to=100)
delta.grid(row=i+6, column=1)

tk.Label(gui, text='Enter error criterion').grid(row=i+7)
err_crit = tk.Spinbox(gui, from_=0.00001, to=10, increment=0.00001, command=on_value_change_1)
err_crit.grid(row=i+7, column=1)

tk.Label(gui, text='Enter the number of informants').grid(row=i+8)
num_informants = tk.Spinbox(gui, from_=1, to=100000000)
num_informants.grid(row=i+8, column=1)

next_button = tk.Button(gui, text='Next', command=next_screen1)
next_button.grid(row=i+9, column=1, padx=(3, 220))

# Start the GUI event loop
gui.mainloop()