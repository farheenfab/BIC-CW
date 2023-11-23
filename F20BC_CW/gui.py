import tkinter as tk
from sklearn.datasets import load_iris
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from annpso import PSO
import threading
import os
import tkinter.scrolledtext as scrolledtext
import matplotlib.pyplot as plt
import networkx as nx

#initializing the variables to store informaton
no_layers = None
n_nodes = []
temp_node_entries = []
functions = []
temp_funcs = []
f_path = None
filename_label = None
iris = False
X, y = None, None
loss_var = None
global_pso_results = None

#opens a file dialog and asks the user to select a dataset file
#the file chosen is stored in the f_path variable
def browse():
   global f_path, filename_label
   f_path = filedialog.askopenfilename(
       initialdir="/",
       title="Select File", 
       filetypes=(("Text files", "*.txt"), ("Excel files", "*.xlsx"), ("Excel files 2", "*.xls"), ("CSV files", "*.csv"), ("All files", "*.*")))
   
   if f_path:
        filename = os.path.basename(f_path)
        file_explorer.configure(text="File Selected :")
        filename_label.configure(text=filename)
        browse_button.configure(text="Select Another Dataset")

#loads the iris dataset from sckit-learn
def select_iris_dataset():
    global X, y, iris, f_path
    irisd = load_iris()
    X, y = irisd.data, irisd.target
    iris = True
    f_path = "Iris"
    file_explorer.configure(text="File Selected :")
    filename_label.configure(text="Iris Dataset Selected")
    browse_button.configure(text="Select Another Dataset")
    return

def check_dataset():
    if f_path == None:
        messagebox.showwarning("Warning", "Please Select a Dataset!")
        return
    elif iris:
        next_screen1()
    else:
        next_screen1()

def on_value_change(*args):
    value = alpha_w.get()
    print(value)

def on_value_change_1(*args):
    value1 = err_crit.get()
    print(value1)
    
def on_value_change_2(*args):
    value2 = beta.get()
    print(value2)    

def on_value_change_3(*args):
    value3 = gamma.get()
    print(value3)

def on_value_change_4(*args):
    value4 = delta.get()
    print(value4)    
    
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
            tk.Label(gui, text=f"Select activation function for {layer_label}").grid(row=current_row, padx=(0, 20), sticky="w")
            current_row += 1
            tk.Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").grid(row=current_row, padx=(0, 50), sticky="w")
            tk.Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").grid(row=current_row, padx=(100, 100), sticky="w")
            tk.Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").grid(row=current_row, padx = (270,5), sticky="w")
            current_row += 1

        else:
            tk.Label(gui, text=f"Select activation function for {layer_label}").grid(row=current_row, padx=(0, 20), sticky="w")
            current_row += 1
            tk.Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").grid(row=current_row, padx=(0,50), sticky="w")
            tk.Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").grid(row=current_row, padx=(100, 100), sticky="w")
            tk.Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").grid(row=current_row, padx = (270,5), sticky="w")

        temp_funcs.append(activation_var)

    next_button = tk.Button(gui, text='Next', command=check_funcs)
    next_button.grid(row=(num_layers*2)+2, padx=(200, 200))

def check_funcs():
    global functions
    functions = [entry.get() for entry in temp_funcs]  # Get values from Entry widgets
    print(functions)

    all_selected = all(func.get() for func in temp_funcs)
    if not all_selected:
        messagebox.showwarning("Warning", "Select activation function for all layers!")
        return
    else:
        next_screen3()

def next_screen3():

    for widget in gui.winfo_children():
        widget.grid_forget()

    progress_bar = ttk.Progressbar(gui, orient="horizontal", length=300, mode="determinate")
    progress_bar.grid(column=0, row=0, pady=20, padx=10)

    progress_label = tk.Label(gui, text="We will now start with PSO!")
    progress_label.grid(row=i+10, column=0, columnspan=2)

    # Start PSO in a separate thread
    pso_thread = threading.Thread(target=run_pso, args=(progress_bar, progress_label))
    pso_thread.start()

    check_pso_completion()

def run_pso(progress_bar, progress_label):
    # Extract parameters from the GUI inputs
    # Assuming you have a way to extract X, y (input data and labels) from the GUI
    global X, y, global_pso_results
    if not iris:
        X, y = get_data_from_gui()
    else:
        pass
    
    total_epochs = int(epochs.get())
    # Run PSO and update progress bar
    # The PSO function needs to be adapted to update the progress bar
    gbest_fitness, best_acc, best_params, loss_per_epoch, fitness_per_epoch = PSO(X, y, int(no_layers.get()) + 1, n_nodes, functions, total_epochs, int(pop_size.get()),
                                                                                float(alpha_w.get()), float(beta.get()), float(gamma.get()), float(delta.get()),
                                                                                float(err_crit.get()), int(num_informants.get()), loss_var.get(), progress_callback=lambda epoch: update_progress(progress_bar, progress_label, epoch, total_epochs))
    
    print (gbest_fitness)
    print (best_acc)
    print (best_params)
    global_pso_results = (gbest_fitness, best_acc, best_params, loss_per_epoch, fitness_per_epoch)

def check_pso_completion():
    if global_pso_results is not None:
        # PSO computation is done, update GUI
        gbest_fitness, best_acc, best_params, loss_per_epoch, fitness_per_epoch = global_pso_results
        next_screen4(gbest_fitness, best_acc, best_params, loss_per_epoch, fitness_per_epoch)
    else:
        # PSO computation is not done, check again after some delay
        gui.after(100, check_pso_completion)

def next_screen4(gbest_fitness, best_acc, best_params, loss_per_epoch, fitness_per_epoch):
    
    for widget in gui.winfo_children():
        widget.grid_forget()

    # Show pop-up with "PSO Completed!" message
    messagebox.showinfo("PSO Completed!", "PSO Completed!")

    formatted_params = format_best_params(best_params, n_nodes)

    # Create a scrolled text widget
    result_text = scrolledtext.ScrolledText(gui, wrap=tk.WORD, width=40, height=10)
    result_text.grid(column=0, row=1, columnspan=2, padx=10, pady=10)

    # Insert the results into the scrolled text widget
    result_text.insert(tk.END, f"Best Fitness: {gbest_fitness}\n")
    result_text.insert(tk.END, f"Best Accuracy: {best_acc}\n")
    
    for layer, params in formatted_params.items():
        result_text.insert(tk.END, f"{layer} Weights:\n{params['Weights']}\n")
        result_text.insert(tk.END, f"{layer} Biases:\n{params['Biases']}\n")

    # Disable editing in the scrolled text widget
    result_text.configure(state='disabled')

    next_button = tk.Button(gui, text='Next', command=lambda: next_screen5(loss_per_epoch, fitness_per_epoch))
    next_button.grid(row=2, padx=(100, 100))

def update_progress(progress_bar, progress_label, current_epoch, total_epochs):
    epoch_percentage = (current_epoch / total_epochs) * 100
    progress_bar['value'] = epoch_percentage
    progress_label.config(text=f"Epoch: {current_epoch}/{total_epochs} ({epoch_percentage:.2f}%)")
    gui.update_idletasks()

def next_screen5(loss_per_epoch, fitness_per_epoch):

    for widget in gui.winfo_children():
        widget.grid_forget()

    network_layers = [int(entry.get()) for entry in node_entries]  # Get values from Entry widgets
    draw_ann(network_layers)

    next_button = tk.Button(gui, text='Next', command=lambda: next_screen6(loss_per_epoch, fitness_per_epoch))
    next_button.grid(row=0, column=1, padx=(20, 20))

def next_screen6(loss_per_epoch, fitness_per_epoch):

    for widget in gui.winfo_children():
        widget.grid_forget()
    
    epochs_value = int(epochs.get())
    plt.plot(range(epochs_value), loss_per_epoch, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Loss per Epoch")
    plt.show()

    next_button = tk.Button(gui, text='Next', command=lambda: next_screen7(epochs_value, fitness_per_epoch))
    next_button.grid(row=0, column=1, padx=(20, 20))

def next_screen7(epochs, fitness_per_epoch):

    for widget in gui.winfo_children():
        widget.grid_forget()

    plt.plot(range(epochs), fitness_per_epoch, label='Fitness')
    plt.xlabel('Epochs')
    plt.ylabel('Global Fitness')
    plt.legend()
    plt.title("Global Fitness per Epoch")
    plt.show()

    restart_button = tk.Button(gui, text='Restart', command=restart_gui)
    restart_button.grid(row=0, padx=(20, 20))

def restart_gui():
    global no_layers, n_nodes, temp_node_entries, functions, temp_funcs, f_path, filename_label, iris, X, y

    # Reset global variables to their initial state
    no_layers = None
    n_nodes = []
    temp_node_entries = []
    functions = []
    temp_funcs = []
    f_path = None
    filename_label = None
    iris = False
    X, y = None, None

    # Clear the existing elements
    for widget in gui.winfo_children():
        widget.grid_forget()

    # Recreate the initial elements
    create_initial_elements()

def on_loss_selection(loss_var):
    selected_option = loss_var.get()
    print("Selected Option:", selected_option)

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

def format_best_params(best_params, nodes):
    formatted_params = {}
    pointer = 0

    for i in range(len(nodes) - 1):
        current_layer_nodes = nodes[i]
        next_layer_nodes = nodes[i+1]

        weight_size = current_layer_nodes * next_layer_nodes
        bias_size = next_layer_nodes

        weights = best_params[pointer:pointer + weight_size].reshape(next_layer_nodes, current_layer_nodes)
        pointer += weight_size

        biases = best_params[pointer:pointer + bias_size]
        pointer += bias_size

        formatted_params[f'Layer {i+1}'] = {'Weights': weights, 'Biases': biases}

    return formatted_params

def draw_ann(layers):
    # Create a new graph
    G = nx.DiGraph()

    # Variables to keep track of node positions
    pos = {}
    v_spacing = 1
    h_spacing = 2

    # Node counter
    node_count = 0

    # Create nodes with positions
    layer_nodes = []  # List to hold nodes for each layer
    for i, layer_size in enumerate(layers):
        nodes = range(node_count, node_count + layer_size)
        layer_nodes.append(nodes)
        for j, node in enumerate(nodes):
            pos[node] = ((-i * h_spacing), (j - layer_size / 2) * v_spacing)
        node_count += layer_size

    # Add nodes and edges to the graph
    for i, nodes in enumerate(layer_nodes):
        G.add_nodes_from(nodes)
        if i > 0:
            for n1 in layer_nodes[i - 1]:
                for n2 in nodes:
                    G.add_edge(n1, n2)

    # Draw the ANN
    plt.title('ANN Diagram') 
    nx.draw(G, pos, with_labels=False, node_size=700, node_color="skyblue", linewidths=2, font_size=15, font_weight='bold')
    plt.gca().invert_xaxis()  # Invert X-axis to have input layer on the left
    plt.show()

def create_initial_elements():

    global file_explorer, filename_label, browse_button, iris_button, header, no_layers, epochs, pop_size, alpha_w, beta, gamma, delta, err_crit, num_informants, loss_var

    file_explorer = tk.Label(gui, text="Select dataset file")
    file_explorer.grid(row=0, column=0, sticky="w")

    filename_label = tk.Label(gui, text="")
    filename_label.grid(row=0, column=1, sticky="w")

    browse_button = tk.Button(gui, text="Browse Dataset", command=browse)
    browse_button.grid(row=1, column=0, sticky="w")

    iris_button = tk.Button(gui, text="Use Iris Dataset", command=select_iris_dataset)
    iris_button.grid(row=1, column=1, padx=(10, 0), sticky="w")

    tk.Label(gui, text='Does the data contain a header?').grid(row=2, sticky="w")
    header = IntVar()
    Checkbutton(gui, variable=header).grid(row=2, column=1, sticky=W)
    tk.Label(gui, text='Please keep it unchecked for CW dataset', fg='blue').grid(row=2, column=1, padx=30)

    i = 3
    tk.Label(gui, text='Enter number of hidden layers').grid(row=i, sticky="w")
    no_layers = tk.Spinbox(gui, from_=1, to=100)
    no_layers.grid(row=i, column=1)

    epoch_def = tk.StringVar(gui, value="50")
    tk.Label(gui, text='Enter number of epochs').grid(row=i+1, sticky="w")
    epochs = tk.Spinbox(gui, from_=1, to=100000, textvariable=epoch_def)
    epochs.grid(row=i+1, column=1)

    pop_def = tk.StringVar(gui, value="100")
    tk.Label(gui, text='Enter population size').grid(row=i+2, sticky="w")
    pop_size = tk.Spinbox(gui, from_=1, to=100000000, textvariable=pop_def)
    pop_size.grid(row=i+2, column=1)

    alpha_def = tk.StringVar(gui, value="0.5")
    tk.Label(gui, text='Enter alpha weight value').grid(row=i+3, sticky="w")
    alpha_w = tk.Spinbox(gui, from_=0.0, to=10, increment=0.1, format="%0.1f", command = on_value_change, textvariable=alpha_def)
    alpha_w.grid(row=i+3, column=1)

    beta_def = tk.StringVar(gui, value="2.0")
    tk.Label(gui, text='Enter beta value').grid(row=i+4, sticky="w")
    beta = tk.Spinbox(gui, from_=0.0, to=10, increment=0.1, format="%0.1f", command = on_value_change_2, textvariable=beta_def)
    beta.grid(row=i+4, column=1)

    gamma_def = tk.StringVar(gui, value="2.0")
    tk.Label(gui, text='Enter gamma value').grid(row=i+5, sticky="w")
    gamma = tk.Spinbox(gui, from_=0.0, to=10, increment=0.1, format="%0.1f", command = on_value_change_3, textvariable=gamma_def)
    gamma.grid(row=i+5, column=1)

    delta_def = tk.StringVar(gui, value="2.0")
    tk.Label(gui, text='Enter delta value').grid(row=i+6, sticky="w")
    delta = tk.Spinbox(gui, from_=0.0, to=10, increment=0.1, format="%0.1f", command = on_value_change_4, textvariable=delta_def)
    delta.grid(row=i+6, column=1)

    tk.Label(gui, text='Enter error criterion').grid(row=i+7, sticky="w")
    err_crit = tk.Spinbox(gui, from_=0.00001, to=10, increment=0.00001, command=on_value_change_1)
    err_crit.grid(row=i+7, column=1)

    noi_def = tk.StringVar(gui, value="3")
    tk.Label(gui, text='Enter the number of informants').grid(row=i+8, sticky="w")
    num_informants = tk.Spinbox(gui, from_=0, to=100000000, textvariable=noi_def)
    num_informants.grid(row=i+8, column=1)

    tk.Label(gui, text='Select the Loss function').grid(row=i+9, sticky="w")
    loss_var = tk.StringVar(gui)
    options = ["MSE", "Binary Cross Entropy", "Hinge"]
    loss_var.set(options[0])
    loss_menu_width = 18
    loss_menu = tk.OptionMenu(gui, loss_var, *options)
    loss_menu.config(width=loss_menu_width)  # Set the width of the OptionMenu
    loss_menu.bind("<Configure>", on_loss_selection(loss_var))
    loss_menu.grid(row=i+9, column=1)

    next_button = tk.Button(gui, text='Next', command=check_dataset)
    next_button.grid(row=i+10, column=1, padx=(3, 220))

# Create the main window
gui = tk.Tk()
gui.title('ANN-PSO GUI')

i = 3
create_initial_elements()

# Start the GUI event loop
gui.mainloop()