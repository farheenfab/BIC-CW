# import tkinter as tk
# from tkinter import *

# gui = Tk()
# gui.title('ANN-PSO GUI')

# Label(gui, text='Enter number of hidden layers').grid(row=0)

# # no_layers = Spinbox(gui, from_ = 1, to = 100)
# # no_layers.grid(row=0, column=1)

# # n_nodes = []
# # for i in range(int(no_layers.get())):
# #     if i != no_layers:
# #         Label(gui, text=f'Enter number of nodes for Layer {i+1}').grid(row=i+1)
# #         n_nodes.append(Entry(gui))
# #     else:
# #         Label(gui, text='Enter number of nodes for Output Layer').grid(row=i+1)
# #         n_nodes.append(Entry(gui))
# #         n_nodes[-1].grid(row=i+1, column=1)

# # functions = []
# # for i in range(len(n_nodes)):
# #     layer_label = f"Layer {i + 1}" if i != len(n_nodes) - 1 else f"Output Layer {i + 1}"
# #     activation_var = tk.StringVar()

# #     Label(gui, text=f"Select activation function for {layer_label}").pack()

# #     Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").pack()
# #     Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").pack()
# #     Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").pack()

# #     functions.append(activation_var)

# # Label(gui, text='Enter number of epochs').grid(row=0)
# # epoch = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter population size').grid(row=0)
# # pop_size = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter alpha weight value').grid(row=0)
# # alpha_w = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter beta value').grid(row=0)
# # beta = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter gamma value').grid(row=0)
# # gamma = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter delta value').grid(row=0)
# # delta = Spinbox(gui, from_ = 1, to_ = 100)

# # Label(gui, text='Enter error criterion').grid(row=0)
# # err_crit = Spinbox(gui, from_ = 0, to_ = 100)

# # Label(gui, text='Enter the number of informants').grid(row=0)
# # pop_size = Spinbox(gui, from_ = 0, to_ = 100)

# #e2 = Entry(gui)

# # e1.grid(row=0, column=1)
# # e2.grid(row=1, column=1)


# gui.mainloop()

import tkinter as tk
from tkinter import *
from tkinter import Spinbox, Label, Radiobutton, StringVar

import tkinter as tk
from tkinter import Spinbox, Label, Radiobutton, StringVar

def create_nodes():
    num_layers = int(no_layers.get())
    
    for i in range(num_layers):
        if i != num_layers - 1:
            tk.Label(gui, text=f'Enter number of nodes for Layer {i+1}').grid(row=i+1, column=0, columnspan=2)
        else:
            tk.Label(gui, text='Enter number of nodes for Output Layer').grid(row=i+1, column=0, columnspan=2)

        n_nodes.append(tk.Entry(gui))
        n_nodes[-1].grid(row=i+1, column=2, columnspan=2)

        layer_label = f"Layer {i + 1}" if i != num_layers - 1 else f"Output Layer {i + 1}"
        activation_var = StringVar()

        tk.Label(gui, text=f"Select activation function for {layer_label}").grid(row=i+1, column=4)
        tk.Radiobutton(gui, text="Logistic", variable=activation_var, value="Logistic").grid(row=i+1, column=5)
        tk.Radiobutton(gui, text="Hyperbolic tangent", variable=activation_var, value="Hyperbolic tangent").grid(row=i+1, column=6)
        tk.Radiobutton(gui, text="ReLU", variable=activation_var, value="ReLU").grid(row=i+1, column=7)

        functions.append(activation_var)

# Create the main window
gui = tk.Tk()
gui.title('ANN-PSO GUI')

# Set up the Spinbox for the number of layers
tk.Label(gui, text='Enter number of hidden layers').grid(row=0)
no_layers = tk.Spinbox(gui, from_=1, to=100)
no_layers.grid(row=0, column=1)

# Create a button to submit the form
submit_button = tk.Button(gui, text='Submit', command=create_nodes)
submit_button.grid(row=0, column=2, columnspan=2)

# Create empty lists to store nodes and functions
n_nodes = []
functions = []

# Start the GUI event loop
gui.mainloop()
