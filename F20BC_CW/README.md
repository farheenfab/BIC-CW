## F20BC - Coursework (Farheen and Prasitha)

We will walk you through how to access the PSO through GUI and through the terminal.

### For GUI:
1. Please open the 'gui.py' file and run it.
2. As soon as it runs, the GUI should open up for you.
3. You could test by selecting a local dataset. The GUI accepts .txt, .xlsx, .xls, and .csv files to be used as the dataset.
4. You can now experiment with the hyper-parameters and use any values that you want.
5. If the GUI crashes because of an invalid input, please reload it and try not to give invalid inputs as due to time-constraint we were not able to handle the invalid input cases.
6. Please note that as soon as the results of PSO are shown, we have implemented 3 graphs. To access all of them one by one, you should close the current graph, and a 'Next' button will appear. Please click on the button to access the next graph. The graphs used include:
   4.1. A diagram of the ANN structure
   4.2. A graph comparing Loss with each Epoch
   4.3. A graph comparing global Fitness with each Epoch
7. If you want to restart and check with other hyperparameters, you could close the last graph and a 'Restart' button will appear. This will help you run the GUI again.

### For terminal:
1. Please open the 'annpso.py' file.
2. Please go towards the end. You will see a part of the code that is commented. It will be written: "Uncomment the below lines to run the PSO on Terminal".
3. Please uncomment all the lines of code below it to use the terminal instead of the GUI. (Line 440 - Line 462)
4. If you want to use the GUI after using the terminal, please comment the lines that you had uncommented in the 'annpso.py' file.
