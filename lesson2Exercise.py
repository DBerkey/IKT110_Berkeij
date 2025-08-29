"""
Author: Douwe Berkeij
Date: 29-08-2025
"""

import numpy as np
import json
import fortunaAlgorithm as fa

def runFortuna(dataX, dataY):
    # Convert to numpy arrays
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Print the shapes
    print("dataX shape:", dataX.shape)
    print("dataY shape:", dataY.shape)

    # Call the fortuna_algorithm function
    formula = "a*x1+b*x2+c*x3+d"
    loss_func = "mean((y_true - y_pred)**2)"
    fa.fortuna_algorithm(dataX, dataY, formula, loss_func, max_iter=1000000)

def processFile(lineJSON):
    with open(lineJSON, 'r') as file:
        data = file.readlines()
    dataX = []
    dataY = []
    for line in data:
        json_line = json.loads(line)
        dataX.append(json_line['x'])
        dataY.append(json_line['y'])
    return dataX, dataY

if __name__ == "__main__":
    lineJSON = "C:\\Users\\berke\\Downloads\\some_data_for_class.txt"
    dataX, dataY = processFile(lineJSON)
    runFortuna(dataX, dataY)