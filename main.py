import numpy as np
import csv
import mvr

class Main:

    filename  = "antelopestudy.csv"
    predict   = "antelope_predict.csv"
    delimiter = ','

    # Load CSV headers only
    with open(filename) as file:
        reader = csv.reader(file)
        headers = next(reader)

    # Load remaining CSV data
    data = np.loadtxt(filename, delimiter=delimiter, skiprows=1)
    prediction_data = np.loadtxt(predict, delimiter=delimiter, skiprows=1)

    # Perform the regression
    r = mvr.MultiVariableRegression(data, headers, estimates=prediction_data)
    r.plotData()
