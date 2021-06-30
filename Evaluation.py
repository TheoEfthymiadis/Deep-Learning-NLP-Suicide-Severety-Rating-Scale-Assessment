import pandas as pd
import math
from sklearn import metrics
import numpy as np
import tensorflow as tf
import os
import sys
from sklearn.metrics import accuracy_score

# Run the script as
# python evaluation.py model_name config_name
# Example: python evaluation.py LSTM1 config5
# The script imports the specific model with the specific configuration and evaluate it on the test set using the CEN

# Define the directory paths
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'Data')  # Data directory
config_dir = (os.path.join(root_dir, 'Configurations'))   # Directory with configuration files
model_dir = (os.path.join(root_dir, 'Models'))   # Directory with exported trained models

# Setting tensorflow seed for reproducibility
tf.random.set_seed(0)

# Reading the user arguments
model_name = sys.argv[1]
config_name = sys.argv[2]

# Importing the best model
imported_model = tf.keras.models.load_model(os.path.join(model_dir,
                                                         os.path.join(model_name, config_name+'\\Checkpoint')))

# Importing the test set for evaluation
x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

# Prediction using the imported model
y_estimate = imported_model.predict(x_test)

# Round the regression outputs to represent classes
for index, element in enumerate(y_estimate):
    element[0] = round(element[0])


def cem_calculation(y_test, y_estimate):
    y_pred_cat = y_estimate.astype(int)
    y_predict_cat = y_pred_cat.reshape(-1, 1)

    # Calculate the confusion matrix
    matrix = (metrics.confusion_matrix(y_test, y_predict_cat, labels=[0, 1, 2, 3, 4]))
    mat = pd.DataFrame(matrix)

    # Calculate the CEM
    mat.loc['Total'] = pd.Series(mat.sum())
    mat['Total'] = mat.sum(axis=1)
    class_proxim = pd.DataFrame(np.zeros((len(mat) - 1, len(mat) - 1)))
    ttl = mat.iloc[-1, -1]
    for row in range(0, len(mat) - 1):
        for column in range(0, len(mat.columns) - 1):
            if row == column:
                if mat.iloc[-1, column] == 0:
                    class_proxim.iloc[row, column] = 0.0000001
                else:
                    class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, column] / 2) / ttl)
            elif abs(row - column) == 1:
                if mat.iloc[-1, row] / 2 + mat.iloc[-1, column] == 0:
                    class_proxim.iloc[row, column] = 0.0000001
                else:
                    class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, row] / 2 + mat.iloc[-1, column]) / ttl)
            elif abs(row - column) == 2:
                if row > column:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 1] + mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2(
                            (mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 1] + mat.iloc[-1, column]) / ttl)
                else:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 1] + mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2(
                            (mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 1] + mat.iloc[-1, column]) / ttl)
            elif abs(row - column) == 3:
                if row > column:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 2] + mat.iloc[-1, row - 1] + \
                            mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 2] +
                                                                     mat.iloc[-1, row - 1] + mat.iloc[
                                                                         -1, column]) / ttl)
                else:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 2] + mat.iloc[-1, column - 1] + \
                            mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 2] +
                                                                     mat.iloc[-1, column - 1] + mat.iloc[
                                                                         -1, column]) / ttl)
            elif abs(row - column) == 4:
                if row > column:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 3] + mat.iloc[-1, row - 2] + mat.iloc[-1, row - 1] + \
                            mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, row] / 2 + mat.iloc[-1, row - 3] +
                                                                     mat.iloc[-1, row - 2] + mat.iloc[-1, row - 1] +
                                                                     mat.iloc[-1, column]) / ttl)
                else:
                    if mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 3] + mat.iloc[-1, column - 2] + \
                            mat.iloc[-1, column - 1] + mat.iloc[-1, column] == 0:
                        class_proxim.iloc[row, column] = 0.0000001
                    else:
                        class_proxim.iloc[row, column] = -math.log2((mat.iloc[-1, row] / 2 + mat.iloc[-1, column - 3] +
                                                                     mat.iloc[-1, column - 2] + mat.iloc[
                                                                         -1, column - 1] + mat.iloc[-1, column]) / ttl)
    cem_ = mat.iloc[:-1, :-1].mul(class_proxim)
    numerator = cem_.to_numpy().sum()
    denominator = 0
    for column in range(0, len(mat) - 1):
        denom = mat.iloc[-1, column] * class_proxim.iloc[column, column]
        denominator = denominator + denom
    cem = numerator / denominator
    print("CEM = ", cem)
    return cem


accuracy = accuracy_score(y_test, y_estimate)
print('Accuracy = ', accuracy)
cem_calculation(y_test, y_estimate)
