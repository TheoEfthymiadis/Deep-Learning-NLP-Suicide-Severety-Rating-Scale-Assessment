# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os

# This script:
# Imports the data set
# Performs a basic data exploration
# Executes a series of preprocessing steps
# Imports the Glove word embeddings to encode the vocabulary
# Exports a series of csv files that will be used for the training phase
# x_train.csv, x_test.csv, y_train_csv, y_test.csv, embedding_matrix.csv

# Define the directory paths
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'Data')

# Import the data
# ssrs = pd.read_csv('C:/MyStaff/MscDataScience/DeepLearning/Ergasia/500_Reddit_users_posts_labels.csv')
ssrs = pd.read_csv(os.path.join(data_dir, '500_Reddit_users_posts_labels.csv'))
print(ssrs.head())




