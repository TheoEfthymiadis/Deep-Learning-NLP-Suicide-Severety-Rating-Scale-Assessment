# Deep-Learning-NLP-Suicide-Severity-Rating-Scale-Assessment
This repository contains a series of data files and python scripts to train and evaluate an LSTM neural network for the analysis of Reddit posts through NLP to predict the
probability of the author to commit suicide based on the SSR scale with 5 ordered classes (Supportive->Indicator->Ideation->Behavior->Attempt)

The first step is to run the preprocessing.py python script. In order to do so, type in terminal:

python preprocessing.py

There are no special arguments. This script has the following functionalitites
1) Imports the data set stored in 'Data' directory in the 500_Reddit_users_posts_labels.csv file
2) Executes a series of preprocessing steps required for the NLP
3) Imports the Glove word embeddings to encode the vocabulary.
   The embeddings file was too big to include in the repo. It can be downloaded from here: https://nlp.stanford.edu/projects/glove/
   Make sure to download glove.42B.300d.zip and copy it to the 'Data' directory of the repository
4) Splits the data in training, validation and test and exports a series of .csv files that will be used for the model training phase
   x_train.csv, x_val.csv, x_test.csv, y_train_csv, y_val.csv, y_test.csv, embedding_matrix.csv
