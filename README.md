# Deep-Learning-NLP-Suicide-Severity-Rating-Scale-Assessment
This repository contains a series of data files and python scripts to train and evaluate an LSTM neural network for the analysis of Reddit posts through NLP to predict the
probability of the author to commit suicide based on the SSR scale with 5 ordered classes (Supportive->Indicator->Ideation->Behavior->Attempt). We used the Reddit C-SSRS Suicide Dataset that can be found with DOI 10.5281/zenodo.2667859 (https://doi.org/10.5281/zenodo.2667859)

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
   These files are stored in the 'Data' directory.

As a second step, there are two python scripts that have a very similar functionality: LSTM1.py and LSTM2.py.
These two scripts are responsible for building and training an LSTM neural network on the training set. They train models with slightly different architecture.

- LSTM1.py architecture: Embedding Layer -> LSTM layer -> Dropout Layer -> Fully connected layer -> Output Layer
- LSTM2.py architecture: Embedding Layer -> LSTM layer -> Dropout Layer -> LSTM layer -> Dropout Layer -> Fully connected layer -> Output Layer

In order to run these scripts, the user needs to provide them a configuration file with the specific values for the model parameters. Due to the differentiated 
network architectures, the configuration file for each network is slightly different. All configuration files for LSTM1.py are stored under \Configuration\LSTM1, while
all configuration files for LSTM2.py are stored under \Configuration\LSTM2. The configuration files are in stored in JSON format and their names follow a pattern:

configx.json, where x is an integer.

Of course, they could have any name, but it's easier to follow this approach. 
When running the LSTM scripts you type in the chat something like this:

python LSTM1.py config1

This command will import the data set, as well as the configuration parameters stored in configuration file "config1.json" and perform the model training. Then, it will export
the best model that occured during training to Models/LSTM1/config1/checkpoint. If the directory config1 does not exist within the LSTM1 directory, it will be created by the
script. Moreover, it will generate a plot of Training Loss / Validation Loss over the epochs of the training and store it in Models/LSTM1/config1. Finally, it will print a 
summary of the model architecture, as well as the confusion matrix, calculated for the test set. 

DO NOT type: python LSTM.py config1.json

The .json file extension is added by the scripts, so this will lead to errors, since the script will try to access the config1.json.json configuration file. 
