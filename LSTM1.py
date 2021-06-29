# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import json
import os
import sys

# Define the directory paths
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'Data')  # Data directory
config_dir = (os.path.join(root_dir, 'Configurations'))   # Directory with configuration files
model_dir = (os.path.join(root_dir, 'Models'))   # Directory with exported trained models

# Setting tensorflow seed for reproducibility
tf.random.set_seed(0)

# import files
embedding_matrix = pd.read_csv(os.path.join(data_dir, 'embedding_matrix.csv'))
x_train = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
x_val = pd.read_csv(os.path.join(data_dir, 'x_val.csv'))
x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

# Model configuration #

# ### Constant Parameters ###
max_sequence_length = 3183
verbosity_mode = 1
EMBEDDING_DIM = 300
vocab_size = 9211

# ### Configurable parameters imported from the configuration file ###

# Reading configuration arguments
model_path = 'LSTM1'   # This is the name of the folder that corresponds to the specific model architecture
config = sys.argv[1]

# Creating the configuration file path
config_path = os.path.join(os.path.join(config_dir, model_path), (config + '.json'))

# Read configuration file
with open(config_path) as json_data_file:
    config_data = json.load(json_data_file)   # Dictionary to store the configuration parameters

batch_size = int(config_data["batch_size"])
number_of_epochs = int(config_data["number_of_epochs"])
optimizer = 'rmsprop'
LSTM1_units = int(config_data["LSTM1_units"])
LSTM1_drop = float(config_data["LSTM1_drop"])
LSTM1_rec_drop = float(config_data["LSTM1_rec_drop"])
Dense1_units = int(config_data["Dense1_units"])
Dense1_act = config_data["Dense1_act"]
Dropout1 = float(config_data["Dropout1"])

# Disable eager execution
tf.compat.v1.disable_eager_execution()


# Function to plot the history of training- and validation Loss and Accuracy over training epochs
def plot_history(history):
    f, ax = plt.subplots(1, 1, figsize=(16, 7))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, marker='o', color='r', label='Training loss')
    plt.plot(epochs, val_loss, marker='o', color='b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# Define the Keras model
model = Sequential()

# Embedding Layer
model.add(tf.keras.layers.Embedding(vocab_size,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False))
# 1st LSTM Layer
model.add(LSTM(LSTM1_units, dropout=LSTM1_drop, recurrent_dropout=LSTM1_rec_drop, return_sequences=False,
               input_shape=(max_sequence_length, vocab_size)))
# 1st Dropout Layer
model.add(Dropout(Dropout1))

# 1st Dense Layer
model.add(Dense(units=Dense1_units, activation=Dense1_act))

# Output Layer
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer=optimizer, loss='mse')

# Give a summary of the model
model.summary()

# Defining the Learning Rate decay
ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1, min_lr=0.01, monitor='loss', verbose=1)

# Creating a checkpoint callback to export the best model of the training phase
checkpoint_filepath = os.path.join(os.path.join(model_dir, model_path), sys.argv[1] + '\\Checkpoint')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
     filepath=checkpoint_filepath,
     save_weights_only=False,
     monitor='val_loss',
     mode='min',
     save_best_only=True)

# Fetching features and labels for the training process
train_data_n = np.c_[x_train, y_train]
valid_data_n = np.c_[x_val, y_val]

# Training the model and storing the history
history = model.fit(train_data_n[:, :-1], train_data_n[:, -1], batch_size=batch_size, epochs=number_of_epochs,
                    verbose=verbosity_mode, shuffle=True, validation_data=(valid_data_n[:, :-1], valid_data_n[:, -1]),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5), model_checkpoint_callback,
                               ReduceLROnPlateau])

# Export a training-validation loss over epochs plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(os.path.join(os.path.join(model_dir, model_path), sys.argv[1] + '\\Training-Validation_Loss.png'))

# Test the model after training
y_estimate = model.predict(x_test)

for index, element in enumerate(y_estimate):
    element[0] = round(element[0])

print('Confusion Matrix')
print(confusion_matrix(y_test, y_estimate))
