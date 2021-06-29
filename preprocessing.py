# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re
import nltk
from nltk.corpus import stopwords
import os
import time
import operator

# This script:
# 1) Imports the data set
# 2) Executes a series of preprocessing steps
# 3) Imports the Glove word embeddings to encode the vocabulary.
#   The file was too big to include in the repo.It can be downloaded from here: https://nlp.stanford.edu/projects/glove/
#   Make sure to download glove.42B.300d.zip and copy it to the 'Data' directory of the repository
# 4) Exports a series of csv files that will be used for the model training phase
#   x_train.csv, x_val.csv, x_test.csv, y_train_csv, y_val.csv, y_test.csv, embedding_matrix.csv

# Define the directory paths
root_dir = os.getcwd()
data_dir = os.path.join(root_dir, 'Data')

# Import the data
ssrs = pd.read_csv(os.path.join(data_dir, '500_Reddit_users_posts_labels.csv'))

# Preprocessing of the Reddit Posts
english_stemmer = nltk.stem.SnowballStemmer('english')


def post_to_wordlist(post, remove_stopwords=True):
    # Function to convert a post to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove non-letters
    post_text = re.sub("[^a-zA-Z]", " ", post)

    # 2. Convert words to lower case and split them
    words = post_text.lower().split()

    # 3. Optionally remove stop words (True by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    b = []
    stemmer = english_stemmer  # PorterStemmer()
    for word in words:
        b.append(stemmer.stem(word))

    # 4. Return a list of stemmed words
    return b


# Using the post_to_wordlist function iteratively on all posts
clean = []
for post in ssrs['Post']:
    clean.append(" ".join(post_to_wordlist(post)))

# Join the cleaned posts with the corresponding labels in a new DF
ssrs_c = pd.DataFrame(clean, columns=['Post']).join(ssrs['Label'])

# Split Dataset into Training and Test data to be stored in different files
training_data, test_data = train_test_split(ssrs_c[['Post', 'Label']], test_size=0.2, stratify=ssrs_c[['Label']],
                                            random_state=6)

# Split Training Data into Training set and Validation set to be stored in different files
train_data, val_data = train_test_split(training_data[['Post', 'Label']], test_size=0.2,
                                        stratify=training_data[['Label']], random_state=6)

# Fitting a tokenizer on the Training Set to index the known vocabulary
tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(train_data['Post'])

# Saving the size and the index of our vocabulary
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

# Not all posts have the same length. We need to address that through zero padding
MAX_SEQUENCE_LENGTH = max(pd.DataFrame(len(post.split()) for post in clean)[0])
print("Max Sequence Length :", MAX_SEQUENCE_LENGTH)

# All posts are translated to sequences of numbers through the Tokenizer that was trained earlier
# Additionally, zero padding is applied so that all posts have the same length, equal to MAX_SEQUENCE_LENGTH
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data['Post']), maxlen=MAX_SEQUENCE_LENGTH)
x_val = pad_sequences(tokenizer.texts_to_sequences(val_data['Post']), maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data['Post']), maxlen=MAX_SEQUENCE_LENGTH)

# Manual encoding of our labels due to their Ordinal Nature
y_train = np.where(train_data.Label == 'Supportive', 0,
                   np.where(train_data.Label == 'Indicator', 1,
                            np.where(train_data.Label == 'Ideation', 2,
                                     np.where(train_data.Label == 'Behavior', 3, 4))))
y_val = np.where(val_data.Label == 'Supportive', 0,
                 np.where(val_data.Label == 'Indicator', 1,
                          np.where(val_data.Label == 'Ideation', 2,
                                   np.where(val_data.Label == 'Behavior', 3, 4))))
y_test = np.where(test_data.Label == 'Supportive', 0,
                  np.where(test_data.Label == 'Indicator', 1,
                           np.where(test_data.Label == 'Ideation', 2,
                                    np.where(test_data.Label == 'Behavior', 3, 4))))
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Export the x_train, x_val, x_test, y_train, y_val, y_test matrices as csv files
pd.DataFrame(x_train).to_csv(os.path.join(data_dir, 'x_train.csv'), index=False)
pd.DataFrame(x_val).to_csv(os.path.join(data_dir, 'x_val.csv'), index=False)
pd.DataFrame(x_test).to_csv(os.path.join(data_dir, 'x_test.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
pd.DataFrame(y_val).to_csv(os.path.join(data_dir, 'y_val.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)

# Importing a set of pretrained word embeddings from the Glove project and tuning them to our vocabulary
embeddings_index = {}   # Dictionary to store the vocabulary and embeddings of Glove

tic = time.time()   # Time variable to check the import speed

# Reading the file from glove.
f = open(os.path.join(data_dir, 'glove.42B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Print the number of embeddings that were imported
print(f'loaded {len(embeddings_index)} word vectors from GloVe in {time.time()-tic} seconds')


# After importing the word embeddings, we want to see what part of our vocabulary is covered by GloVe
def check_coverage(vocab, embeddings_index):
    # This function compares a vocabulary against an word frequency dictionary to find the coverage
    # Prints the coverage and returns the frequency dictionary sorted in descending order
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of training vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all training text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def build_vocab(sentences):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


# Count the word frequencies in our vocabulary
vocab = build_vocab(list(train_data['Post'].apply(lambda x: x.split())))

# Calculating the coverage and sorting the word frequency dictionary
oov = check_coverage(vocab, embeddings_index)

# We now join the dictionary containing the embeddings from GloVe with our vocabulary to only keep the ones we need
EMBEDDING_DIM = 300   # This is the length of the embedding vectors provided by the specific GloVe file

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))   # This will be the matrix to store the final embeddings
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Export the embeddings for our vocabulary in a csv file
pd.DataFrame(embedding_matrix).to_csv(os.path.join(data_dir, 'embedding_matrix.csv'), index=False)
