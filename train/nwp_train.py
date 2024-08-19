import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# Flag to reuse saved model
reuse_saved_model = True

# Paths to saved model and tokenizer
model_path = '../models/nwp/next_word_prediction_model.keras'
tokenizer_path = '../models/nwp/tokenizer.pickle'
max_sequence_len_path = '../models/nwp/max_sequence_len.pickle'

# Learning rate parameter
learning_rate = 0.001

# Load the corpus from CSV file
corpus_df = pd.read_csv('../datasets/nwp/corpus.csv')
corpus = corpus_df['text'].tolist()

corpus = [text for text in corpus]

# Initialize model, tokenizer, and max_sequence_len variables
model = None
tokenizer = None
max_sequence_len = None

# Load or create tokenizer and max_sequence_len
if reuse_saved_model and os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(
        max_sequence_len_path):
    # Load the model
    model = load_model(model_path)

    # Load tokenizer and max_sequence_len
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(max_sequence_len_path, 'rb') as handle:
        max_sequence_len = pickle.load(handle)

    # Recompile the model with the new learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
else:
    # Tokenize the text
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)

# Whether new or old tokenizer, we still need to calculate X and y for training
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
if max_sequence_len is None:
    max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split data into features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model if not reusing
if model is None:
    model = Sequential()
    model.add(Embedding(total_words, 100))
    model.add(Dropout(0.2))
    model.add(LSTM(150))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, verbose=1)

# Save tokenizer word index
with open('../log/tokenizer_word_index.txt', 'w') as file:
    for word, index in tokenizer.word_index.items():
        file.write(f'{word} - {index}\n')

# Save the model
model.save(model_path)

# Save tokenizer and max_sequence_len
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(max_sequence_len_path, 'wb') as handle:
    pickle.dump(max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)
