import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Sample corpus
corpus = [
    "I like machine learning",
    "I love natural language processing",
    "Deep learning is fascinating",
    "Natural language processing is a powerful tool"
]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split data into features and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = Sequential()
model.add(Embedding(total_words, 100))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Construct the path to save the model in the parent directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "next_word_prediction_model.keras")

# Save the model
model.save(model_path)

# Save tokenizer and max_sequence_len
with open('../models/nwp/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../models/nwp/max_sequence_len.pickle', 'wb') as handle:
    pickle.dump(max_sequence_len, handle, protocol=pickle.HIGHEST_PROTOCOL)