import pickle

import numpy as np
import tensorflow as tf
from keras.src.utils import pad_sequences

# Load the model
model = tf.keras.models.load_model('../models/nwp/next_word_prediction_model.keras')

# Load tokenizer and max_sequence_len
with open('../models/nwp/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('../models/nwp/max_sequence_len.pickle', 'rb') as handle:
    max_sequence_len = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    return tokenizer.index_word[predicted_word_index]


# Loop to get user input and predict the next word
while True:
    seed_text = input("Enter a word or sentence (or type 'exit' to stop): ")
    if seed_text.lower() == 'exit':
        break
    next_word = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
    print("Next word prediction:", next_word)
