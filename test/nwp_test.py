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
    predicted_word = tokenizer.index_word[predicted_word_index]
    predicted_probability = predicted[0][predicted_word_index]
    return predicted_word, predicted_probability, text + ' ' + predicted_word


# Function to predict a sentence up to 10 words or until a period is predicted or probability falls below threshold
def predict_sentence(model, tokenizer, text, max_sequence_len, probability_threshold=0.1):
    while len(text.split()) < 10:
        next_word, probability, text = predict_next_word(model, tokenizer, text, max_sequence_len)
        if next_word == '.' or probability < probability_threshold:
            text += next_word if next_word == '.' else ''
            break
    return text


# Mode variable
mode = 'sentence'  # Change to 'sentence' for self-feeding mode

# Loop to get user input and predict the next word or sentence
while True:
    seed_text = input("Enter a word or sentence (or type 'exit' to stop): ")
    if seed_text.lower() == 'exit':
        break
    if mode == 'word':
        next_word, _, _ = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
        print("Next word prediction:", next_word)
    elif mode == 'sentence':
        sentence = predict_sentence(model, tokenizer, seed_text, max_sequence_len)
        print("Predicted sentence:", sentence)
