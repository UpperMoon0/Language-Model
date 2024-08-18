import numpy as np
import pandas as pd
import pickle
from keras.src.saving import load_model
from keras.src.utils import pad_sequences

# Load the model
model = load_model('intent_recognition_model.keras')

# Load the tokenizer and the LabelEncoder
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pickle', 'rb') as f:
    le = pickle.load(f)


def predict_intent(p_sentence):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([p_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # assuming 100 was the max length used during training

    # Predict the intent
    prediction = model.predict(np.array(padded_sequence))

    # Decode the predicted intent
    predicted_intent = le.inverse_transform([np.argmax(prediction)])

    return predicted_intent[0]


# Keep asking for sentences and predicting intents
while True:
    sentence = input("Enter a sentence (or 'quit' to stop): ")
    if sentence.lower() == 'quit':
        break
    print("Predicted intent:", predict_intent(sentence))
