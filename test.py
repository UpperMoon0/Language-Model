import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.saving import load_model
from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('intent_recognition_model.keras')

# Load the data
data = pd.read_csv('datasets/data.csv', comment='#')

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['sentence'])

# Encode the intents
le = LabelEncoder()
le.fit(data['intent'])


def predict_intent(p_sentence):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([p_sentence])
    padded_sequence = pad_sequences(sequence)

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
