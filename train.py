import pandas as pd
from keras import Sequential
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.regularizers import regularizers
from keras.src.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the data
data = pd.read_csv('datasets/data.csv', comment='#')

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['sentence'])
sequences = tokenizer.texts_to_sequences(data['sentence'])
x = pad_sequences(sequences)

# Encode the intents
le = LabelEncoder()
y = le.fit_transform(data['intent'])
y = to_categorical(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
model.add(LSTM(64, kernel_regularizer=regularizers.L2(0.01)))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=60)

# Save the model
model.save('intent_recognition_model.keras')
