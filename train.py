import pyarrow.ipc as ipc
from keras import Sequential
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.regularizers import regularizers
from keras.src.utils import pad_sequences, to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Load the data
with open('./datasets/train/data-00000-of-00001.arrow', 'rb') as f:
    reader = ipc.RecordBatchStreamReader(f)
    train_data = reader.read_all().to_pandas()

with open('./datasets/validation/data-00000-of-00001.arrow', 'rb') as f:
    reader = ipc.RecordBatchStreamReader(f)
    validation_data = reader.read_all().to_pandas()

# Extract the 'sentence' and 'intent' columns
x_train = train_data['text']
y_train = train_data['label']
x_val = validation_data['text']
y_val = validation_data['label']

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = pad_sequences(tokenizer.texts_to_sequences(x_train))
x_val = pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=x_train.shape[1])

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Encode the intents
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train))
y_val = to_categorical(le.transform(y_val))

# Save the LabelEncoder
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))
model.add(LSTM(64, kernel_regularizer=regularizers.L2(0.01)))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=15)

# Save the model
model.save('intent_recognition_model.keras')
