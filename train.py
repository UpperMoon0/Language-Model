import pyarrow.ipc as ipc
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
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

# Tokenize the sentences using Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)

# Pad the sequences
max_len = max(len(seq) for seq in x_train)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_val = pad_sequences(x_val, maxlen=max_len, padding='post')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Encode the intents
le = LabelEncoder()
y_train = tf.keras.utils.to_categorical(le.fit_transform(y_train))
y_val = tf.keras.utils.to_categorical(le.transform(y_val))

# Save the LabelEncoder
with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the model
model.save('intent_recognition_model')
