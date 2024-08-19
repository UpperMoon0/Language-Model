import pickle
import pyarrow.ipc as ipc
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical

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
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)

# Pad the sequences
max_len = max(len(seq) for seq in x_train)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
x_val = pad_sequences(x_val, maxlen=max_len, padding='post')

# Encode the intents
with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

y_train = to_categorical(le.transform(y_train))
y_val = to_categorical(le.transform(y_val))

# Continue training a pretrained model or start a new one
continue_training = False

if continue_training:
    # Load the pretrained model
    model = tf.keras.models.load_model('intent_recognition_model.keras')
else:
    # Define a new model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
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
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('intent_recognition_model.keras')
