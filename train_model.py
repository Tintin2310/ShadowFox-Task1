import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import pickle

# Load the training data (replace the filename if necessary)
with open('train.txt', 'r', encoding='utf-8') as file:
    train_text = file.read()

# Preprocessing function
def preprocess_text(text, max_sequence_length=20):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]
    
    predictors, labels = [], []
    for i in range(len(sequences) - max_sequence_length):
        predictors.append(sequences[i:i + max_sequence_length])
        labels.append(sequences[i + max_sequence_length])
    
    predictors = np.array(predictors)
    labels = np.array(labels)
    
    # Pad the sequences
    predictors = pad_sequences(predictors, maxlen=max_sequence_length, padding='pre')
    
    return predictors, labels, tokenizer, len(tokenizer.word_index) + 1, max_sequence_length

# Preprocess the training data
train_predictors, train_labels, tokenizer, total_words, max_sequence_length = preprocess_text(train_text)

# Create the RNN model
def create_rnn_model(total_words, max_sequence_length):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_length))  # Embedding layer
    model.add(LSTM(128, return_sequences=False))  # LSTM layer
    model.add(Dropout(0.2))  # Dropout to avoid overfitting
    model.add(Dense(total_words, activation='softmax'))  # Output layer
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

# Create and compile the model
model = create_rnn_model(total_words, max_sequence_length)

# Train the model
model.fit(train_predictors, train_labels, epochs=10, batch_size=64, verbose=1)

# Save the trained model and tokenizer
model.save('next_word_predictor.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully.")
