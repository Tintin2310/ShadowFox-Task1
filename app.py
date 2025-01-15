import streamlit as st
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict next words
def predict_next_word(model, tokenizer, text, max_sequence_length=20, top_k=3):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_length, padding='pre')
    predicted = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(predicted)[-top_k:][::-1]
    predicted_words = [(tokenizer.index_word.get(idx, ''), predicted[idx]) for idx in top_indices]
    return predicted_words

# Streamlit UI
st.title("Autocorrect Keyboard - Predict Next Word")
user_input = st.text_input("Enter text:", "")
max_sequence_length = st.slider("Max sequence length:", 5, 50, 20)

if user_input:
    predictions = predict_next_word(model, tokenizer, user_input, max_sequence_length)
    st.write("Top Predictions:")
    for word, prob in predictions:
        st.write(f"- **{word}** (Probability: {prob:.4f})")
