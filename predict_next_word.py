from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to predict the next word with probabilities
def predict_next_word(model, tokenizer, text, max_sequence_length=20, top_k=3):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])
    if not sequence or len(sequence[0]) == 0:
        return [("No valid tokens found", 0.0)]
    
    # Pad the sequence to the max length
    sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='pre')
    
    # Predict probabilities
    predicted = model.predict(sequence, verbose=0)[0]
    
    # Get top-k predictions
    top_indices = np.argsort(predicted)[-top_k:][::-1]
    predicted_words = [
        (tokenizer.index_word.get(idx, ''), predicted[idx])
        for idx in top_indices if tokenizer.index_word.get(idx, '') not in ['unk', '']
    ]
    
    # Ensure there are valid predictions
    if not predicted_words:
        return [("No valid prediction", 0.0)]
    
    return predicted_words

# Test prediction
text = "I am working on a"
predicted_words = predict_next_word(model, tokenizer, text)
print("Top Predictions:")
for word, prob in predicted_words:
    print(f"{word} (Probability: {prob:.4f})")
