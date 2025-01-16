# Next Word Predictor  

The **Next Word Predictor** is an AI-driven application designed to assist users by predicting the next word based on their input text. Built using an LSTM-based neural network, this project focuses on improving the user experience with accurate predictions and a user-friendly interface powered by Streamlit.

---

## Features  

- **Accurate Predictions**: Provides top predictions with probabilities for the next word based on user input.  
- **Real-Time Interaction**: User-friendly Streamlit interface for live predictions.  
- **Customizable Prediction Depth**: Returns the top `k` predictions, giving users flexibility.  
- **Seamless Tokenization**: Handles text preprocessing to ensure smooth prediction workflows.  
- **User-Centric Approach**: Designed to avoid displaying placeholder tokens like "unk" for a better user experience.  

---

## Tech Stack  

- **Frontend**: Streamlit for a responsive and interactive UI.  
- **Backend**: Python with Keras and TensorFlow for the machine learning model.  
- **Model**: LSTM-based neural network trained on sequential text data.  

---

## Installation  

To set up the **Next Word Predictor** locally, follow these steps:  

1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/username/ShadowFox-Task1.git  
   cd ShadowFox-Task1

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3.**Run the Application**:
  ```bash
  streamlit run app.py
```
## Access the Application

Open your browser and go to [http://127.0.0.1:8501](http://127.0.0.1:8501).

---

## Usage

1. Launch the application in your browser.
2. Enter a partial sentence or phrase in the input box.
3. The application will display the top predictions with their probabilities.

---

## Scalability and Novelty

- **Enhanced Predictions**: Avoids placeholder outputs like "unk" for a refined user experience.
- **Interactive UI**: Streamlit ensures responsiveness and ease of use.
- **Expandable Dataset**: Train the model on new datasets to improve accuracy over time.
- **AI-Powered Insights**: Leverages machine learning for real-time text generation support.

---

## License

This project is open-sourced under the [MIT License](LICENSE).

---

Contributions are welcome! Feel free to fork the repository and enhance the project with your ideas. ✨


