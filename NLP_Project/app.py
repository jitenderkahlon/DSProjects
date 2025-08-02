import streamlit as st
import tensorflow as tf
import numpy as np
import re
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model, tokenizer, and label encoder
model = tf.keras.models.load_model('sentiment_model.h5')
tokenizer = load('tokenizer.joblib')
label_encoder = load('label_encoder.joblib')

# Text preprocessing
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", layout="centered")
st.title("🧠 Sentiment Analysis App")
st.write("Enter a sentence and let the model predict the sentiment.")

# User input
user_input = st.text_area("Text input", height=150)

# Predict
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=100, padding='post')
        prediction = model.predict(padded)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

        st.subheader("🔍 Prediction:")
        st.success(f"Sentiment: **{predicted_label}**")

        st.subheader("📊 Confidence Scores:")
        for i, score in enumerate(prediction[0]):
            st.write(f"{label_encoder.classes_[i]}: {score:.2%}")
