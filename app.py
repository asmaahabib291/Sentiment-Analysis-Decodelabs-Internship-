import streamlit as st
import joblib

# Page config (important for better UI)
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

# Load model & vectorizer safely
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# Title
st.title("💬 Sentiment Analysis Dashboard")
st.write("Type a sentence and the model will predict its sentiment")

# Input
text = st.text_area("Enter your text:")

# Button
if st.button("Analyze"):

    if text and text.strip():

        # Transform text
        text_vec = vectorizer.transform([text])

        # Prediction
        prediction = model.predict(text_vec)[0]

        # Result
        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")

    else:
        st.warning("Please enter some text first")