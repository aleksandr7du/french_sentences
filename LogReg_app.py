import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Load your pre-trained logistic regression model and label encoder
model_filename = 'best_logistic_regression_model.joblib'
label_encoder_filename = 'label_encoder.joblib'

model = joblib.load(model_filename)
label_encoder = joblib.load(label_encoder_filename)

# Predefined article links based on difficulty level and interests
article_links = {
    "A1": {
        "Sports": "https://example.com/a1-sports",
        "Technology": "https://example.com/a1-technology",
        # Add more links for A1 level
    },
    "A2": {
        "Sports": "https://example.com/a2-sports",
        "Technology": "https://example.com/a2-technology",
        # Add more links for A2 level
    },
    # Add more difficulty levels and corresponding links
}

def predict_difficulty(sentence):
    # Transform the sentence using the TF-IDF vectorizer in the pipeline
    prediction = model.predict([sentence])
    difficulty = label_encoder.inverse_transform(prediction)
    return difficulty[0]

# Custom background set-up with enhanced readability
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: #000000;
        }}
        .css-2trqyj {{
            padding: 5px;
            background-color: rgba(255, 255, 255, 0.8);
            border: 1px solid #f5f5dc;
        }}
        .css-2trqyj textarea {{
            background-color: #ffffff;
            border: 1px solid #f5f5dc;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background to a Paris view
set_background("https://all.accor.com/magazine/imagerie/1-c4c1.jpg")

# Set up the Streamlit interface
st.title('French Sentence Difficulty Classifier')
sentence = st.text_input("Enter a sentence in French:")

if sentence:
    interests = ["Sports", "Technology", "Travel", "Food", "Health", "Science", "Education", "Culture", "Politics", "Finance"]
    user_interest = st.selectbox("Choose your interest:", interests)

    if user_interest:
        difficulty = predict_difficulty(sentence)
        st.write(f"Predicted difficulty level of the sentence: {difficulty}")

        if user_interest in article_links[difficulty]:
            article_link = article_links[difficulty][user_interest]
            st.write(f"Here is an article for you: [Read here]({article_link})")

# Newsletter subscription
st.write("If you like the app, subscribe to our newsletter!")
email = st.text_input("Enter your email:")
if st.button("Subscribe"):
    st.write("Thank you for subscribing!")
    st.button("Subscribe", disabled=True)
