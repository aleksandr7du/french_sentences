import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

# Load the model and tokenizer
MODEL = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(MODEL)
model = CamembertForSequenceClassification.from_pretrained(MODEL, num_labels=6)

def predict_difficulty(sentence):
    # Tokenize the text
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1)
    difficulty_index = torch.argmax(predictions).item()
    
    # Map model output to CEFR levels
    levels = {0: "A1", 1: "A2", 2: "B1", 3: "B2", 4: "C1", 5: "C2"}
    return levels.get(difficulty_index, "Unknown")

# Custom background set-up
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
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
    options = ["A1", "A2", "B1", "B2", "C1", "C2"]
    user_choice = st.selectbox("Choose the expected difficulty level:", options)

    # Only predict and display results after the user has made their choice
    if user_choice:
        difficulty = predict_difficulty(sentence)
        if user_choice == difficulty:
            st.success("Well done! You have a good intuition!")
        else:
            st.error("Pas de soucis: mistakes are part of success!")
        st.write(f"Predicted difficulty level of the sentence: {difficulty}")
