import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch

# Load the model and tokenizer
MODEL = "camembert-base"  # Use an actual model path if you have a fine-tuned model
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
set_background("https://your_image_url_here.jpg")

# Set up the Streamlit interface
st.title('French Sentence Difficulty Classifier')
sentence = st.text_input("Enter a sentence in French:", "")

if sentence:
    difficulty = predict_difficulty(sentence)
    st.write(f"The difficulty level of the sentence is: {difficulty}")
