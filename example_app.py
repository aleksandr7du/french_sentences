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

# Custom background set-up with enhanced readability
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-position: center;
            color: #000000;  /* Sets the text color to black */
        }}
        .css-2trqyj {{
            padding: 5px;  /* More padding around text input fields */
            background-color: rgba(255, 255, 255, 0.8);  /* Slightly transparent background for inputs */
            border: 1px solid #f5f5dc;  /* Beige border around the input field */
        }}
        .css-2trqyj textarea {{
            background-color: #ffffff;  /* White background for the textarea */
            border: 1px solid #f5f5dc;  /* Beige border for the textarea */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background to a Paris view
set_background("https://github.com/aleksandr7du/french_sentences/blob/main/background_streamlit_french_sentences.jpg")

# Set up the Streamlit interface
st.title('French Sentence Difficulty Classifier')
sentence = st.text_input("Enter a sentence in French:")

if sentence:
    options = ["Select a difficulty level", "A1", "A2", "B1", "B2", "C1", "C2"]
    user_choice = st.selectbox("Choose the expected difficulty level:", options, index=0)

    if user_choice != "Select a difficulty level":
        difficulty = predict_difficulty(sentence)
        st.write(f"Predicted difficulty level of the sentence: {difficulty}")

        if user_choice == difficulty:
            st.success("Well done! You have a good intuition!")
        else:
            st.error("Pas de soucis: mistakes are part of success!")

# Newsletter subscription
st.write("If you like the app, subscribe to our newsletter!")
email = st.text_input("Enter your email:")
if st.button("Subscribe"):
    st.write("Thank you for subscribing!")
    st.button("Subscribe", disabled=True)
