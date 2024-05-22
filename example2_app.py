import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import requests

# Load the model and tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForSequenceClassification.from_pretrained("camembert-base")

# Function to assess text difficulty
def assess_difficulty(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1).squeeze()
    labels = ['Easy', 'Intermediate', 'Hard']
    result_index = scores.argmax().item()
    return labels[result_index], "This is a placeholder for simplified text."

# Function to fetch word definitions
def get_definition(word):
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
    if response.status_code == 200:
        data = response.json()
        return data[0]['meanings'][0]['definitions'][0]['definition']
    else:
        return "Definition not found."

# Function to fetch pronunciation audio URL
def get_pronunciation(text):
    return "https://api.text-to-speech.com/pronounce?text=" + text

# Streamlit UI setup
st.title("French Sentence Difficulty Assessor")

# Text input
user_input = st.text_area("Enter a French sentence:", height=150)

if st.button("Analyze Text"):
    difficulty, simplified_sentence = assess_difficulty(user_input)
    st.write("### Original Text")
    st.write(user_input)
    st.write("### Difficulty Level")
    st.write(difficulty)
    st.write("### Simplified Sentence")
    st.write(simplified_sentence)

    # Vocabulary and definitions
    words = user_input.split()
    st.write("### Vocabulary")
    for word in words:
        with st.expander(word):
            definition = get_definition(word)
            st.write(definition)

    # Audio pronunciation
    audio_url = get_pronunciation(user_input)
    st.audio(audio_url)

    # Quiz section (dummy quiz for demonstration)
    st.write("### Quiz")
    if difficulty == "Intermediate":
        quiz_question = f"What does the word '{words[0]}' mean?"
        user_answer = st.text_input("Answer:", key="quiz1")
        correct_answer = get_definition(words[0])
        if st.button("Check Answer"):
            if user_answer.lower() == correct_answer.lower():
                st.success("Correct!")
            else:
                st.error("Incorrect. The correct answer is: " + correct_answer)

# Run the script using: streamlit run your_script_name.py
