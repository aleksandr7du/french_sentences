import streamlit as st
import torch
import requests
from transformers import CamembertTokenizer, CamembertModel
import torch.nn as nn
import os

# Define the model architecture
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(self.camembert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

# Function to download the model from GitHub
def download_model():
    url = 'https://github.com/aleksandr7du/french_sentences/raw/main/models/trained_model.pth'
    response = requests.get(url)
    with open('trained_model.pth', 'wb') as f:
        f.write(response.content)

# Load the model without caching
def load_model():
    if not os.path.exists('trained_model.pth'):
        download_model()
    model = Net(num_classes=6)  # Adjust num_classes as per your model
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Instantiate the model and tokenizer
model = load_model()
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Function to make predictions
def predict(sentence):
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Streamlit application
st.title("French Sentence Difficulty Predictor")

input_sentence = st.text_area("Enter a French sentence:", "")

if st.button("Predict Difficulty"):
    if input_sentence:
        difficulty = predict(input_sentence)
        st.write(f"The predicted difficulty level is: {difficulty}")
    else:
        st.write("Please enter a sentence.")
