from flask import Flask, request, jsonify
import numpy as np  # Example library for data processing
import torch
import torch.nn as nn
import json
from sentence_transformers import SentenceTransformer
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters, punctuation, and numbers using regex
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the preprocessed words back into a string
    processed_text = ' '.join(words)
    
    return processed_text

class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SiameseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward_once(self, x):
        
        h0 = torch.zeros(1, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm1(x, (h0, c0))  # Output of first LSTM layer
        out, _ = self.lstm2(out)  # Output of second LSTM layer
        
        # Decode the hidden state of the last time step
        if isinstance(_, tuple):
            hidden_state, cell_state = _
        else:
            hidden_state = _
        out = self.fc(hidden_state.squeeze(0))

        return out

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2


def compute_similarity_scores_json(model, sia_model, data):
    similarity_score = 0
    with torch.no_grad():
        
        text1 = data['text1']
        # print(text1)
        text2 = data['text2']
        # print(text2)

        embeddings1 = model.encode(text1)
        embeddings2 = model.encode(text2)
        output1, output2 = sia_model(torch.tensor(embeddings1).unsqueeze(0), torch.tensor(embeddings2).unsqueeze(0))
        

        similarity_score = torch.sigmoid(output1 - output2).item()  # Sigmoid to get similarity score between 0 and 1
            
    return similarity_score


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()  # Get input data from JSON request
        
        json_data['text1'] = preprocess_text(json_data['text1'])
        json_data['text2'] = preprocess_text(json_data['text2'])

        input_size = model.get_sentence_embedding_dimension()
        hidden_size = 2 * input_size
        output_size = 1
        num_layers = 1

        model_path = 'siamese_lstm2_chkp40.pth'
        model_state_dict = torch.load(model_path, map_location='cpu')

        sia_model = SiameseLSTM(input_size, hidden_size, num_layers, output_size)
        sia_model.load_state_dict(model_state_dict)
        sia_model.eval()

        prediction = compute_similarity_scores_json(model, sia_model, json_data)  # Function to predict a number
        
        input_size = model.get_sentence_embedding_dimension()
        hidden_size = 2 * input_size
        output_size = 1
        num_layers = 1


        response = {'similarity score': prediction}
        return jsonify(response)
    else:
        response = {'similarity score': prediction}
        return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
