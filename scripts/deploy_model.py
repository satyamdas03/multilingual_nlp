from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

app = Flask(__name__)

with open("../config.yaml", 'r') as file:
    config = yaml.safe_load(file)

model = AutoModelForSequenceClassification.from_pretrained(config['model']['output_dir'])
tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = tokenizer(data['text'], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({'prediction': predictions})

if __name__ == '__main__':
    app.run(debug=True)
