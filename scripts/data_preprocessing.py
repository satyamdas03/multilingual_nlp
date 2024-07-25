import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer
import yaml

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_texts(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def load_and_preprocess_data(filepath, tokenizer):
    df = pd.read_csv(filepath)
    df['text'] = df['text'].apply(preprocess_text)
    tokenized_data = tokenize_texts(df['text'].tolist(), tokenizer)
    return tokenized_data, df['label'].tolist()

if __name__ == "__main__":
    with open("../config.yaml", 'r') as file:
        config = yaml.safe_load(file)
        
    data_path = config['data']['raw_data_path']
    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    tokenized_data, labels = load_and_preprocess_data(data_path, tokenizer)
    
    processed_data_path = config['data']['processed_data_path']
    torch.save(tokenized_data, os.path.join(processed_data_path, "tokenized_data.pt"))
    torch.save(labels, os.path.join(processed_data_path, "labels.pt"))
