import torch
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import yaml

def load_data(processed_data_path):
    tokenized_data = torch.load(os.path.join(processed_data_path, "tokenized_data.pt"))
    labels = torch.load(os.path.join(processed_data_path, "labels.pt"))
    dataset = Dataset.from_dict({'input_ids': tokenized_data['input_ids'], 'attention_mask': tokenized_data['attention_mask'], 'labels': labels})
    return dataset.train_test_split(test_size=0.1)

def evaluate_model(eval_dataset, model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(model=model)
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    accuracy = accuracy_score(eval_dataset['labels'], preds)
    f1 = f1_score(eval_dataset['labels'], preds, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    with open("../config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    datasets = load_data(config['data']['processed_data_path'])
    eval_dataset = datasets['test']
    evaluate_model(eval_dataset, config['model']['output_dir'])
