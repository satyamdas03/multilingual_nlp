import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import yaml

def load_data(processed_data_path):
    tokenized_data = torch.load(os.path.join(processed_data_path, "tokenized_data.pt"))
    labels = torch.load(os.path.join(processed_data_path, "labels.pt"))
    dataset = Dataset.from_dict({'input_ids': tokenized_data['input_ids'], 'attention_mask': tokenized_data['attention_mask'], 'labels': labels})
    return dataset.train_test_split(test_size=0.1)

def train_model(train_dataset, eval_dataset, config):
    model = AutoModelForSequenceClassification.from_pretrained(config['model']['model_name'], num_labels=config['model']['num_labels'])
    training_args = TrainingArguments(
        output_dir=config['model']['output_dir'],
        evaluation_strategy="epoch",
        learning_rate=config['training']['learning_rate'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        num_train_epochs=config['training']['num_epochs'],
        weight_decay=config['training']['weight_decay'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config['model']['output_dir'])

if __name__ == "__main__":
    with open("../config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    datasets = load_data(config['data']['processed_data_path'])
    train_dataset = datasets['train']
    eval_dataset = datasets['test']
    train_model(train_dataset, eval_dataset, config)
