from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def load_tokenizer(model_name="bert-base-multilingual-cased"):
    return AutoTokenizer.from_pretrained(model_name)

def load_model(model_name="xlm-roberta-base", num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def train_model(train_dataset, eval_dataset, model_name="xlm-roberta-base", output_dir="../models"):
    model = load_model(model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(f"{output_dir}/final_model")
    return model, trainer
