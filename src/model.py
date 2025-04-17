import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

def load_model(model_name="gpt2"):
    """Load pre-trained GPT-2 model. """
    return GPT2LMHeadModel.from_pretrained(model_name)

def setup_training(model, train_dataset, eval_dataset=None, 
                   output_dir="./models/webtext-model",
                   epochs=3, batch_size=8, tokenizer=None):
    """Setup training arguments and trainer."""
    from transformers import DataCollatorForLanguageModeling
    
    # Add data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling (use causal LM instead)
    )
    
    eval_steps = None
    if eval_dataset:
        evaluation_strategy = "steps"
        eval_steps = len(train_dataset) // batch_size
    else:
        evaluation_strategy = "no"
        
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        eval_steps=eval_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True if eval_dataset else False,
    )
    
    training_args.evaluation_strategy = evaluation_strategy  # Set it after initialization
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,  # Add the data collator for language modeling
    )
    
    return trainer

def train_model(trainer):
    """Train the model using the provided trainer."""
    return trainer.train()

def save_model(model, tokenizer, output_dir):
    """Save model and tokenizer to disk."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")