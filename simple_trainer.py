import argparse
import os
import torch
import gc
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

def get_dataset_loader(dataset_name, max_samples=5000, batch_size=4):
    """Load dataset in smaller chunks to reduce memory usage."""
    if dataset_name.lower() == "webtext":
        dataset = load_dataset("stas/openwebtext-10k", split="train")
    elif dataset_name.lower() == "gutenberg":
        dataset = load_dataset("bookcorpusopen", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit dataset size
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    # Create a simple loader function that will yield batches
    def data_loader():
        for i in range(0, len(dataset), batch_size):
            yield dataset[i:i+batch_size]
    
    return data_loader(), len(dataset)

def tokenize_batch(batch, tokenizer, max_length=128):
    """Tokenize a batch of texts."""
    inputs = tokenizer(
        batch["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length,
        return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

def train_in_batches(model, tokenizer, data_loader, total_samples, 
                    output_dir, epochs=1, batch_size=4, 
                    learning_rate=5e-5, gradient_accumulation_steps=4):
    """Train the model in batches with manual memory management."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    steps_per_epoch = total_samples // batch_size
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        
        running_loss = 0.0
        step = 0
        optimizer.zero_grad()
        
        for batch_data in data_loader:
            # Tokenize batch
            inputs = tokenize_batch(batch_data, tokenizer)
            
            # Move tensors to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            labels = inputs["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            running_loss += loss.item() * gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Print progress
                if (step + 1) % (gradient_accumulation_steps * 10) == 0:
                    avg_loss = running_loss / (gradient_accumulation_steps * 10)
                    print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}, Loss: {avg_loss:.4f}")
                    running_loss = 0.0
            
            # Free up memory
            del inputs, input_ids, attention_mask, labels, outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            step += 1
        
        # Save model checkpoint after each epoch
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return model

def train_with_trainer(model, train_dataset, tokenizer, output_dir, epochs=1, batch_size=4):
    """Alternative method using the Trainer API with memory optimizations."""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling (use causal LM instead)
    )
    
    # Set up memory-efficient training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if GPU available
        gradient_checkpointing=True,  # Memory optimization
        optim="adamw_torch",
        report_to="none",  # Disable reporting to save memory
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient GPT-2 Training")
    parser.add_argument("--dataset", type=str, default="webtext",
                       choices=["webtext", "gutenberg"],
                       help="Dataset to use for training")
    parser.add_argument("--output_dir", type=str, default="./models/webtext-model",
                       help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size (smaller for memory efficiency)")
    parser.add_argument("--max_samples", type=int, default=2000,
                       help="Maximum number of samples to use")
    parser.add_argument("--use_trainer", action="store_true",
                       help="Use HuggingFace Trainer API (otherwise use custom training loop)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    print("Loading GPT-2 model with memory optimizations...")
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", 
        use_cache=False  # Disable past key values cache to save memory
    )
    
    print(f"Loading {args.dataset} dataset with max_samples={args.max_samples}...")
    
    if args.use_trainer:
        # Prepare dataset for trainer API
        print("Preparing dataset for Trainer API...")
        
        # Load dataset
        if args.dataset.lower() == "webtext":
            dataset = load_dataset("stas/openwebtext-10k", split="train")
        else:
            dataset = load_dataset("bookcorpusopen", split="train")
            
        # Limit dataset size
        if args.max_samples and args.max_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(args.max_samples))
        
        # Process dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=128,
                return_tensors=None
            )
        
        # Process in smaller batches to save memory
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=["text"],
        )
        
        # Set labels for language modeling
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
        )
        
        print("Starting training with Trainer API...")
        train_with_trainer(
            model, 
            tokenized_dataset, 
            tokenizer, 
            args.output_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
    else:
        # Use custom training loop
        print("Using custom training loop...")
        data_loader, total_samples = get_dataset_loader(
            args.dataset, 
            args.max_samples, 
            args.batch_size
        )
        
        print("Starting training...")
        train_in_batches(
            model, 
            tokenizer, 
            data_loader, 
            total_samples, 
            args.output_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
    
    print("Training complete!")

if __name__ == "__main__":
    main()