import argparse
import os
from transformers import GPT2Tokenizer
import sys
# resolve path to src folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import prepare_dataset
from src.model import load_model, setup_training, train_model, save_model

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 for autocomplete")
    parser.add_argument("--dataset", type=str, default="webtext",
                       choices=["webtext", "gutenberg"],
                       help="Dataset to use for training")
    parser.add_argument("--output_dir", type=str, default="./models/webtext-model",
                       help="Directory to save the model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--max_samples", type=int, default=10000,
                       help="Maximum number of samples to use (None for all)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and prepare dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading and preprocessing {args.dataset} dataset...")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    
    tokenized_dataset = prepare_dataset(
        args.dataset, 
        tokenizer, 
        max_samples=args.max_samples
    )
    
    # Prepare training and evaluation datasets
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    
    # Load model
    print("Loading GPT-2 model...")
    model = load_model()
    
    # Setup training
    print("Setting up training...")
    trainer = setup_training(
        model, 
        train_dataset, 
        eval_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Train model
    print("Starting training...")
    train_model(trainer)
    
    # Save the final model
    print("Saving model...")
    save_model(model, tokenizer, args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()