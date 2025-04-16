import argparse
from transformers import GPT2Tokenizer
from src.model import load_model
from src.autocomplete import AutocompleteGenerator

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Autocomplete Demo")
    parser.add_argument("--model_path", type=str, default="./models/webtext-model",
                        help="Path to the trained model")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model = load_model(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    
    # Create autocomplete generator
    generator = AutocompleteGenerator(model, tokenizer)
    
    print("GPT-2 Autocomplete System")
    print("Type a sentence prefix and press Enter to see completions. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour text: ")
        if user_input.lower() == 'exit':
            break
            
        completions = generator.generate_completions(user_input)
        print("\nAutocomplete suggestions:")
        for i, completion in enumerate(completions, 1):
            print(f"{i}. {user_input}{completion}")

if __name__ == "__main__":
    main()