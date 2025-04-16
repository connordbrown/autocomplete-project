import argparse
import json
from transformers import GPT2Tokenizer
from src.data_utils import prepare_dataset
from src.model import load_model
from src.autocomplete import AutocompleteGenerator
from src.evaluation import AutocompleteEvaluator
from src.error_analysis import ErrorAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 autocomplete model")
    parser.add_argument("--model_path", type=str, default="./models/webtext-model",
                       help="Path to the trained model")
    parser.add_argument("--dataset", type=str, default="webtext",
                       choices=["webtext", "gutenberg"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--output_file", type=str, default="./evaluation_results.json",
                       help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    
    # Load dataset
    print(f"Loading evaluation dataset ({args.dataset})...")
    dataset = prepare_dataset(args.dataset, tokenizer)
    test_data = dataset["test"]["text"][:args.num_samples] if "test" in dataset else dataset["train"]["text"][-args.num_samples:]
    
    # Create evaluator and generator
    generator = AutocompleteGenerator(model, tokenizer)
    evaluator = AutocompleteEvaluator(model, tokenizer)
    error_analyzer = ErrorAnalyzer()
    
    # Run evaluation
    print("Running evaluation...")
    evaluation_results = evaluator.evaluate(test_data, autocomplete_generator=generator)
    
    # Analyze errors
    print("Categorizing prediction errors...")
    full_results = error_analyzer.analyze_errors(evaluation_results)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Perplexity: {full_results['perplexity']:.2f}")
    print(f"Word-level Accuracy: {full_results['word_accuracy']:.4f}")
    print(f"Character-level Accuracy: {full_results['char_accuracy']:.4f}")
    print(f"BLEU Score: {full_results['bleu_score']:.4f}")
    print(f"ROUGE-1 F1: {full_results['rouge_scores']['rouge-1']['f']:.4f}")
    
    print("\nError Analysis:")
    print(f"Semantic Errors: {full_results['error_analysis']['semantic_errors']}")
    print(f"Grammatical Errors: {full_results['error_analysis']['grammatical_errors']}")
    print(f"Style Inconsistencies: {full_results['error_analysis']['style_inconsistencies']}")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        # Create a serializable version of the results
        serializable_results = {
            "perplexity": float(full_results['perplexity']),
            "word_accuracy": float(full_results['word_accuracy']),
            "char_accuracy": float(full_results['char_accuracy']),
            "bleu_score": float(full_results['bleu_score']),
            "rouge_scores": full_results['rouge_scores'],
            "error_analysis": full_results['error_analysis']
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()