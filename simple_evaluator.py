import argparse
import json
import os
import random
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

# Ensure NLTK dependencies are downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class SimpleAutocompleteTester:
    def __init__(self, model_path):
        """Initialize with a model path."""
        print(f"Loading model from {model_path}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        
    def load_data(self, dataset_name, num_samples=100):
        """Load a small sample of data for testing."""
        print(f"Loading small sample of {dataset_name} dataset...")
        
        if dataset_name == "webtext":
            from datasets import load_dataset
            dataset = load_dataset("stas/openwebtext-10k", split="train")
            texts = [sample["text"] for sample in dataset]
        elif dataset_name == "gutenberg":
            from datasets import load_dataset
            dataset = load_dataset("bookcorpusopen", split="train")
            texts = [sample["text"] for sample in dataset]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Take a random sample to reduce memory usage
        if len(texts) > num_samples:
            random.seed(42)  # For reproducibility
            texts = random.sample(texts, num_samples)
            
        # Further filter texts that are too long
        filtered_texts = []
        for text in texts:
            if isinstance(text, str) and len(text) > 20:  # Ensure text is valid
                # Truncate very long texts
                filtered_texts.append(text[:1000])
            
            if len(filtered_texts) >= num_samples:
                break
                
        return filtered_texts
    
    def generate_completion(self, prefix, max_tokens=5):
        """Generate a simple completion for a given prefix."""
        input_ids = self.tokenizer.encode(prefix, return_tensors="pt")
        
        # If the model would run out of memory, return an empty string
        if input_ids.shape[1] > 100:  # Skip very long inputs
            return ""
        
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.7,
                )
            
            # Decode the completion
            completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the new tokens (not the prefix)
            if completion.startswith(prefix):
                return completion[len(prefix):]
            else:
                return completion
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""
    
    def calculate_perplexity(self, text, stride=50, max_tokens=100):
        """Calculate perplexity with memory optimization."""
        # Truncate text to prevent memory issues
        text = text[:500]  # Limit to 500 chars max
        
        try:
            # Encode the text
            encodings = self.tokenizer(text, 
                                      return_tensors="pt",
                                      truncation=True, 
                                      max_length=max_tokens)
            
            seq_len = encodings.input_ids.size(1)
            if seq_len <= 1:  # Skip very short texts
                return float('inf')
                
            log_likelihood = 0
            total_tokens = 0
            
            # Process in smaller chunks
            for i in range(0, seq_len, stride):
                end = min(i + stride, seq_len)
                chunk_len = end - i
                
                # Skip the last chunk if it's too small
                if chunk_len <= 1:
                    continue
                    
                input_ids = encodings.input_ids[:, i:end]
                labels = input_ids.clone()
                
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=labels)
                    log_likelihood += outputs.loss.item() * chunk_len
                    total_tokens += chunk_len
                    
            # Calculate perplexity
            if total_tokens > 0:
                return np.exp(log_likelihood / total_tokens)
            else:
                return float('inf')
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')
    
    def calculate_accuracy(self, pred, ref, word_level=True):
        """Calculate accuracy at word or character level."""
        if not pred or not ref:
            return 0
            
        if word_level:
            pred_words = pred.split()
            ref_words = ref.split()
            
            min_len = min(len(pred_words), len(ref_words))
            if min_len == 0:
                return 0
                
            correct = sum(1 for i in range(min_len) if pred_words[i] == ref_words[i])
            return correct / len(ref_words) if ref_words else 0
        else:
            # Character level
            min_len = min(len(pred), len(ref))
            if min_len == 0:
                return 0
                
            correct = sum(1 for i in range(min_len) if pred[i] == ref[i])
            return correct / len(ref) if ref else 0
    
    def calculate_bleu(self, ref, pred):
        """Calculate BLEU score."""
        if not pred or not ref:
            return 0
            
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        
        if not pred_tokens:
            return 0
            
        return sentence_bleu(ref_tokens, pred_tokens)
    
    def calculate_rouge(self, ref, pred):
        """Calculate ROUGE-1 F1 score."""
        if not pred or not ref:
            return 0
            
        try:
            scores = self.rouge_scorer.score(ref, pred)
            return scores['rouge1'].fmeasure
        except Exception:
            return 0
    
    def analyze_formality(self, text):
        """Simple formality analysis."""
        if not text:
            return 0.5
            
        formal_words = ["shall", "would", "could", "may", "therefore"]
        informal_words = ["gonna", "wanna", "gotta", "kinda", "yeah"]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_words if word in text_lower)
        informal_count = sum(1 for word in informal_words if word in text_lower)
        
        if formal_count + informal_count == 0:
            return 0.5
            
        return formal_count / (formal_count + informal_count)
    
    def categorize_error(self, pred, ref):
        """Simplified error categorization."""
        if not pred or not ref:
            return None
            
        # Style check
        pred_formality = self.analyze_formality(pred)
        ref_formality = self.analyze_formality(ref)
        
        if abs(pred_formality - ref_formality) > 0.5:
            return "style"
            
        # Simple grammatical check
        try:
            pred_tokens = nltk.word_tokenize(pred)
            if pred_tokens:
                pos_tags = nltk.pos_tag(pred_tokens)
                first_pos = pos_tags[0][1]
                if first_pos not in ['VB', 'NN', 'PRP', 'JJ', 'DT', 'IN']:
                    return "grammatical"
        except Exception:
            pass
            
        # Default to semantic error
        return "semantic"
    
    def evaluate(self, num_samples=100, prefix_length=10):
        """Run a simplified evaluation."""
        # Load a small sample of test data
        texts = self.load_data("webtext", num_samples)
        
        results = {
            "perplexity": 0,
            "word_accuracy": 0,
            "char_accuracy": 0,
            "bleu_score": 0,
            "rouge_f1": 0,
            "semantic_errors": 0,
            "grammatical_errors": 0,
            "style_inconsistencies": 0,
            "examples": []
        }
        
        print(f"Evaluating on {len(texts)} samples...")
        valid_samples = 0
        perplexity_samples = 0
        
        # Calculate average perplexity on a subset for efficiency
        perplexity_texts = texts[:min(10, len(texts))]
        perplexity_sum = 0
        
        for text in perplexity_texts:
            ppl = self.calculate_perplexity(text)
            if ppl != float('inf'):
                perplexity_sum += ppl
                perplexity_samples += 1
                
        if perplexity_samples > 0:
            results["perplexity"] = perplexity_sum / perplexity_samples
        
        # Evaluate completions
        for text in texts:
            if len(text) < prefix_length + 10:
                continue
                
            # Split text into prefix and reference
            prefix = text[:prefix_length]
            reference = text[prefix_length:prefix_length+20]  # Limit reference length
            
            # Generate prediction
            prediction = self.generate_completion(prefix)
            
            # Skip empty predictions
            if not prediction:
                continue
            
            # Save example
            if len(results["examples"]) < 5:
                results["examples"].append({
                    "prefix": prefix,
                    "prediction": prediction,
                    "reference": reference
                })
            
            # Calculate metrics
            word_acc = self.calculate_accuracy(prediction, reference, word_level=True)
            char_acc = self.calculate_accuracy(prediction, reference, word_level=False)
            bleu = self.calculate_bleu(reference, prediction)
            rouge = self.calculate_rouge(reference, prediction)
            
            results["word_accuracy"] += word_acc
            results["char_accuracy"] += char_acc
            results["bleu_score"] += bleu
            results["rouge_f1"] += rouge
            
            # Categorize errors
            error_type = self.categorize_error(prediction, reference)
            if error_type == "semantic":
                results["semantic_errors"] += 1
            elif error_type == "grammatical":
                results["grammatical_errors"] += 1
            elif error_type == "style":
                results["style_inconsistencies"] += 1
            
            valid_samples += 1
            
            # Print progress
            if valid_samples % 10 == 0:
                print(f"Processed {valid_samples} valid samples...")
        
        # Calculate averages
        if valid_samples > 0:
            results["word_accuracy"] /= valid_samples
            results["char_accuracy"] /= valid_samples
            results["bleu_score"] /= valid_samples
            results["rouge_f1"] /= valid_samples
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Simple Autocomplete Evaluation")
    parser.add_argument("--model_path", type=str, default="./models/webtext-model",
                       help="Path to the trained model")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for evaluation")
    parser.add_argument("--output_file", type=str, default="./simple_evaluation_results.json",
                       help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SimpleAutocompleteTester(args.model_path)
    
    # Run evaluation with smaller sample size
    results = evaluator.evaluate(num_samples=args.num_samples)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Word-level Accuracy: {results['word_accuracy']:.4f}")
    print(f"Character-level Accuracy: {results['char_accuracy']:.4f}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"ROUGE-1 F1: {results['rouge_f1']:.4f}")
    
    print("\nError Analysis:")
    print(f"Semantic Errors: {results['semantic_errors']}")
    print(f"Grammatical Errors: {results['grammatical_errors']}")
    print(f"Style Inconsistencies: {results['style_inconsistencies']}")
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()