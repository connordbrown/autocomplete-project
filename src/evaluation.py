import torch
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

class AutocompleteEvaluator:
    def __init__(self, model, tokenizer):
        """Initialize with a model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        
    def calculate_perplexity(self, text):
        """Calculate perplexity for a given text."""
        encodings = self.tokenizer(text, return_tensors="pt")
        max_length = self.model.config.n_positions
        stride = 512
        seq_len = encodings.input_ids.size(1)
        
        nlls = []
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                
            nlls.append(neg_log_likelihood)
            
        return torch.exp(torch.stack(nlls).sum() / end_loc).item()
    
    def calculate_accuracy(self, predictions, references, word_level=True):
        """Calculate prediction accuracy at word or character level."""
        if word_level:
            pred_words = [p.split() for p in predictions]
            ref_words = [r.split() for r in references]
            
            correct = 0
            total = 0
            
            for pred, ref in zip(pred_words, ref_words):
                min_len = min(len(pred), len(ref))
                correct += sum(1 for i in range(min_len) if pred[i] == ref[i])
                total += len(ref)
                
            return correct / total if total > 0 else 0
        else:
            # Character-level accuracy
            correct = 0
            total = 0
            
            for pred, ref in zip(predictions, references):
                min_len = min(len(pred), len(ref))
                correct += sum(1 for i in range(min_len) if pred[i] == ref[i])
                total += len(ref)
                
            return correct / total if total > 0 else 0
    
    def calculate_bleu(self, references, predictions):
        """Calculate BLEU score."""
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = [ref.split()]
            pred_tokens = pred.split()
            score = sentence_bleu(ref_tokens, pred_tokens)
            scores.append(score)
        return np.mean(scores)
    
    def calculate_rouge(self, references, predictions):
        """Calculate ROUGE scores."""
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)
        return scores
    
    def evaluate(self, test_data, prefix_length=10, autocomplete_generator=None):
        """Evaluate the autocomplete system using test data."""
        if autocomplete_generator is None:
            from .autocomplete import AutocompleteGenerator
            autocomplete_generator = AutocompleteGenerator(self.model, self.tokenizer)
        
        predictions = []
        references = []
        
        for text in test_data:
            if len(text) < prefix_length + 10:  # Skip too short texts
                continue
                
            # Split text into prefix and reference
            prefix = text[:prefix_length]
            reference = text[prefix_length:]
            
            # Generate prediction
            prediction = autocomplete_generator.generate_completions(
                prefix, max_length=len(text), num_return_sequences=1)[0]
            
            predictions.append(prediction)
            references.append(reference)
        
        # Calculate metrics
        results = {
            "perplexity": self.calculate_perplexity(test_data),
            "word_accuracy": self.calculate_accuracy(predictions, references, word_level=True),
            "char_accuracy": self.calculate_accuracy(predictions, references, word_level=False),
            "bleu_score": self.calculate_bleu(references, predictions),
            "rouge_scores": self.calculate_rouge(references, predictions),
            "predictions": predictions,
            "references": references
        }
        
        return results