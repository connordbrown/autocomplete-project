import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class ErrorAnalyzer:
    def __init__(self):
        """Initialize the ErrorAnalyzer with NLTK."""
        pass
    
    def analyze_formality(self, text):
        """Analyze text formality based on simple heuristics."""
        tokens = word_tokenize(text)
        
        # Count formal indicators
        formal_indicators = sum([
            1 for token in tokens if token.lower() in 
            ["shall", "would", "could", "may", "therefore", "thus", "hence"]
        ])
        
        # Count informal indicators
        informal_indicators = sum([
            1 for token in tokens if token.lower() in 
            ["gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "nah", "hey"]
        ])
        
        # Count contractions
        contractions = sum([1 for token in tokens if "'" in token])
        
        # Count interjections
        interjections = sum([1 for token, pos in pos_tag(tokens) if pos == "UH"])
        
        # Calculate formality score (0 = informal, 1 = formal)
        total_indicators = formal_indicators + informal_indicators + contractions
        if total_indicators == 0:
            return 0.5  # Neutral
        
        formality_score = formal_indicators / (formal_indicators + informal_indicators + contractions)
        return formality_score + 0.1 * interjections / len(tokens)
    
    def categorize_errors(self, predictions, references):
        """Categorize prediction errors into semantic, grammatical, and style categories."""
        error_categories = {
            "semantic_errors": [],
            "grammatical_errors": [],
            "style_inconsistencies": []
        }
        
        for pred, ref in zip(predictions, references):
            # Check for grammatical errors using NLTK
            pred_tokens = word_tokenize(pred)
            if len(pred_tokens) > 0:
                pos_tags = pos_tag(pred_tokens)
                first_word_pos = pos_tags[0][1]
                has_grammatical_error = first_word_pos not in ['VB', 'NN', 'PRP', 'JJ', 'DT', 'IN']
            else:
                has_grammatical_error = False
            
            # Basic style consistency check
            style_mismatch = False
            if len(ref) > 0 and len(pred) > 0:
                ref_formality = self.analyze_formality(ref)
                pred_formality = self.analyze_formality(pred)
                style_mismatch = abs(ref_formality - pred_formality) > 0.3
            
            # If no grammatical errors but prediction differs from reference, 
            # consider it a semantic error
            if pred != ref and not has_grammatical_error and not style_mismatch:
                error_categories["semantic_errors"].append((pred, ref))
            
            if has_grammatical_error:
                error_categories["grammatical_errors"].append((pred, ref))
                
            if style_mismatch:
                error_categories["style_inconsistencies"].append((pred, ref))
        
        return error_categories
    
    def analyze_errors(self, evaluation_results):
        """Analyze errors from evaluation results."""
        predictions = evaluation_results["predictions"]
        references = evaluation_results["references"]
        
        error_categories = self.categorize_errors(predictions, references)
        
        # Add error analysis to results
        evaluation_results["error_analysis"] = {
            "semantic_errors": len(error_categories["semantic_errors"]),
            "grammatical_errors": len(error_categories["grammatical_errors"]),
            "style_inconsistencies": len(error_categories["style_inconsistencies"]),
            "error_examples": {
                "semantic": error_categories["semantic_errors"][:5],
                "grammatical": error_categories["grammatical_errors"][:5],
                "style": error_categories["style_inconsistencies"][:5]
            }
        }
        
        return evaluation_results