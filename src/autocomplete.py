import torch

class AutocompleteGenerator:
    def __init__(self, model, tokenizer):
        """Initialize with a trained model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_completions(self, input_text, max_length=50, num_return_sequences=3):
        """Generate autocomplete suggestions for the given input text."""
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Calculate input length and make sure max_length is greater
        input_length = input_ids.shape[1]
        
        # Use max_new_tokens instead of max_length to specify additional tokens beyond input
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_length,  # Generate this many new tokens beyond input
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
        
        # Decode the completions and extract only the continuation
        results = []
        for ids in output:
            completion = self.tokenizer.decode(ids, skip_special_tokens=True)
            if completion.startswith(input_text):
                continuation = completion[len(input_text):]
                results.append(continuation)
            else:
                results.append(completion)
                
        return results