from datasets import load_dataset
from transformers import GPT2Tokenizer

def load_webtext():
    """Load WebText dataset."""
    # "stas/openwebtext-10k" is a smaller subset with 10k samples
    return load_dataset("stas/openwebtext-10k")

def load_gutenberg():
    """Load Gutenberg dataset."""
    return load_dataset("bookcorpusopen")

def preprocess_data(dataset, tokenizer, max_length=128):
    """Preprocess and tokenize dataset."""
    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, padding="max_length", 
                         max_length=max_length)
        # Set labels equal to input_ids for causal language modeling
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def limit_dataset_size(dataset, max_samples=10000, seed=42):
    """Limit dataset to a maximum number of samples."""
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    
    # Create a smaller shuffled subset
    shuffled_dataset = dataset.shuffle(seed=seed)
    
    # Take only the first max_samples
    return shuffled_dataset.select(range(max_samples))

def prepare_dataset(dataset_name="webtext", use_tokenizer=None, max_samples=10000):
    """Prepare dataset for training/evaluation with size limit."""
    if use_tokenizer is None:
        use_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        use_tokenizer.pad_token = use_tokenizer.eos_token
    
    if dataset_name.lower() == "webtext":
        dataset = load_webtext()
    elif dataset_name.lower() == "gutenberg":
        dataset = load_gutenberg()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit dataset size before tokenization to save memory
    if max_samples is not None:
        dataset = {split: limit_dataset_size(data, max_samples) 
                   for split, data in dataset.items()}
    
    # Tokenize the limited dataset
    tokenized_dataset = {split: preprocess_data(data, use_tokenizer) 
                         for split, data in dataset.items()}
    
    return tokenized_dataset