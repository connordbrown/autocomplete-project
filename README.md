# autocomplete-project

This guide explains how to use my GPT-2 autocomplete training and evaluation system.

## System Overview

This system consists of two main components:

1. Trainer (`simple_trainer.py`): Trains a GPT-2 model on text data for autocomplete functionality.
2. Evaluator (`simple_evaluator.py`): Tests the trained model's performance using various metrics.

There are six auxiliary files and folders:

1. `.idx`: Sets up Nix environment.
2. `.vscode`: Other Nix environment configuration.
3. `.gitignore`: Keeps trained models and cache files local.
4. `flake.nix`: Sets up dependencies.
4. `flake.lock`: Works with `flake.nix`.
5. `simple_evaluation_results.json`: Displays results of running `simple_evaluator.py`.

And three unused/untested/prototype files and folders:

1. `src`: Contains python files for a more sophisticated but computationally expensive autocomplete.
2. `scripts`: Contains python files for running the more complex autocomplete system.
3.  `app.py`: Script for running a user-testable CLI that uses the complex autocomplete system.

## Dependencies

### Using Nix Environment

If you're using Nix, you can set up the environment with:

```bash
nix develop
```

### Using pip

Otherwise, install the required dependencies with pip:

```bash
pip install torch transformers datasets nltk rouge_score numpy
```

## Training the Model

The training script (`simple_trainer.py`) fine-tunes a GPT-2 model on your choice of dataset.

### Basic Usage

```bash
python simple_trainer.py --dataset webtext --output_dir ./models/webtext-model
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use (webtext or gutenberg) | webtext |
| `--output_dir` | Directory to save the model | ./models/webtext-model |
| `--epochs` | Number of training epochs | 1 |
| `--batch_size` | Training batch size | 4 |
| `--max_samples` | Maximum number of samples to use | 2000 |
| `--use_trainer` | Use HuggingFace Trainer API instead of custom training loop | False |

### Example Commands

**Train on webtext dataset with default settings:**
```bash
python simple_trainer.py --dataset webtext
```

**Train on gutenberg dataset with 3 epochs:**
```bash
python simple_trainer.py --dataset gutenberg --epochs 3 --output_dir ./models/gutenberg-model
```

**Train with HuggingFace Trainer API using a larger dataset:**
```bash
python simple_trainer.py --dataset webtext --use_trainer --max_samples 5000
```

## Evaluating the Model

After training, use the evaluation script (`simple_evaluator.py`) to assess model performance.

### Basic Usage

```bash
python simple_evaluator.py --model_path ./models/webtext-model
```

### Evaluation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_path` | Path to the trained model | ./models/webtext-model |
| `--num_samples` | Number of samples for evaluation | 100 |
| `--output_file` | Path to save evaluation results | ./simple_evaluation_results.json |

### Example Commands

**Evaluate with default settings:**
```bash
python simple_evaluator.py --model_path ./models/webtext-model
```

**Evaluate with more samples:**
```bash
python simple_evaluator.py --model_path ./models/webtext-model --num_samples 200
```

**Evaluate and save results to a custom location:**
```bash
python simple_evaluator.py --model_path ./models/gutenberg-model --output_file ./gutenberg_eval_results.json
```

## Understanding Evaluation Metrics

The evaluation script measures:

- **Perplexity**: Lower values indicate better language modeling
- **Word-level Accuracy**: How often predicted words match the reference
- **Character-level Accuracy**: How often predicted characters match the reference
- **BLEU Score**: Measures n-gram overlap between prediction and reference
- **ROUGE-1 F1**: Measures unigram overlap between prediction and reference
- **Error Analysis**: Categorizes errors as semantic, grammatical, or style inconsistencies

## Workflow Example

1. **Train the model**:
   ```bash
   python simple_trainer.py --dataset webtext --epochs 2 --max_samples 3000
   ```

2. **Evaluate the model**:
   ```bash
   python simple_evaluator.py --model_path ./models/webtext-model
   ```

3. **Review evaluation results** in the generated JSON file.

## Notes and Tips

- Training is memory-intensive. Lower `--batch_size` if you encounter memory issues.
- For better results, train for more epochs (3-5) and use more samples.
- The evaluator tests on "webtext" data by default for consistency.
- The model generates completions based on context, not specific word prediction.
- Both scripts have built-in memory optimizations for working with limited resources.

## Troubleshooting

- If you encounter CUDA memory errors, try reducing `--batch_size` or `--max_samples`.
- For CPU-only training, the scripts will automatically detect and use CPU mode.
- If loading large models fails, ensure you have enough system memory available.