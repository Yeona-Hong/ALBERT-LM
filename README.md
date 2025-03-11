# ALBERT-LM

## Usage Guide

### 1. Prepare Files

First, prepare the necessary files:
- Input files: `a.input`, `b.input`, `c.input`, etc.
- Output files: `a.output`, `b.output`, `c.output`, etc.

### 2. Training Commands

```
python script.py --mode train
```

By default, all `.input` and `.output` file pairs will be detected and used for training.

### 3. Evaluation Commands

```
python script.py --mode eval
```

**Evaluate with specific files:**
```
python script.py --mode eval --eval_files a.input,a.output b.input,b.output
```

### 4. Test Inference Commands

**Test with individual sentences:**
```
python script.py --mode test --test_inputs "Hi I'm yeona" "Nice to meet you"
```

**Process all input files and save results:**
```
python script.py --mode test
```
This command will process all `.input` files and save the results to `.predicted` files.

### 5. Run All Steps At Once

```
python script.py --mode all
```
This command will run training, evaluation, and testing in sequence.

### 6. Specify Trained Model Directory

```
python script.py --mode test --model_dir custom_model_dir
```
The default value is "albert_lm_finetuned".

### 7. Detailed Metrics Evaluation

```
python script.py --mode eval --eval_files a.input,a.output
```
This command calculates various evaluation metrics including exact match accuracy, character-level accuracy, and BLEU score.

### Practical Usage Example

For example, you can use the script in the following workflow:

1. Prepare input and output files
2. Run `python script.py --mode train` to train the model
3. Run `python script.py --mode test --test_inputs "Hi I'm yeona"` to test specific sentences
4. Run `python script.py --mode eval` to evaluate model performance

Detailed progress and results will be displayed for each command execution.
