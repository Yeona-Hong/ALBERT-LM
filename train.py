import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import glob

# Define a custom model with a language modeling head on top of Albert
class AlbertWithLMHead(torch.nn.Module):
    def __init__(self, albert_model_name="albert-base-v2", vocab_size=30000):
        super(AlbertWithLMHead, self).__init__()
        self.albert = AlbertModel.from_pretrained(albert_model_name)
        self.lm_head = torch.nn.Linear(self.albert.config.hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.lm_head(sequence_output)
        
        return prediction_scores
    
    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        # Save the Albert config
        self.albert.config.save_pretrained(path)


# Create a custom dataset that reads from input files
class FileConversationDataset(Dataset):
    def __init__(self, tokenizer, input_files, output_files, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        
        # Read data from input and output files
        for input_file, output_file in zip(input_files, output_files):
            with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'r', encoding='utf-8') as out_f:
                input_lines = in_f.readlines()
                output_lines = out_f.readlines()
                
                # Ensure the files have the same number of lines
                assert len(input_lines) == len(output_lines), f"Input file {input_file} and output file {output_file} have different number of lines"
                
                for input_line, output_line in zip(input_lines, output_lines):
                    self.conversations.append({
                        "input": input_line.strip(),
                        "output": output_line.strip()
                    })
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_text = conversation["input"]
        output_text = conversation["output"]
        
        # Tokenize the input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize the output (target)
        target_encoding = self.tokenizer(
            output_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get the input IDs and attention mask
        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        
        # Get the target IDs
        target_ids = target_encoding["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids
        }


def get_input_output_files():
    """Get all input and output file pairs"""
    input_files = sorted(glob.glob("*.input"))
    output_files = []
    
    for input_file in input_files:
        output_file = input_file.replace(".input", ".output")
        if os.path.exists(output_file):
            output_files.append(output_file)
        else:
            print(f"Warning: Output file {output_file} not found for input file {input_file}")
            input_files.remove(input_file)
    
    return input_files, output_files


def train():
    # Initialize tokenizer
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    
    # Get input and output files
    input_files, output_files = get_input_output_files()
    
    if not input_files:
        print("No valid input/output file pairs found. Please ensure files like a.input, b.input, c.input and their corresponding .output files exist.")
        return None, None
    
    print(f"Found {len(input_files)} input/output file pairs: {input_files}")
    
    # Create dataset and dataloader
    dataset = FileConversationDataset(tokenizer, input_files, output_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = AlbertWithLMHead(albert_model_name="albert-base-v2", vocab_size=tokenizer.vocab_size)
    
    # Set up the optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Number of training epochs
    epochs = 5
    
    # Total number of training steps
    total_steps = len(dataloader) * epochs
    
    # Create a scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            # Reshape outputs to [batch_size*seq_length, vocab_size]
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Save the model
    os.makedirs("albert_lm_finetuned", exist_ok=True)
    model.save_pretrained("albert_lm_finetuned")
    tokenizer.save_pretrained("albert_lm_finetuned")
    
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_length=50):
    """
    Generate a response using the fine-tuned model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize input
    input_encoding = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Get input IDs and attention mask
    input_ids = input_encoding["input_ids"]
    attention_mask = input_encoding["attention_mask"]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.to(device)
    
    # Generate output
    with torch.no_grad():
        # Initial model prediction
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Simple greedy decoding
        generated_ids = input_ids
        
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Append to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Update attention mask
            new_attention_mask = torch.ones((attention_mask.shape[0], 1), device=device)
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def process_test_files():
    """
    Process test input files and generate corresponding outputs
    """
    # Initialize tokenizer and model
    tokenizer = AlbertTokenizer.from_pretrained("albert_lm_finetuned")
    
    # Load the model
    config = AlbertConfig.from_pretrained("albert_lm_finetuned")
    model = AlbertWithLMHead(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(os.path.join("albert_lm_finetuned", "model.pt")))
    
    # Get test input files (any .input files)
    test_files = glob.glob("*.input")
    
    for test_file in test_files:
        output_file = test_file.replace(".input", ".predicted")
        
        with open(test_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            lines = f_in.readlines()
            
            for line in tqdm(lines, desc=f"Processing {test_file}"):
                input_text = line.strip()
                response = generate_response(model, tokenizer, input_text)
                f_out.write(response + '\n')
        
        print(f"Processed {test_file} -> {output_file}")


def evaluate_model(model, tokenizer, eval_input_files, eval_output_files):
    """
    Evaluate the model performance on the given evaluation files
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    for input_file, output_file in zip(eval_input_files, eval_output_files):
        with open(input_file, 'r', encoding='utf-8') as in_f, open(output_file, 'r', encoding='utf-8') as out_f:
            input_lines = in_f.readlines()
            output_lines = out_f.readlines()
            
            assert len(input_lines) == len(output_lines), f"Input file {input_file} and output file {output_file} have different number of lines"
            
            print(f"Evaluating on {input_file} and {output_file}...")
            
            for i, (input_line, output_line) in enumerate(zip(input_lines, output_lines)):
                input_text = input_line.strip()
                expected_output = output_line.strip()
                
                # Generate response using the model
                generated_text = generate_response(model, tokenizer, input_text)
                
                # Compare with expected output
                if generated_text.strip() == expected_output:
                    total_correct += 1
                
                total_samples += 1
                
                # Print some examples
                if i < 5:  # Print first 5 examples
                    print(f"Input: {input_text}")
                    print(f"Expected: {expected_output}")
                    print(f"Generated: {generated_text}")
                    print("-" * 50)
    
    # Calculate accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    return accuracy


def test_inference(model, tokenizer, test_inputs):
    """
    Run test inference on a list of test inputs
    """
    model.eval()
    results = []
    
    for input_text in test_inputs:
        # Generate response
        generated_text = generate_response(model, tokenizer, input_text)
        results.append((input_text, generated_text))
        
        # Print the result
        print(f"Input: {input_text}")
        print(f"Output: {generated_text}")
        print("-" * 50)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Albert model for sentence correction")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "test", "all"],
                        help="Operation mode: train, eval, test, or all")
    parser.add_argument("--model_dir", type=str, default="albert_lm_finetuned",
                        help="Directory for the saved model")
    parser.add_argument("--test_inputs", type=str, nargs="+", default=[],
                        help="Test input sentences for inference mode")
    parser.add_argument("--eval_files", type=str, nargs="+", default=[],
                        help="Evaluation files for eval mode (should be in format: input1.txt,output1.txt input2.txt,output2.txt)")
    
    args = parser.parse_args()
    
    # Initialize tokenizer and model
    if args.mode in ["eval", "test"] or (args.mode == "all" and os.path.exists(args.model_dir)):
        print(f"Loading model from {args.model_dir}...")
        tokenizer = AlbertTokenizer.from_pretrained(args.model_dir)
        config = AlbertConfig.from_pretrained(args.model_dir)
        model = AlbertWithLMHead(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.pt")))
    else:
        model, tokenizer = None, None
    
    # Train mode
    if args.mode in ["train", "all"]:
        print("Training model...")
        model, tokenizer = train()
        
        if model is None or tokenizer is None:
            print("Training failed. Exiting...")
            return
    
    # Evaluation mode
    if args.mode in ["eval", "all"] and model is not None and tokenizer is not None:
        eval_input_files = []
        eval_output_files = []
        
        # Parse evaluation files
        if args.eval_files:
            for file_pair in args.eval_files:
                input_file, output_file = file_pair.split(",")
                eval_input_files.append(input_file)
                eval_output_files.append(output_file)
        else:
            # Use a portion of the training data for evaluation
            all_input_files, all_output_files = get_input_output_files()
            # Use 20% of files for evaluation
            eval_size = max(1, int(len(all_input_files) * 0.2))
            eval_input_files = all_input_files[-eval_size:]
            eval_output_files = all_output_files[-eval_size:]
        
        print(f"Evaluating model on {len(eval_input_files)} file pairs...")
        accuracy = evaluate_model(model, tokenizer, eval_input_files, eval_output_files)
    
    # Test inference mode
    if args.mode in ["test", "all"] and model is not None and tokenizer is not None:
        test_inputs = args.test_inputs
        
        if not test_inputs:
            # Use some sample inputs if none provided
            test_inputs = [
                "Hi I'm yeona",
                "Nice to meet you",
                "How are you doing today"
            ]
        
        print(f"Running test inference on {len(test_inputs)} inputs...")
        results = test_inference(model, tokenizer, test_inputs)
    
    # Process all test files
    if args.mode in ["all"] and model is not None and tokenizer is not None:
        print("Processing all test files...")
        process_test_files()


if __name__ == "__main__":
    main()
