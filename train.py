import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

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


# Create a custom dataset
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # This is a simple example. In a real scenario, you'd load from a file
        self.conversations = [
            {"input": "hello I'm yeona", "output": "hello. I'm yeona."},
            # Add more examples here to improve model training
            {"input": "hi I'm john", "output": "hi. I'm john."},
            {"input": "hey I'm alex", "output": "hey. I'm alex."},
            {"input": "greetings I'm sam", "output": "greetings. I'm sam."}
        ]
        
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


def train():
    # Initialize tokenizer
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    
    # Create dataset and dataloader
    dataset = ConversationDataset(tokenizer)
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


def main():
    # Train the model
    print("Training model...")
    model, tokenizer = train()
    
    # Test the model
    test_input = "hello I'm yeona"
    print(f"Input: {test_input}")
    
    response = generate_response(model, tokenizer, test_input)
    print(f"Generated response: {response}")


if __name__ == "__main__":
    main()
