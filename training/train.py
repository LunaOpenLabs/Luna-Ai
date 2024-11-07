import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Determine device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Function to load BERT model
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Set num_labels as per your dataset
    model.to(device)  # Move the model to the correct device (GPU or CPU)
    return tokenizer, model

# Function to train BERT model
def train_bert(model, tokenizer, texts, labels, epochs=1, batch_size=4):  # Reduced epochs and batch size
    logger.info("Starting BERT training...")
    
    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text, add_special_tokens=True, padding='max_length', truncation=True, max_length=256, return_tensors='pt'  # Reduced sequence length
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    # Prepare DataLoader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, leave=True)
        try:
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            break
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    logger.info("BERT training completed.")

# Function to load T5 model
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.to(device)  # Move the model to the correct device (GPU or CPU)
    return tokenizer, model

# Function to generate text using T5
def generate_text_with_t5(tokenizer, model, input_text):
    # Encoding the input text
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=256, truncation=True)  # Reduced sequence length
    inputs = inputs.to(device)  # Move inputs to the correct device

    # Generate text using T5
    summary_ids = model.generate(inputs, max_length=50, num_beams=4, early_stopping=True)

    # Decode and return the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main function to run both BERT training and T5 text generation
def main():
    # Example dataset (replace with your actual dataset)
    texts = ["I love machine learning.", "Deep learning is great for NLP."]
    labels = [0, 1]  # Binary classification labels

    # Load BERT model for classification
    bert_tokenizer, bert_model = load_bert_model()
    
    # Train the BERT model
    try:
        train_bert(bert_model, bert_tokenizer, texts, labels)
    except Exception as e:
        logger.error(f"Error during BERT training: {e}")

    # Load T5 model for NLP task (summarization in this case)
    t5_tokenizer, t5_model = load_t5_model()

    # Test T5 for text generation (summarization)
    input_text = "This is an example sentence that I would like to summarize using T5."
    t5_output = generate_text_with_t5(t5_tokenizer, t5_model, input_text)
    logger.info(f"T5 Output: {t5_output}")

if __name__ == "__main__":
    main()
