import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        """
        Custom Dataset class for reading and tokenizing the CSV file data.
        Args:
            csv_file (str): Path to the CSV file with 'text' and 'label' columns.
            tokenizer (BertTokenizer): Hugging Face tokenizer for text tokenization.
            max_length (int): Maximum token length for sequences.
        """
        self.data = pd.read_csv(csv_file)

        # Ensure labels are integers, even if they are strings
        if self.data['label'].dtype == 'O':  # Object type (usually strings)
            logging.info("Label column contains strings. Encoding to integers...")
            encoder = LabelEncoder()
            self.data['label'] = encoder.fit_transform(self.data['label'])

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        # Tokenize the text and convert it to the format required by BERT
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Return the tokenized input and label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure label is a long tensor
        }

def train():
    # Load the tokenizer from Hugging Face
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the dataset
    logging.info("Loading the dataset...")
    try:
        dataset = TextDataset('data/data.csv', tokenizer)
    except FileNotFoundError:
        logging.error("The file 'data/data.csv' was not found.")
        return

    # Create DataLoader for training
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Batch size reduced to 8

    # Load BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Training loop
    epochs = 3
    logging.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Create progress bar using tqdm
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch in loop:
            # Move batch to GPU if available
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_postfix(loss=loss.item())

        # Logging the average loss after each epoch
        avg_loss = running_loss / len(train_dataloader)
        logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    logging.info("Training completed successfully.")

    # Save the trained model
    logging.info("Saving the trained model...")
    model.save_pretrained('luna_model')
    tokenizer.save_pretrained('luna_model')

    logging.info("Model saved successfully.")

if __name__ == "__main__":
    train()
