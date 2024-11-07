from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import pandas as pd
import torch.optim as optim
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Custom Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)  # Labels should be integers, not strings
        }

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels as needed

# Load dataset
dataset = TextDataset('data/data.csv', tokenizer)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_dataloader) * 4  # 4 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(4):  # Adjust number of epochs as needed
    model.train()
    loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}')
    
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

# Save the model after training
model.save_pretrained('model')
tokenizer.save_pretrained('model')
