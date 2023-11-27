import random
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Instructions and Synonyms
instructions = ["take off", "land", "go up", "go down", "go forward", "go backward", "move right", "move left", "turn right", "turn left", "stop"]
synonyms = {
    "take off": ["take off", "depart", "blast off", "launch", "become airborne"],
    "land": ["land", "touch down", "arrive", "settle", "touch ground", "alight", "rest", "dismount", "earth"],
    "go up": ["go up", "ascend", "rise", "move upward", "climb", "scale", "lift", "uplift", "soar", "skyward"],
    "go down": ["go down", "descend", "fall", "move downward", "plunge", "dive", "drop", "sink", "dip"],
    "go forward": ["go forward", "move forward", "advance", "proceed", "progress", "travel forward", "push forward", "move ahead"],
    "go backward": ["go backward", "move backward", "retreat", "reverse", "back up", "draw back", "withdraw", "retrocede"],
    "move right": ["move right", "go to the right", "shift right", "sidestep right", "strafe right", "slide right", "scoot right", "glide right"],
    "move left": ["move left", "go to the left", "shift left", "sidestep left", "strafe left", "slide left", "scoot left", "glide left"],
    "turn right": ["turn right", "rotate right", "rotate clockwise", "veer right", "swivel right", "twist right", "pivot right", "steer right", "angle right"],
    "turn left": ["turn left", "rotate left", "rotate counterclockwise", "veer left", "swivel left", "twist left", "pivot left", "steer left", "angle left"],
    "stop": ["stop", "halt", "cease", "stand still", "come to a halt", "pause", "end", "freeze", "discontinue", "suspend"]
}

# Create the augmented dataset and save to CSV
dataset = []
for instruction_idx, instruction in enumerate(instructions):
    for _ in range(50):  # Original instruction appears 50 times
        dataset.append((instruction_idx, instruction))
    for _ in range(50):  # Synonyms appear 50 times
        if instruction in synonyms:
            dataset.append((instruction_idx, random.choice(synonyms[instruction])))

# Get the full path for the dataset
current_path = os.getcwd()
dataset_name = 'test_instructions_dataset.csv'
dataset_path = os.path.join(current_path, dataset_name)

# Create a DataFrame and save to CSV
df = pd.DataFrame(dataset, columns=['label', 'instruction'])
df.to_csv(dataset_path, index=False)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', truncation=True)

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label).long()
        }

# Load data from CSV
df = pd.read_csv(dataset_path)
texts = list(df['instruction'])
labels = list(df['label'])

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Create custom datasets for training and validation
max_length = 64
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define and train the BERT model
num_classes = len(instructions)
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=num_classes)

# Switch to CPU
device = torch.device("cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Initialize lists to track metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float('inf')
patience, trials = 3, 0  # Early stopping parameters

num_epochs = 5
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        batch_input_ids, batch_input_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_input_mask)
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    val_loss, correct, total = 0, 0, 0
    val_losses = [] 
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch_input_ids, batch_input_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(batch_input_ids, attention_mask=batch_input_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            val_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = sum(val_losses) / len(val_losses)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping triggered")
            break

# Plotting training and validation loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.show()

# Get the full path for the model
model_name = 'bert_model'
model_path = os.path.join(current_path, model_name)

# Save the model and tokenizer
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)