# Install required libraries
#!pip install datasets transformers torch huggingface_hub

import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig
from huggingface_hub import login, HfApi, HfFolder, Repository

# Step 1: Load and split the dataset
dataset = load_dataset("mtpti5iD/redhat-docs_dataset")

split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_valid = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": split_dataset["train"],
    "val": test_valid["train"],
    "test": test_valid["test"]
})

print("Dataset split completed!")

# Step 2: Define the model architecture

class TransformerModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(TransformerModel, self).__init__()
        self.bert = BertModel(BertConfig())
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        logits = self.fc(pooled_output)
        return logits

hidden_size = 768
num_classes = 2
model = TransformerModel(hidden_size, num_classes)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

print("Model architecture defined!")

# Step 3: Train the model


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]['content']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = CustomDataset(dataset_dict['train'], tokenizer, max_len=512)
val_dataset = CustomDataset(dataset_dict['val'], tokenizer, max_len=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, torch.zeros(len(outputs), dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, torch.zeros(len(outputs), dtype=torch.long))
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

print("Training completed!")


# Step 5: Save model weights to Google Drive

save_directory = "/myfolder/saved_model"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Model and tokenizer saved folder")

# Step 6: Upload to Hugging Face

login(token="YOUR_HUGGINGFACE_TOKEN")

repo_name = "redhat-docs-llm"
user_name = "mtpti5iD"

api = HfApi()
api.create_repo(repo_name, token=HfFolder.get_token(), private=False)

repo_path = f"{user_name}/{repo_name}"
repo = Repository(repo_path, clone_from=repo_path)

os.system(f"cp -r {save_directory}/* {repo_path}")

repo.push_to_hub()

print("Model uploaded to Hugging Face!")
