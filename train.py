import torch
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import torch
import torch.nn as nn
from tqdm import tqdm
import transformers



### GPTNeo
class TokenAcceptanceModel(torch.nn.Module):
    def __init__(self, input_size, tokenizer, gpt_neo_model_name='EleutherAI/gpt-neo-125m'): # can also change to smaller one, then the hidden layer dim is 768, best acc we can get is 70%
        super(TokenAcceptanceModel, self).__init__()
        self.tokenizer = tokenizer
        self.gpt_neo = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name)
        for param in self.gpt_neo.parameters():
            param.requires_grad = False  # Freeze GPT-Neo
        self.classifier = torch.nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.gpt_neo(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # Use the last token's hidden state
        return self.classifier(hidden_states)


class AcceptanceDataset(Dataset):
    def __init__(self, prompts, token_indices, labels, tokenizer):
        self.prompts = prompts
        self.token_indices = token_indices
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx, addin = False):
        prompt = self.prompts[idx]
        token_index = self.token_indices[idx]
        label = self.labels[idx]
        token_text = self.tokenizer.decode([token_index])
        if addin == True:
            full_text = f"{prompt} {token_text}"
        else:
            full_text = f"{prompt}"
        return full_text, torch.tensor(label, dtype=torch.float)




def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    print("Start Training")
    for texts, labels in data_loader:
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions.squeeze(), labels.cuda())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(data_loader)}")




def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0
    print("Start testing")
    with torch.no_grad():
        for texts, labels in data_loader:
            predictions = model(texts)
            loss = criterion(predictions.squeeze(), labels.cuda())
            predicted_labels = (predictions.squeeze() >= 0.5).long()
            total_correct += (predicted_labels == labels.cuda()).sum().item()
            total += len(labels)
            total_loss += loss.item()
    print(f"Accuracy: {total_correct / total:.2f}")
    print(f"Test Loss = {total_loss / len(data_loader)}")
    acc = total_correct / total
    return acc, model




if __name__ == "__main__":
    file_name = 'collection_length03_ratio1_lm1b_neo.csv'
    target_name = 'EleutherAI/gpt-neo-1.3B'
    draft_name = 'EleutherAI/gpt-neo-125m'
    save_model_name =  'DRT_length03_ratio15_lm1b_neo.pth'
    epochs = 20
    best_acc = 0
    lrate = 0.0005
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataframe = pd.read_csv(file_name)
    train, test = train_test_split(dataframe, test_size=0.3)
    tokenizer = transformers.AutoTokenizer.from_pretrained(draft_name)#transformers.AutoTokenizer.from_pretrained('openai-community/gpt2')#transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = AcceptanceDataset(train['prompt'].tolist(), train['token'].tolist(),  train['threshold_label'].tolist(), tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataset = AcceptanceDataset(test['prompt'].tolist(), test['token'].tolist(),  test['threshold_label'].tolist(), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    model = TokenAcceptanceModel(input_size=768, tokenizer=tokenizer).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lrate)
    criterion = torch.nn.BCELoss()
    for epoch in tqdm(range(epochs)):
        train_model(model, train_loader, optimizer, criterion)
        acc, model = evaluate_model(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.classifier.state_dict(),save_model_name)