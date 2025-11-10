import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from load_data import (
    train_data, valid_data, test_data,
    TEXT_DIM, AUDIO_DIM, VIDEO_DIM,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    HIDDEN_DIM, OUTPUT_DIM
)

from model import MOSIDataset, collate_fn, SimpleFusionModel

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        text, audio, video = batch['text'].to(device), batch['audio'].to(device), batch['video'].to(device)
        text_len, audio_len, video_len = batch['text_lengths'].to(device), batch['audio_lengths'].to(device), batch['video_lengths'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        predictions = model(text, audio, video, text_len, audio_len, video_len)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            text, audio, video = batch['text'].to(device), batch['audio'].to(device), batch['video'].to(device)
            text_len, audio_len, video_len = batch['text_lengths'].to(device), batch['audio_lengths'].to(device), batch['video_lengths'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(text, audio, video, text_len, audio_len, video_len)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    mae = mean_absolute_error(all_labels, all_preds)
    preds_binary = [1 if p > 0 else 0 for p in all_preds]
    labels_binary = [1 if l > 0 else 0 for l in all_labels]
    acc_2 = accuracy_score(labels_binary, preds_binary)
    f1_2 = f1_score(labels_binary, preds_binary, average='weighted')
    
    return epoch_loss / len(dataloader), mae, acc_2, f1_2

print("\n Initializing Model and DataLoaders")

# Create DataLoaders
train_dataset = MOSIDataset(train_data)
valid_dataset = MOSIDataset(valid_data)
test_dataset = MOSIDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize Model, Loss, and Optimizer
model = SimpleFusionModel(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM, OUTPUT_DIM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.L1Loss() # Mean Absolute Error for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training on {device}...")

# Run Training
for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_mae, valid_acc, valid_f1 = evaluate_epoch(model, valid_loader, criterion, device)
    
    print(f'Epoch: {epoch:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid MAE: {valid_mae:.3f}')
    print(f'\tValid Acc-2: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.3f}')

test_loss, test_mae, test_acc, test_f1 = evaluate_epoch(model, test_loader, criterion, device)
print("\n--- Testing Complete ---")
print(f'Test Loss: {test_loss:.3f} | Test MAE: {test_mae:.3f}')
print(f'Test Acc-2: {test_acc*100:.2f}% | Test F1: {test_f1:.3f}')

print("\n Training Complete ")

torch.save(model.state_dict(), 'best_model.pth')
print("Model saved to best_model.pth")