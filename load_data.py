import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
import numpy as np
import pickle

DATA_PATH = 'mosi_data.pkl'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
HIDDEN_DIM = 64
OUTPUT_DIM = 1

print(f"Loading data from {DATA_PATH}...")
try:
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
except Exception as e:
    print(f"Error loading pickle file: {e}")
    print("Trying without 'latin1' encoding...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

print("Data loaded successfully.")

train_data = data['train']
valid_data = data['valid']
test_data = data['test']

print(f"Keys in 'train' split: {train_data.keys()}")

train_text = train_data['text']
train_audio = train_data['audio']
train_video = train_data['vision']
train_labels = train_data['labels']

TEXT_DIM = train_text[0].shape[1]
AUDIO_DIM = train_audio[0].shape[1]
VIDEO_DIM = train_video[0].shape[1]

print(f"\n--- Detected Feature Dimensions ---")
print(f"TEXT_DIM = {TEXT_DIM}")
print(f"AUDIO_DIM = {AUDIO_DIM}")
print(f"VIDEO_DIM = {VIDEO_DIM}")

__all__ = [
    'train_data', 'valid_data', 'test_data',
    'TEXT_DIM', 'AUDIO_DIM', 'VIDEO_DIM',
    'BATCH_SIZE', 'EPOCHS', 'LEARNING_RATE',
    'HIDDEN_DIM', 'OUTPUT_DIM'
]
