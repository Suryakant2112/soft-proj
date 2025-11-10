import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class MOSIDataset(Dataset):
    def __init__(self, data_split):
        self.text_data = data_split['text']
        self.audio_data = data_split['audio']
        self.video_data = data_split['vision']
        self.labels = data_split['labels']
        self.n_samples = len(self.labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return {
            'text': torch.tensor(self.text_data[index], dtype=torch.float32),
            'audio': torch.tensor(self.audio_data[index], dtype=torch.float32),
            'video': torch.tensor(self.video_data[index], dtype=torch.float32),
            'label': torch.tensor(self.labels[index][0], dtype=torch.float32)
        }

def collate_fn(batch):
    text_data = [item['text'] for item in batch]
    audio_data = [item['audio'] for item in batch]
    video_data = [item['video'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    text_lengths = torch.tensor([len(seq) for seq in text_data])
    audio_lengths = torch.tensor([len(seq) for seq in audio_data])
    video_lengths = torch.tensor([len(seq) for seq in video_data])

    text_padded = pad_sequence(text_data, batch_first=True, padding_value=0)
    audio_padded = pad_sequence(audio_data, batch_first=True, padding_value=0)
    video_padded = pad_sequence(video_data, batch_first=True, padding_value=0)

    return {
        'text': text_padded,
        'audio': audio_padded,
        'video': video_padded,
        'text_lengths': text_lengths,
        'audio_lengths': audio_lengths,
        'video_lengths': video_lengths,
        'labels': labels
    }

class SimpleFusionModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, hidden_dim, output_dim, dropout=0.3):
        super(SimpleFusionModel, self).__init__()
        
        self.text_lstm = nn.LSTM(text_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_lstm = nn.LSTM(audio_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.video_lstm = nn.LSTM(video_dim, hidden_dim, batch_first=True, bidirectional=True)

        fused_dim = (hidden_dim * 2) * 3
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, output_dim)
        )

    def forward(self, text, audio, video, text_lengths, audio_lengths, video_lengths):
        text_lengths_cpu = text_lengths.cpu()
        audio_lengths_cpu = audio_lengths.cpu()
        video_lengths_cpu = video_lengths.cpu()

        text_packed = pack_padded_sequence(text, text_lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (text_hidden, _) = self.text_lstm(text_packed)
        
        audio_packed = pack_padded_sequence(audio, audio_lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (audio_hidden, _) = self.audio_lstm(audio_packed)
        
        video_packed = pack_padded_sequence(video, video_lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (video_hidden, _) = self.video_lstm(video_packed)

        text_last_hidden = torch.cat((text_hidden[0], text_hidden[1]), dim=1)
        audio_last_hidden = torch.cat((audio_hidden[0], audio_hidden[1]), dim=1)
        video_last_hidden = torch.cat((video_hidden[0], video_hidden[1]), dim=1)

        fused_output = torch.cat((text_last_hidden, audio_last_hidden, video_last_hidden), dim=1)
        
        prediction = self.fusion_layer(fused_output)
        
        return prediction.squeeze(1)


__all__ = ['MOSIDataset', 'collate_fn', 'SimpleFusionModel']