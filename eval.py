import torch
from model import SimpleFusionModel, MOSIDataset, collate_fn
from load_data import (
    test_data, TEXT_DIM, AUDIO_DIM, VIDEO_DIM,
    HIDDEN_DIM, OUTPUT_DIM
)

# --- 1. LOAD TRAINED MODEL ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleFusionModel(TEXT_DIM, AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

print("\n✅ Model loaded successfully from 'best_model.pth' and ready for inference!")

# --- 2. PREPARE TEST SAMPLE ---
test_dataset = MOSIDataset(test_data)

# Change this index to test different samples
sample_index = 4
sample = test_dataset[sample_index]

print(f"\n--- Running Inference on Sample #{sample_index} ---")

# The collate_fn prepares a batch (even if we have only 1 sample)
batch = collate_fn([sample])

# Move data to device
text = batch['text'].to(device)
audio = batch['audio'].to(device)
video = batch['video'].to(device)
text_len = batch['text_lengths'].to(device)
audio_len = batch['audio_lengths'].to(device)
video_len = batch['video_lengths'].to(device)
label = batch['labels'][0].item()  # True label

# --- 3. RUN INFERENCE ---
with torch.no_grad():
    prediction = model(text, audio, video, text_len, audio_len, video_len)
    prediction_value = prediction[0].item()

# --- 4. PRINT RESULTS ---
print(f"   🎯 Actual Label: {label:.4f}")
print(f"   🤖 Model Prediction: {prediction_value:.4f}")

if (prediction_value > 0 and label > 0) or (prediction_value <= 0 and label <= 0):
    print("✅ Result: Correct (Positive/Negative match)")
else:
    print("❌ Result: Incorrect (Positive/Negative mismatch)")

print("\n--- Inference Complete ---")