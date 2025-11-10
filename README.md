# Multimodal Sentiment Analysis

This project is a deep learning implementation for sentiment analysis that fuses information from three distinct modalities—**text, audio, and video**—to predict sentiment intensity.

## 1. Problem Statement

Traditional sentiment analysis systems, which rely solely on text, are fundamentally limited. They fail to capture the full spectrum of human emotion, as they are blind to non-verbal cues (like tone of voice or facial expressions) that provide essential context. These systems are easily misled by complex phenomena such as **sarcasm, irony, and ambiguity**, leading to lower accuracy.

This project aims to address this gap by building a system that processes and fuses information from multiple modalities to achieve a more accurate and nuanced understanding of sentiment.

## 2. Proposed Approach

### 2.1 Dataset

The model is trained and evaluated on the **MOSI (Multimodal Opinion Sentiment and Sentiment Intensity)** dataset. This benchmark dataset provides video clips with word-level alignment for:

* **Text:** 300-dim GloVe word embeddings.

* **Audio:** 5-dim acoustic features (COVAREP).

* **Visual:** 20-dim visual features (FACET).

The task is a regression task to predict the sentiment intensity score (e.g., -3 to +3).

### 2.2 Model Architecture: `SimpleFusionModel`

The core of this project is a `SimpleFusionModel` built in PyTorch. The architecture can be broken down into three stages:

1. **Unimodal Encoders:** Each modality (text, audio, visual) is independently processed by its own **Bidirectional LSTM (Bi-LSTM)**. This captures the temporal patterns and context within each sequence.

2. **Late Fusion:** The final hidden state (a rich summary) is extracted from each of the three LSTMs. These three summary vectors are then **concatenated** into a single, large feature vector.

3. **Regression Head:** This fused vector is passed through a simple Multi-Layer Perceptron (MLP) that outputs a single continuous value, which is the predicted sentiment score.

## 3. Technology Stack

* **PyTorch:** The primary deep learning framework.

* **NumPy:** For numerical operations.

* **scikit-learn:** For calculating evaluation metrics (MAE, F1, Accuracy).

* **Pickle:** For loading the pre-processed MOSI dataset.

## 4. How to Run the Project

### 4.1 Prerequisites

* Python (3.8 or newer)

* pip

### 4.2 Setup and Installation

1. **Open a Command Prompt/Terminal.**

2. **Navigate to Your Project Directory:**

3. **Install Required Packages:**
(Ensure you have a `requirements.txt` file with libraries like `torch`, `numpy`, `scikit-learn`)

### 4.3 Training the Model

To train the model from scratch, run the following scripts:
1. `load_data.py`
2. `model.py`
3. `train.py`

The trained model weights will be saved automatically (here `best_model.pth`).

### 4.4 Evaluating the Model

Once the model is trained, you can evaluate it on the test set by running `eval.py`.

This will load the saved model weights and print the final test metrics.

To test a different *individual sample*, you must **edit the `eval.py` file** to change the `sample_index` variable.

## 6. Results

After training for 20 epochs, the model achieved the following performance on the test set:

| **Metric** | **Score** |
| **Test Loss (MAE)** | `1.098` |
| **Test MAE** | `1.107` |
| **Test Accuracy (Binary)** | `72.01%` |

| **Test F1-Score (Weighted)** | `0.722` |
