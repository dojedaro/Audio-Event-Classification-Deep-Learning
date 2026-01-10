🔊 Environmental Audio Classification with Deep Learning

🎧 Multi-Class Sound Event Recognition using Classical ML & CNNs

📌 Overview

This project presents an end-to-end audio classification pipeline for environmental and mechanical sound events, progressing from raw waveform analysis to deep learning with mel-spectrograms.

The system classifies audio clips into four real-world sound categories using:

✅ A classical machine learning baseline (Decision Tree on raw audio)

✅ A 2D Convolutional Neural Network (CNN) trained on mel-spectrogram representations

The project emphasizes data inspection, preprocessing rigor, and model comparison, making it suitable for both academic study and applied audio AI portfolios.

🎯 Sound Categories

The dataset is organized into four labeled classes:

🔹 Explosion
🔹 Emergency Signals
🔹 Engine Running
🔹 Take Off

Each class contains real-world audio recordings with varying durations and acoustic characteristics.

🗂️ Dataset Handling

📁 Audio data is stored on Google Drive and accessed via Colab mounting.
📁 Each category corresponds to a dedicated folder.

🎧 Audio Properties

Multiple formats: .wav, .mp3, .ogg, .flac

Variable-length recordings

Mono / stereo inputs handled automatically

Sampling rates preserved or normalized when needed

🔍 Exploratory Audio Analysis

Before modeling, the project performs:

📊 Waveform visualization
📏 Sample-length and duration analysis per class
📈 Minimum, maximum, and average audio lengths

This step ensures:

Data consistency

Safe truncation and padding strategies

Fair model input preparation

🧪 Baseline Model — Classical ML
🌳 Decision Tree Classifier

A Decision Tree is trained directly on flattened raw audio samples:

Audio signals truncated to a fixed length

No handcrafted features

Direct waveform-based classification

📊 Baseline Performance

Accuracy: ~86%

Macro F1 Score: ~0.83

Weighted F1 Score: ~0.86

🔎 This baseline establishes a reference point and highlights the limitations of raw-signal classical models.

🧠 Deep Learning Approach — 2D CNN
🎼 Feature Extraction

Audio signals are transformed into mel-spectrograms:

Sample rate: 22,050 Hz

Duration: 5 seconds

Mel bands: 128

Log-scaled (dB) representation

Each spectrogram is treated as an image, enabling spatial pattern learning.

🧱 CNN Architecture

🧩 Model characteristics:

3 convolutional blocks

Batch normalization + max pooling

Dense layers with dropout

Softmax output for multi-class prediction

📦 Total parameters: ~5.8M
⚡ Optimizer: Adam
🎯 Loss: Categorical Cross-Entropy

📈 CNN Training & Evaluation
🏋️ Training Setup

Train / test split: 80 / 20

Early stopping enabled

Batch size: 32

Epochs: up to 30

🧪 Final CNN Results (Test Set)

✅ Accuracy: 100%
✅ Macro F1: 1.00
✅ Weighted F1: 1.00

📊 Per-Class F1 Scores

Explosion: 1.00

Emergency Signals: 1.00

Engine Running: 1.00

Take Off: 1.00

🚀 The CNN significantly outperforms the classical baseline, demonstrating the power of time–frequency representations.

🔁 Pipeline Summary

1️⃣ Load and inspect raw audio
2️⃣ Analyze duration and waveform properties
3️⃣ Build a classical ML baseline
4️⃣ Convert audio to mel-spectrograms
5️⃣ Train a deep CNN
6️⃣ Evaluate using accuracy and F1 metrics
7️⃣ Compare baseline vs deep learning
