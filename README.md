# Multimodal Stock Movement Prediction via Temporal NLP and Technical Indicators

## 📊 Project Overview
This repository implements a deep sequence learning framework to predict next-day directional stock movements using the benchmark **Stocknet Dataset**. The pipeline combines technical market data with unstructured financial micro-blogs (Twitter) to perform binary classification (Up/Down).

## 🏗️ Folder Structure
```text
stocknet-prediction/
├── src/
│   ├── data_preprocessing.py   # Feature engineering & Sentiment extraction
│   ├── data_merger.py          # Calendar alignment & data fusion
│   ├── dataset.py              # PyTorch Custom Sequence Dataset
│   ├── models.py               # BiLSTM & GRU with Attention mechanisms
│   └── train.py                # PyTorch training loops & MCC evaluation
└── README.md                   # Visual presentation layer
```

## 🧠 Model Architecture & Methodology
- **Data Alignment:** Uses asymmetric temporal matching to shift weekend tweet features to active trading days without data leakage.
- **Deep Sequence Learning:** Implements Bi-directional LSTMs and GRUs.
- **Attention Wrapper:** Features a temporal attention mechanism to automatically weight highly predictive trading frames.
- **Metrics:** Evaluated using Accuracy and the Matthews Correlation Coefficient (MCC).

## 📚 Dataset Source & Citation
Built using data from Xu and Cohen (2018): *"Stock Movement Prediction from Tweets and Historical Prices"*.
