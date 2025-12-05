# ECG-Hypoglycemia-Detection

This repository contains a **deep learning framework** for detecting hypoglycemia from 10-second ECG segments. The model combines **Convolutional Neural Networks (CNNs)** and **Bidirectional LSTM (BiLSTM)** layers to capture both morphological and temporal patterns in ECG signals for binary classification (normal vs. hypoglycemic).  

This project was developed for **COE 747: Deep Learning**.

---

## Motivation

Hypoglycemia is a condition where blood glucose levels fall below normal, posing severe health risks including cognitive impairment, seizures, or loss of consciousness. Traditional glucose monitoring systems are invasive, costly, and not always accessible.  

ECG signals provide a **non-invasive and economical alternative** since hypoglycemia affects the autonomic nervous system, which in turn alters cardiac electrical activity. This project aims to detect hypoglycemia using deep learning applied to ECG signals.

---

## Dataset

The dataset contains **45,630 ten-second ECG segments** sampled at 250 Hz (2500 samples per segment) from **9 patients**, with severe class imbalance:  

- **Label 0:** Normal (Normoglycemic)  
- **Label 1:** Hypoglycemic  

Challenges:  

- Highly imbalanced class distribution  
- Different ECG patterns across patients  
- Limited number of hypoglycemic episodes  

Data split strategy: 80/20 train-test split with stratification to preserve class ratios.  

---

## Preprocessing

1. **Data Cleaning:** Removed metadata except `label`, discarded NaN and flat signals.  
2. **Normalization:** Standardized based on training set, clipped to [-5, 5].  
3. **Reshaping:** ECG reshaped to `(2500,1)` for CNN input.  
4. **Class Imbalance Handling:**  
   - Conservative oversampling  
   - Enhanced mixup for positive samples  
   - Class weights (positive: 3.7, negative: 1.0)  

---

## Model Architecture

The model is a **hybrid CNN-BiLSTM** network with the following layers:

### Convolutional Layers
- Conv1D(32 filters, kernel 21, stride 2, ReLU, BatchNorm, MaxPooling, Dropout 0.3)  
- Conv1D(64 filters, kernel 13, stride 2, ReLU, BatchNorm, MaxPooling, Dropout 0.4)  
- Conv1D(128 filters, kernel 7, ReLU, BatchNorm, MaxPooling, Dropout 0.4)  

### Bidirectional LSTM Layers
- BiLSTM(64 units, return_sequences=True, Dropout 0.2, Recurrent Dropout 0.2)  
- BiLSTM(32 units, Dropout 0.2, Recurrent Dropout 0.2)  

### Fully Connected Layers
- Dense(128 units, ReLU, BatchNorm, Dropout 0.5, L2 regularization)  
- Dense(64 units, ReLU, Dropout 0.4, L2 regularization)  
- Dense(1 unit, Sigmoid output)  

### Optimization & Regularization
- **Loss Function:** Binary cross-entropy  
- **Optimizer:** Adam, learning rate 5e-5  
- **Regularization:** Dropout, L2, BatchNorm, Gradient Clipping  
- **Training Callbacks:** EarlyStopping (F1-score), ReduceLROnPlateau, ModelCheckpoint, ResumeCheckpoint  

---

## Training

- **Batch size:** 32  
- **Epochs:** 140  
- **Metrics:** F1-score, Precision, Recall, AUC-ROC, AUC-PR, Specificity, Balanced Accuracy  

---

## Results

| Metric       | Validation | Training |
|-------------|------------|---------|
| F1-Score     | 0.6938     | 0.6848 |
| AUROC        | 0.9729     | -       |
| AUPR         | 0.7530     | -       |
| Precision    | 0.7244     | -       |
| Recall       | 0.6890     | -       |
| Specificity  | 0.9851     | -       |
| Balanced Acc | 0.8370     | -       |

The model shows stable learning and proper generalization, successfully detecting hypoglycemic episodes despite dataset challenges.

---

## Usage

1. Clone this repository:  
```bash
git clone https://github.com/<username>/ECG-Hypoglycemia-Detection.git
cd ECG-Hypoglycemia-Detection
