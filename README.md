# ECG-Hypoglycemia-Detection

This repository contains a **deep learning framework** for detecting hypoglycemia from 10-second ECG segments. The model combines **Convolutional Neural Networks (CNNs)** and **Bidirectional LSTM (BiLSTM)** layers to capture both morphological and temporal patterns of ECG signals for binary classification (normoglycemic vs. hypoglycemic).  

Developed for **COE 747: Deep Learning**.

---

## Motivation

Hypoglycemia occurs when blood glucose drops below normal levels, posing serious health risks including confusion, seizures, or loss of consciousness. Traditional monitoring methods, such as fingerstick tests or continuous glucose monitoring (CGM), are invasive and costly.  

ECG signals offer a **non-invasive alternative**, as hypoglycemia triggers autonomic nervous system responses that alter cardiac electrical activity. This project leverages deep learning to detect hypoglycemic episodes from ECG segments.

---

## Dataset

- **45,630 ECG segments**, each 10 seconds long (2,500 samples per segment) from **9 diabetic patients**  
- **Labels:**  
  - `0` → Normal (Normoglycemic)  
  - `1` → Hypoglycemic  
- Each segment is labeled using CGM measurements synchronized with ECG recordings  
- **Class imbalance:** only ~5.4% of segments are hypoglycemic, making the dataset highly imbalanced  
- **Challenges:** patient-specific variability, limited hypoglycemic events  

---

## Preprocessing

- Removed unnecessary metadata and invalid segments  
- Standardized ECG signals and clipped  
- Reshaped ECGs to `(2500,1)` for CNN input  
- Handled class imbalance using oversampling, mixup augmentation, and class weighting  

---

## Model Overview

The model is a **hybrid CNN-BiLSTM** network capturing both spatial and temporal features of ECG signals. Regularization techniques such as dropout, batch normalization, and L2 penalties are applied to improve generalization. The model uses **binary cross-entropy loss** and the **Adam optimizer**.  

---

## Training

- **Batch size:** 32  
- **Epochs:** 140  
- **Metrics:** F1-score, Precision, Recall, AUROC, AUPR, Specificity, Balanced Accuracy  

---

## Results

| Metric       | Testing  | Training|
|------------- |--------- |---------|
| F1-Score     | 0.6938   | 0.6848  |
| AUROC        | 0.9729   | -       |
| AUPR         | 0.7530   | -       |
| Precision    | 0.7244   | -       |
| Recall       | 0.6890   | -       |
| Specificity  | 0.9851   | -       |
| Balanced Acc | 0.8370   | -       |

The model demonstrates stable learning and generalization, successfully detecting hypoglycemic episodes despite dataset challenges.

---

Licensed under MIT
