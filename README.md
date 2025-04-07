# CAR-AI

# Multi-Task Learning Pipeline for Protein Sequence Classification & Signal Prediction

## Overview
This project implements a multi-task deep learning pipeline that:
- Classifies protein sequences into 4 biological classes (A, B, C, D)
- Predicts three biochemical signal values (Signal-1, Signal-2, Signal-3)

The model uses semantic embeddings generated from pretrained Word2Vec models and is trained on labeled biological data.

---

## Folder Structure
```
SEQUENCES_WITH_SIGNALS/
├── input/                      # Raw sequence data with signals
├── Final Sequence Output/        # Signal-labeled data
├── Final Sequence Output_with_classes/ # Data with class labels (A-D)
├── Split Data/                # train/test splits + embeddings + saved model
├── validation results/        # Predictions on new unseen sequences
```

---

## Steps

### 1. Signal Labeling
**Script**: `label_signal_levels.py`
- Assigns labels (low, middle, high, very high) to each of Signal-1, Signal-2, Signal-3
- Saved to `Final Sequence Output/`

### 2. Class Label Assignment
**Script**: `assign_class_labels.py`
- Rule-based labeling to assign final class: A, B, C, or D based on signal labels
- Output saved in `Final Sequence Output_with_classes/`

### 3. Manual Class-Balanced Split
**Script**: `manual_split.py`
- Stratified sampling into train/test sets using manual class-based proportions
- Files saved as:
  - `Split Data/train_data.xlsx`
  - `Split Data/test_data.xlsx`

### 4. Embedding Generation (Word2Vec)
**Script**: `generate_embeddings.py`
- Uses pretrained GoogleNews Word2Vec (via `kagglehub`)
- Sequences are broken into 3-mers and embedded
- Output:
  - `train_embeddings.npy`
  - `test_embeddings.npy`

### 5. Model Training (Multi-task Learning)
Two variations:

#### Basic Model:
- Loss: `sparse_categorical_crossentropy` + `mse`
- Saved to `multi_task_model_with_weights.h5`

#### Final Model:
**Script**: `final_model_with_label_smoothing.py`
- Loss: `categorical_crossentropy` with `label_smoothing=0.1` + `mse`
- Optimizer: AdamW
- Sample weights used only for class imbalance in classification task
- Saved to: `multi_task_model.h5`

### 6. Evaluation
**Output Metrics:**
- Classification report (precision, recall, F1-score)
- Mean squared error per signal
- Plots:
  - `combined_actual_vs_predicted_signals.png`
  - `measured_vs_predicted_signals.png`
- Metrics JSON:
  - `evaluation_metrics_with_weights.json`

### 7. Validation on Unlabeled Data
**Script**: `validate_new_sequences.py`
- Loads `.xlsx` files from `validation-data-with-no-values/`
- Predicts class and signals
- Saves prediction results per file + combined scatter plot:
  - `combined_predicted_signals_scatter_with_legends.png`

---

## Requirements
- Python 3.9+
- TensorFlow / Keras
- Gensim
- Pandas, NumPy, Matplotlib
- Pretrained Word2Vec from GoogleNews (download via `kagglehub`)

---

## Author
**Nisha Chaudhary**  
Created as part of AI-based biological signal prediction and classification research.

---

## License
This code is for academic and research use only. Contact the author for licensing or reuse inquiries.

