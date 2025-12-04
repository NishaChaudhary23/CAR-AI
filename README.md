# CARMSeD: Multi-Task Learning for Synergy Class and Signal Prediction

This project implements the CARMSeD model pipeline described in the manuscript "AI-Guided CAR Designs and AKT3 Degradation Synergize to Enhance Bispecific and Trispecific CAR T Cell Persistence and Overcome Antigen Escape." for classifying CAR (Chimeric Antigen Receptor) sequences into functional synergy classes and predicting associated signal strengths. The system integrates sequence encoding, signal binning, class assignment, and a dual-head convolutional neural network for simultaneous classification and regression.
---

It provides an end-to-end reproducible workflow to:
- Label CAR sequences with functional signal tiers
- Assign synergy classes (L-CARMSeD / M-CARMSeD / H-CARMSeD)
- Train a deep learning model for joint signal regression and class prediction
- Run inference on new, unseen CAR sequences

---

Pipeline Overview:

Step 1: signal-label.py
    - Assigns Signal-1/2/3 labels (low/middle/high)

Step 2: CAMSeD-label.py
    - Assigns CARMSeD class labels (L/M/H)

Step 3: train.py
    - Trains a multi-task CNN on labeled data

Step 4: Prediction-on-given-sequence.py
    - Predicts signal and class for new sequences

---

Folder Structure:

CAR-AI/
├── raw_excel_data/                      # Sample Excel files for labeling

├── Final sequence output/              # Output from signal-labeling

├── Final sequence output_with_classes/ # Output from CAMSeD-labeling

├── Sequences-for-prediction/           # Excel files for inference

├── Predictions-output/                 # Predicted outputs and summary plots

├── model/                              # Trained model files, scalers, history

---

How to Run:

1. python signal-label.py
2. python CAMSeD-label.py
3. python train.py
4. python Prediction-on-given-sequence.py

All outputs will be saved automatically into the correct folders.

---

Dependencies:

Install all required packages using:
pip install -r requirements.txt

---

Data Availability:

Only a small set of sample Excel files is included in this capsule for demonstration purposes.
Full datasets and final performance metrics are described in the manuscript.
Please contact the corresponding author for access to the full data.

---

## 🧪 Signal Labeling Logic

### Signal-1

| Value Range | Label   |
|-------------|---------|
| < 0.5       | low     |
| 0.5–1.0     | middle  |
| > 1.0       | high    |

### Signal-2 & Signal-3

| Value Range | Label   |
|-------------|---------|
| < 0.25      | low     |
| 0.25–0.5    | middle  |
| > 0.5       | high    |

---

## 🏷️ Class Label Assignment (CARMSeD: L/M/H)

Each sequence is assigned to a CARMSeD functional class based on the combination of its three signal labels:

- **L-CARMSeD**: Predominantly low or middle signals  
- **M-CARMSeD**: Mixed signals with moderate strength  
- **H-CARMSeD**: At least one high signal, or strong combinations  

Refer to `CARMSeD-label.py` for the complete rule-based mapping logic.

---

## Sequence Encoding

- Amino acid sequences are truncated or padded to 1024 residues  
- Each residue is integer-encoded using a fixed 20-letter vocabulary  
- Unknown residues are mapped to a placeholder token  

---

## Model Architecture

- **Input**: Integer-encoded sequences  
- **Embedding Layer**: Learnable (128 dimensions)  
- **Convolutional Layers**:  
  - Conv1D → MaxPooling → Conv1D → GlobalMaxPooling  
- **Output 1**: 3-class softmax for CARMSeD classification  
- **Output 2**: Linear regression for Signal-1, Signal-2, and Signal-3

```python
loss = {
  'class_out': 'categorical_crossentropy',
  'signal_out': 'mse'
}
loss_weights = {'class_out': 3.0, 'signal_out': 1.0}
```

**Training settings**:  
- Stratified train/val split (80:20)  
- MinMaxScaler for signal normalization  
- Early stopping on validation accuracy  

---

## Evaluation & Visualization

- Classification accuracy and confusion matrix  
- MSE and R² scores for signal regression  
- Measured vs. predicted signal scatter plots  
- Residual distributions and class-wise signal profiles  

**Key results**:
- **Accuracy**: ~96%  
- **R²**: 0.87 (train), 0.83 (validation)  
- **RMSE**: < 0.1 µg mL⁻¹  

---

License:

This code is licensed under the MIT License.  
You are free to use, modify, and distribute it with proper attribution.

Copyright (c) 2025 Nisha Chaudhary
---

For questions or contributions, contact **Nisha Chaudhary** nisha152810@st.jmi.ac.in.
