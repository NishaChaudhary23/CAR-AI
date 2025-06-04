# Multi-Task Learning Pipeline for CAR Sequence-Based Signal Prediction

This project implements a multi-task deep learning workflow (**CAR-AI**) for classifying CAR (Chimeric Antigen Receptor) sequences into functional synergy classes and predicting associated signal strengths. The system integrates sequence encoding, signal binning, class assignment, and a dual-head convolutional neural network for simultaneous classification and regression.

---

## 📁 Project Structure

```
SEQUENCES_WITH_SIGNALS/
├── input/                         # Raw Excel files with CAR sequences and signal values
├── Final sequence output/         # Signal-labeled files
├── Final sequence output_with_classes/  # Files with assigned L/M/H class labels (CARMSeD)
├── model_files/                   # Trained model + scaler + history
├── data/                          # Train/test predictions and true values
├── figs/                          # Evaluation plots
└── train.py / predict.py / plot_results.py  # Core scripts
```

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

Refer to `CAMSeD-label.py` for the complete rule-based mapping logic.

---

## 🧬 Sequence Encoding

- Amino acid sequences are truncated or padded to 1024 residues  
- Each residue is integer-encoded using a fixed 20-letter vocabulary  
- Unknown residues are mapped to a placeholder token  

---

## 🤖 Model Architecture

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

## 📈 Evaluation & Visualization

- Classification accuracy and confusion matrix  
- MSE and R² scores for signal regression  
- Measured vs. predicted signal scatter plots  
- Residual distributions and class-wise signal profiles  

**Key results**:
- **Accuracy**: ~96%  
- **R²**: 0.87 (train), 0.83 (validation)  
- **RMSE**: < 0.1 µg mL⁻¹  

---

## 💾 Final Outputs

- Trained model: `saved_model/`  
- Signal scaler: `signal_scaler.pkl`  
- Training history: `history.pkl`  
- Predictions: `train.csv`, `val.csv`  
- Plots: `figs/`  

---

## 🔁 Next Steps

- Inference on unseen CAR sequences  
- Visual explanation of model attention (future work)  
- Expansion to larger, modular CAR libraries  

---

For questions or contributions, contact **Nisha Chaudhary**.
