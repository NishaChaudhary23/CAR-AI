# CARMSeD: Multi-Task CAR-T Sequence to Phenotype Prediction

**CARMSeD** is a multi-task 1D‑CNN model that takes CAR-T primary sequences as input and simultaneously:

- **Classifies** each sequence into one of three phenotypic classes: Low-, Medium-, or High-CARMSeD.
- **Regresses** three continuous signal values (Signal‑1, Signal‑2, Signal‑3) in original units (µg ml⁻¹).  

This repository contains scripts for data preprocessing, model training, evaluation, plotting, and inference.

---

## 📁 Repository Structure
```
├── train_carmsed.py       # Training pipeline (data load → model → fit → save)
├── plot_results.py        # Standalone plotting of train/val metrics & results
├── predict_carmsed.py     # Batch inference & summary plots on new sequences
│
├── model/                 # Outputs from training
│   ├── data/              # train.csv, val.csv (true vs. pred values)
│   ├── model_files/       # saved_model/, best_model.h5, history.pkl, scaler
│   └── figs/              # epoch curves, scatter, confusion, etc.
│
└── Predictions-output/    # Inference output mirror of input folder
    ├── <subfolder>/…      # *_predictions.csv per input file
    └── figs/              # summary plots: class dist, hist, boxplot
```

---

## 🚀 Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/CARMSeD.git
   cd CARMSeD
   ```
2. **Create Python environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Requirements*: TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, openpyxl

---

## 📊 Data Preparation

- Place your annotated Excel workbooks under:
  ```
  SEQUENCES_WITH_SIGNALS/results-april25/Final sequence output_with_classes/
  ```
  Each `.xlsx` must contain columns: `sequence`, `class` (L‑/M‑/H‑CARMSeD), and `Signal-1`, `Signal-2`, `Signal-3`.

- The **training script** will recursively load `.xlsx` files, clean missing labels, and drop stray temp workbooks.

---

## 🏋️‍♂️ Model Training

Run:
```bash
python train_carmsed.py
```
This will:

1. Load & clean all Excel files.  
2. Print class balance & signal stats.  
3. Encode amino­acid sequences (max length 1024).  
4. Scale signals (MinMax) & stratified train/val split.  
5. Build 1D‑CNN with two heads (classification & regression).  
6. Train for 30 epochs (CPU‑only by default).  
7. Save:
   - `model_files/saved_model/` (TensorFlow SavedModel)
   - `model_files/best_model.h5` (best checkpoint)
   - `model_files/history.pkl` (training history dict)
   - `model_files/signal_scaler.pkl` (signal scaler)
   - `data/train.csv`, `data/val.csv` (true vs. predicted values)

**Console prints** include dataset sizes, class counts, epoch metrics, and file‑save confirmations.

---

## 📈 Plotting Results

Use the standalone `plot_results.py` to regenerate publication‑quality figures anytime:

```bash
python plot_results.py
```

It reads `model/data/train.csv` & `val.csv` and `model/model_files/history.pkl`, then saves PNG & SVG:

- **Epoch curves**: classification accuracy & regression MSE
- **Train vs. Val scatter**: measured vs. predicted mean signal
- **Residual violin** per class
- **Confusion matrix** (validation)
- **Class counts** bar chart (validation)

Plots are saved under `model/figs/`.

---

## 🤖 Inference

Process new sequences and obtain predictions + summary plots:

```bash
python predict_carmsed.py
```

By default, it:

1. Recursively reads all `.csv`/`.xlsx` under:
   ```
   Sequences-for-prediction/
   ```
2. Predicts class probabilities & signals for each sequence.  
3. Writes `<filename>_predictions.csv` to:
   ```
   Predictions-output/
   ```
   preserving subfolders and all original columns.
4. Generates summary plots in `Predictions-output/figs/`:
   - Predicted class distribution (bar + counts)
   - Histogram of mean predicted signal
   - Boxplot of mean signal by class (with counts)

---

## 🧪 Model Architecture

- **Embedding**: 20 + 1 tokens → 128 D  
- **Conv layers**: 2×(Conv1D(256, k=5) → ReLU), plus MaxPool1D  
- **GlobalMaxPool1D** → shared representation  
- **Heads**:
  - Class: Dense(128) → Softmax(3)  
  - Signal: Dense(128) → Linear(3)  
- **Loss**: Weighted sum of cross‑entropy (×3) and MSE (×1)  
- **Optimizer**: Adam

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

