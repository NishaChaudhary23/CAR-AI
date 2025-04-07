# Multi-Task Learning Pipeline for Sequence-Based Prediction

This project implements a multi-task deep learning workflow for classifying biological sequences and predicting associated signal values. The process integrates sequence preprocessing, signal labeling, class assignment, feature extraction via pretrained Word2Vec, and training a joint neural network model.

---

## 📁 Project Structure

```
SEQUENCES_WITH_SIGNALS/
├── input/                         # Raw Excel files with sequences and signal values
├── Final Sequence Output/        # Signal-labeled files
├── Final Sequence Output_with_classes/  # Files with assigned class labels (A/B/C/D)
├── Split Data/                   # Final train/test splits + embeddings
│   ├── train_data.xlsx
│   ├── test_data.xlsx
│   ├── train_embeddings.npy
│   └── test_embeddings.npy
├── multi_task_model.h5           # Trained model
├── measured_vs_predicted_signals.png    # Evaluation plots
└── evaluation_metrics_with_weights.json # Metrics saved after training
```

---

## 🧪 Signal Labeling Logic

### Signal-1:
| Value Range | Label      |
|-------------|------------|
| < 0.5       | low        |
| 0.5 – 1.0   | middle     |
| 1.0 – 1.5   | high       |
| > 1.5       | very high  |

### Signal-2 & Signal-3:
| Value Range | Label      |
|-------------|------------|
| < 0.1       | low        |
| 0.1 – 0.25  | middle     |
| 0.25 – 0.5  | high       |
| > 0.5       | very high  |

---

## 🏷️ Class Label Assignment (A, B, C, D)

Class labels are derived from the combination of signal labels:

- `D`: All 3 are `very high`, or all `high`, or at least 1 is `very high`
- `A`: All 3 are `low`
- `B`: 2 `low` + 1 `middle` **or** 2 `middle` + 1 `low`
- `C`: All `middle`, or if any signal is `high` (but not `very high`)

**Examples:**
| Signal-1_Label | Signal-2_Label | Signal-3_Label | Class |
|----------------|----------------|----------------|-------|
| low            | low            | low            | A     |
| low            | middle         | low            | B     |
| middle         | middle         | middle         | C     |
| high           | middle         | middle         | C     |
| very high      | middle         | middle         | D     |

---

## 🧬 Embedding Generation

- **Pretrained Word2Vec** model from Google News was downloaded using `kagglehub`
- Sequences were broken into overlapping k-mers (k=3)
- Word2Vec embeddings were averaged per sequence

---

## 🤖 Model Architecture

- **Inputs**: 300-dim embedding vectors
- **Shared Layers**: Dense + BatchNorm + Dropout
- **Output 1**: Classification into 4 classes using softmax
- **Output 2**: Regression for 3 signal values using linear activation

Trained using multi-task loss:
```python
loss = {
  'class_output': 'categorical_crossentropy' with label smoothing,
  'signal_output': 'mse'
}
```

With:
- Class weights (for handling imbalance)
- EarlyStopping + ReduceLROnPlateau

---

## 📈 Evaluation & Visualization

- Classification report (precision, recall, F1-score)
- MSE for each signal
- Predicted vs. Actual plots for signals (saved as `.png`)

---

## 💾 Final Outputs

- Trained model: `multi_task_model.h5`
- Metrics: `evaluation_metrics_with_weights.json`
- Plots: `measured_vs_predicted_signals.png`

---

## 🔁 Next Steps

- Deploy model for inference on new sequence data
- Visualize signal profiles across predicted classes
- Extend to larger or multiplexed datasets

---

For questions or contributions, contact **Nisha Chaudhary**.

