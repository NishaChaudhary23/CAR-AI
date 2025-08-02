#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 11:57:09 2025

@author: nishachaudhary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Standalone inference script for Multi-Task CARMSeD model.
Recursively processes each CSV/XLSX (including all sheets) in INPUT_FOLDER,
predicts class & signals, and writes *_predictions.csv into OUTPUT_FOLDER,
preserving subfolder structure and input columns. Also generates summary plots.
"""

import os
import pathlib
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'


# 1) Configuration
INPUT_FOLDER = pathlib.Path("Sequences-for-prediction")
OUTPUT_FOLDER = pathlib.Path("Predictions-output")
MODEL_DIR = pathlib.Path("model/model_files")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_FOLDER / 'figs'
FIG_DIR.mkdir(exist_ok=True)

# Plot settings
dpi = 600
sns.set_context('paper')
plt.rcParams.update({'font.size':18})

# 2) Constants (must match training)
MAX_LEN = 1024
AMINO = list("ACDEFGHIKLMNPQRSTVWY")
vocab = {aa: idx+1 for idx, aa in enumerate(AMINO)}
vocab['X'] = len(vocab) + 1
CLASS_NAMES = ['L-CARMSED', 'M-CARMSED', 'H-CARMSED']

# 3) Utility: encode sequence to fixed length
def encode_seq(seq: str):
    s = str(seq)[:MAX_LEN]
    ids = [vocab.get(c, vocab['X']) for c in s]
    return ids + [0] * (MAX_LEN - len(ids))

# 4) Load model & scaler
print(f"Loading model and scaler from {MODEL_DIR} …")
model = load_model(MODEL_DIR / 'saved_model')
scaler = joblib.load(MODEL_DIR / 'signal_scaler.pkl')
print("Model and scaler loaded\n")

# 5) Gather all files
input_paths = list(INPUT_FOLDER.rglob('*.csv')) + list(INPUT_FOLDER.rglob('*.xlsx'))
print(f"Found {len(input_paths)} files under {INPUT_FOLDER}\n")

def process_df(df_in, name, out_dir):
    if 'sequence' not in df_in.columns:
        print(f"  No 'sequence' column in sheet/file {name}; skipping")
        return
    seqs = df_in['sequence'].dropna().astype(str)
    print(f"  Encoding {len(seqs)} sequences in {name} …")
    X = np.vstack([encode_seq(s) for s in seqs])
    cls_probs, sig_scaled = model.predict(X, verbose=0)
    pred_idx = np.argmax(cls_probs, axis=1)
    sig_pred = scaler.inverse_transform(sig_scaled)

    df_pred = df_in.reset_index(drop=True)
    df_pred['predicted_class']  = [CLASS_NAMES[i] for i in pred_idx]
    df_pred['class_prob_L']     = cls_probs[:,0]
    df_pred['class_prob_M']     = cls_probs[:,1]
    df_pred['class_prob_H']     = cls_probs[:,2]
    df_pred['Signal-1_pred']    = sig_pred[:,0]
    df_pred['Signal-2_pred']    = sig_pred[:,1]
    df_pred['Signal-3_pred']    = sig_pred[:,2]
    df_pred['Signal_mean_pred'] = sig_pred.mean(axis=1)

    out_fp = out_dir / f"{name}_predictions.csv"
    df_pred.to_csv(out_fp, index=False)
    print(f"  → Wrote predictions to {out_fp}")

# 6) Process each file (CSVs and multi-sheet Excels)
for fp in input_paths:
    rel = fp.relative_to(INPUT_FOLDER)
    out_dir = OUTPUT_FOLDER / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {fp.relative_to(INPUT_FOLDER)}")

    if fp.suffix.lower() == '.csv':
        try:
            df_in = pd.read_csv(fp)
            process_df(df_in, fp.stem, out_dir)
        except Exception:
            print(f"  Skip unreadable CSV: {fp}")
    else:
        try:
            sheets = pd.read_excel(fp, engine='openpyxl', sheet_name=None)
            for sheet_name, df_sheet in sheets.items():
                name = f"{fp.stem}_{sheet_name}"
                print(f"  Sheet: {sheet_name} ({len(df_sheet)} rows)")
                process_df(df_sheet, name, out_dir)
        except Exception:
            print(f"  Skip unreadable Excel: {fp}")
    print()

print("All files processed\n")

# 7) Aggregate and summary plotting
pred_files = list(OUTPUT_FOLDER.rglob('*_predictions.csv'))
all_df = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
print(f"Total predictions aggregated: {len(all_df)} rows from {len(pred_files)} files\n")

# 7a) Class distribution bar chart
plt.figure(figsize=(8,6))
ax = sns.countplot(x='predicted_class', data=all_df,
                   order=CLASS_NAMES, palette='pastel')
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x()+p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=16)
plt.title('Predicted Class Distribution')
plt.tight_layout()
plt.savefig(FIG_DIR/'pred_class_distribution.png', dpi=dpi)
plt.savefig(FIG_DIR/'pred_class_distribution.svg')
plt.close()

# 7b) Histogram of mean predicted signal
plt.figure(figsize=(8,6))
sns.histplot(all_df['Signal_mean_pred'], kde=True, bins=30)
plt.title('Distribution of Predicted Mean Signal')
plt.tight_layout()
plt.savefig(FIG_DIR/'signal_mean_hist.png', dpi=dpi)
plt.savefig(FIG_DIR/'signal_mean_hist.svg')
plt.close()

# 7c) Boxplot of mean signal by class
plt.figure(figsize=(6,6))
ax = sns.boxplot(x='predicted_class', y='Signal_mean_pred',
                 data=all_df, order=CLASS_NAMES,
                 palette={'L-CARMSED':'#1f77b4',
                          'M-CARMSED':'#ff7f0e',
                          'H-CARMSED':'#2ca02c'})
counts = all_df['predicted_class'].value_counts().reindex(CLASS_NAMES).values
ymin, ymax = ax.get_ylim()
for i, cnt in enumerate(counts):
    ax.text(i, ymax - (ymax-ymin)*0.05, str(cnt),
            ha='center', va='bottom', fontsize=16)
plt.title('Signal for unknown sequences', fontsize=22)
plt.xlabel('Class', fontsize=20)
plt.ylabel('Predicted Signal', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(FIG_DIR/'signal_mean_boxplot.png', dpi=dpi)
plt.savefig(FIG_DIR/'signal_mean_boxplot.svg')
plt.close()

print(f"Summary plots saved to {FIG_DIR}")
