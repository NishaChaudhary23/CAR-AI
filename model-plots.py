#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py

Standalone plotting script for CARMSeD results.
Reads saved train.csv, val.csv, and history.pkl, then produces and saves:
 1. Epoch curves (classification accuracy & regression MSE)
 2. Combined scatter (Train vs Validation) with black/pink markers
 3. Separate measured vs predicted scatter for L, M, H classes
 4. Residual violin per class (Validation)
 5. Class counts bar chart (Validation)
 6. Confusion matrix (Validation)

Usage:
  python plot_results.py

Ensure BASE_DIR matches the output folder from train_carmsed.py,
and that history (history.pkl) is saved there.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Paths & config
BASE_DIR = ("/Users/nishachaudhary/Documents/others/TA/CAR-AI/New-data-december/SEQUENCES_WITH_SIGNALS/results-april25/model")
DATA_DIR = os.path.join(BASE_DIR, 'data')
FIG_DIR  = os.path.join(BASE_DIR, 'figs')
MODEL_DIR = os.path.join(BASE_DIR, 'model_files')
os.makedirs(FIG_DIR, exist_ok=True)

# Plot settings
dpi = 600
sns.set_context('paper')
plt.rcParams.update({'font.size':20})
size = (6,6)

# Class labels
CLASS_NAMES = ['L-CARMSED','M-CARMSED','H-CARMSED']
CLASS_MAP = {i: name for i,name in enumerate(CLASS_NAMES)}

# 2. Load history
hist_path = os.path.join(MODEL_DIR, 'history.pkl')
if not os.path.exists(hist_path):
    raise FileNotFoundError(f"History file not found: {hist_path}")
hist = joblib.load(hist_path)

# 3. Epoch curves: classification accuracy & regression MSE
# Classification accuracy
# Epoch‑accuracy plot with final values in the legend
final_tr_acc = hist['class_out_accuracy'][-1]
final_va_acc = hist['val_class_out_accuracy'][-1]
plt.figure(figsize=size)
plt.plot(hist['class_out_accuracy'],   label=f'Train Acc=({final_tr_acc:.2f})')
plt.plot(hist['val_class_out_accuracy'], label=f'Val Acc=({final_va_acc:.2f})')
plt.xlabel('Epoch',    fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Model accuracy', fontsize=24)
plt.legend(fontsize=18)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'epoch_accuracy.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR, 'epoch_accuracy.svg'))
plt.close()

# Regression MSE
# Epoch‑MSE plot with final MSE values in the legend
final_tr_mse = hist['signal_out_mse'][-1]
final_va_mse = hist['val_signal_out_mse'][-1]
plt.figure(figsize=size)
plt.plot(
    hist['signal_out_mse'],
    label=f'Train MSE ({final_tr_mse:.3f})'
)
plt.plot(
    hist['val_signal_out_mse'],
    label=f'Val MSE   ({final_va_mse:.3f})'
)
plt.xlabel('Epoch', fontsize=22)
plt.ylabel('MSE',   fontsize=22)
plt.title('Regression MSE', fontsize=22)
plt.legend(fontsize=18)
plt.gca().tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'epoch_mse.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR,'epoch_mse.svg'))
plt.close()

# 4. Load data
df_tr = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))

# Compute per-sequence means
true_cols = [c for c in df_val.columns if c.startswith('True_Signal-')]
pred_cols = [c for c in df_val.columns if c.startswith('Pred_Signal-')]

df_tr['true_mean'] = df_tr[true_cols].mean(axis=1)
df_tr['pred_mean'] = df_tr[pred_cols].mean(axis=1)

df_val['true_mean'] = df_val[true_cols].mean(axis=1)
df_val['pred_mean'] = df_val[pred_cols].mean(axis=1)

# 5. Combined scatter (Train vs Validation) with 2% padding on limits
r2_tr = r2_score(df_tr['true_mean'], df_tr['pred_mean'])
r2_va = r2_score(df_val['true_mean'], df_val['pred_mean'])

plt.figure(figsize=size)
plt.scatter(
    df_tr['true_mean'], df_tr['pred_mean'],
    color='k', s=80, alpha=0.6,
    label=f'Train (R²={r2_tr:.2f})'
)
plt.scatter(
    df_val['true_mean'], df_val['pred_mean'],
    color='#FF69B4', s=80, alpha=0.6,
    label=f'Val (R²={r2_va:.2f})'
)
# compute overall min/max and add 2% padding
all_true = np.concatenate([df_tr['true_mean'], df_val['true_mean']])
all_pred = np.concatenate([df_tr['pred_mean'], df_val['pred_mean']])
mn = min(all_true.min(), all_pred.min())
mx = max(all_true.max(), all_pred.max())
pad = (mx - mn) * 0.02
# plot 1:1 line and set padded limits
plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], 'k--', linewidth=1)
plt.xlim(mn - pad, mx + pad)
plt.ylim(mn - pad, mx + pad)
plt.gca().set_aspect('equal', 'box')
# **increase tick count**
from matplotlib.ticker import MaxNLocator 
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.xlabel('Measured', fontsize=22)
plt.ylabel('Predicted', fontsize=22)
plt.title('Measured vs Predicted–Train & Val', fontsize=22)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'scatter_train_val.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR, 'scatter_train_val.svg'))
plt.close()


# 5b. All classes combined scatter (Validation only)
palette = {'L-CARMSED':'#1f77b4','M-CARMSED':'#ff7f0e','H-CARMSED':'#2ca02c'}
plt.figure(figsize=size)
for idx, cls_name in CLASS_MAP.items():
    mask = df_val['true_class'] == idx
    plt.scatter(
        df_val.loc[mask,'true_mean'], df_val.loc[mask,'pred_mean'],
        s=80, alpha=0.7, color=palette[cls_name], edgecolors='k', linewidths=0.5,
        label=cls_name
    )
lims = [
    min(df_val['true_mean'].min(), df_val['pred_mean'].min()),
    max(df_val['true_mean'].max(), df_val['pred_mean'].max())
]
plt.plot(lims, lims, 'k--', linewidth=1)
plt.xlim(lims); plt.ylim(lims)
plt.gca().set_aspect('equal','box')
plt.xlabel('Measured', fontsize=22)
plt.ylabel('Predicted', fontsize=22)
plt.title('Prediction on Val', fontsize=22)
plt.legend(fontsize=18); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'scatter_all_classes.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR,'scatter_all_classes.svg'))
plt.close()

# 6. Measured vs Predicted per class
for i, cls_name in CLASS_MAP.items():
    mask = df_val['true_class'] == i
    plt.figure(figsize=size)
    plt.scatter(
        df_val.loc[mask,'true_mean'], df_val.loc[mask,'pred_mean'],
        color='#FF69B4', s=80, alpha=0.6, edgecolors='k'
    )
    lims=[
        min(df_val.loc[mask,'true_mean'].min(), df_val.loc[mask,'pred_mean'].min()),
        max(df_val.loc[mask,'true_mean'].max(), df_val.loc[mask,'pred_mean'].max())
    ]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.gca().set_aspect('equal','box')
    plt.xlabel('Measured', fontsize=22); plt.ylabel('Predicted', fontsize=22)
    plt.title(f'Measured vs Predicted – {cls_name}', fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,f'scatter_{cls_name.lower()}.png'), dpi=dpi)
    plt.savefig(os.path.join(FIG_DIR,f'scatter_{cls_name.lower()}.svg'))
    plt.close()

# 7. Residual violin
resid = df_val['pred_mean'] - df_val['true_mean']
plt.figure(figsize=size)
sns.violinplot(
    x=[CLASS_MAP[c] for c in df_val['true_class']],
    y=resid, palette='Pastel1', cut=0, inner='box'
)
plt.axhline(0, ls='--', c='k')
plt.title('Residual distribution per class', fontsize=22)
plt.ylabel('Pred - Measured (µg ml⁻¹)', fontsize=22); plt.xlabel('Class', fontsize=22)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'violin_residuals.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR,'violin_residuals.svg'))
plt.close()

# 8. Class counts bar chart
counts_t = df_val['true_class'].value_counts().sort_index()
counts_p = df_val['pred_class'].value_counts().sort_index()
plt.figure(figsize=size)
inds=np.arange(len(CLASS_NAMES)); width=0.35
plt.bar(inds-width/2, counts_t.values, width, label='Measured', color='#4F81BD')
plt.bar(inds+width/2, counts_p.values, width, label='Predicted', color='#C0504D')
plt.xticks(inds, CLASS_NAMES); plt.ylabel('Number of sequences', fontsize=22)
plt.title('Measured vs Predicted counts per class', fontsize=22)
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,'counts_val.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR,'counts_val.svg'))
plt.close()

# 9. Confusion matrix
cm = confusion_matrix(df_val['true_class'], df_val['pred_class'])
fig, ax = plt.subplots(figsize=size)

# draw heatmap without its own colorbar
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)

ax.set_title('Confusion Matrix (Validation)')

# now add a smaller colorbar
# fraction=width of cbar axes as fraction of parent; pad=space; shrink=scale length
cbar = fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04, shrink=0.7)
cbar.ax.tick_params(labelsize=18)  # optional: adjust tick label size

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_val.png'), dpi=dpi)
plt.savefig(os.path.join(FIG_DIR, 'confusion_val.svg'))
plt.close()


print(f"Plots saved in '{FIG_DIR}'")
