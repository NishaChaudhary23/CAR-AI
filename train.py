#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_carmsed.py

Standalone training script for Multi-Task CARMSeD model (CPU-only).
Loads Excel files, cleans & encodes data, builds & trains the model,
then saves the trained model, scaler, history, and train/val CSVs.

Usage:
  python train_carmsed.py
"""
import os
import glob
import pathlib
import warnings
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 0. Force CPU (comment out to enable GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for dev in ('GPU', 'MPS'):
    try:
        tf.config.set_visible_devices([], dev)
    except Exception:
        pass

# 1. Paths & constants
BASE_OUT = pathlib.Path(
    "/Users/nishachaudhary/Documents/others/TA/CAR-AI/"
    "New-data-december/SEQUENCES_WITH_SIGNALS/results-april25/model"
)
DATA_DIR  = BASE_OUT / 'data'
MODEL_DIR = BASE_OUT / 'model_files'
for p in (DATA_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

INPUT_FOLDER = (
    "/Users/nishachaudhary/Documents/others/TA/CAR-AI/"
    "New-data-december/SEQUENCES_WITH_SIGNALS/results-april25/"
    "Final sequence output_with_classes/"
)

SEQ_COL, CLASS_COL = 'sequence', 'class'
SIGNAL_COLS = ['Signal-1', 'Signal-2', 'Signal-3']
CLASS_NAMES = ['L-CARMSED', 'M-CARMSED', 'H-CARMSED']
class_map = {c: i for i, c in enumerate(CLASS_NAMES)}

MAX_LEN, EMBED_DIM = 1024, 128
BATCH_SIZE, EPOCHS = 64, 50
LOSS_WTS = {'class_out': 3.0, 'signal_out': 1.0}
TEST_SIZE, RAND_STATE = 0.2, 42

# 2. Load and clean Excel files
dfs = []
for fp in glob.glob(os.path.join(INPUT_FOLDER, '**', '*.xlsx'), recursive=True):
    fn = os.path.basename(fp)
    if fn.startswith(('~$', '.$')):
        continue  # skip temp files
    try:
        df = pd.read_excel(fp, engine='openpyxl')
    except Exception:
        warnings.warn(f"Skipping unreadable file: {fp}")
        continue
    if not {SEQ_COL, CLASS_COL, *SIGNAL_COLS}.issubset(df.columns):
        continue
    df = df[[SEQ_COL, CLASS_COL] + SIGNAL_COLS].copy()
    df = df[df[CLASS_COL].notna() & df[CLASS_COL].astype(str).str.strip().ne('')]
    dfn = df.reset_index(drop=True)
    dfn['source_file'] = fp
    dfn['excel_row'] = dfn.index + 2
    dfs.append(dfn)
if not dfs:
    raise RuntimeError('No valid Excel files found in INPUT_FOLDER')

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data)} sequences from {len(dfs)} files.")
# class balance
print("Class distribution:")
print(data[CLASS_COL].value_counts(), "\n")
# signal stats
print("Signal statistics:")
print(data[SIGNAL_COLS].describe(), "\n")

# 3. Clean & encode class labels
def normalize_label(s: str) -> str:
    s = str(s).strip().upper()
    return s.translate(str.maketrans({
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-',
        '\u2014': '-', '\u2015': '-', '\u2212': '-',  # various dashes
        '\u00A0': ' ',  # NBSP
        '\u200B': ''    # zero-width space
    }))

data['clean_class'] = data[CLASS_COL].apply(normalize_label)
mapped = data['clean_class'].map(class_map)
if mapped.isna().any():
    bad = data.loc[mapped.isna(), ['clean_class','source_file','excel_row']]
    print("Unmapped class labels found:")
    print(bad.groupby('clean_class').size())
    raise ValueError('Please fix class labels or extend class_map')
y_cls = mapped.values.astype(int)
y_cls_cat = utils.to_categorical(y_cls, num_classes=len(CLASS_NAMES))

# 4. Encode sequences
AMINO = list("ACDEFGHIKLMNPQRSTVWY")
vocab = {aa: idx+1 for idx, aa in enumerate(AMINO)}
vocab['X'] = len(vocab) + 1
def encode_seq(seq: str) -> list:
    seq = str(seq)[:MAX_LEN]
    ids = [vocab.get(c, vocab['X']) for c in seq]
    return ids + [0] * (MAX_LEN - len(ids))
X = np.array([encode_seq(s) for s in data[SEQ_COL]], dtype='int32')
print(f"Encoded sequences shape: {X.shape}")

# 5. Scale signals & split
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(data[SIGNAL_COLS])
data['sig3_bin'] = pd.qcut(data['Signal-3'], q=6, labels=False)
strata = data['clean_class'] + '_' + data['sig3_bin'].astype(str)
X_tr, X_v, yc_tr, yc_v, ys_tr, ys_v = train_test_split(
    X, y_cls_cat, y_scaled,
    test_size=TEST_SIZE,
    random_state=RAND_STATE,
    stratify=strata
)
print(f"Train samples: {X_tr.shape[0]}, Validation samples: {X_v.shape[0]}")
print("Train class counts:")
print(pd.Series(yc_tr.argmax(1)).map({i:c for i,c in enumerate(CLASS_NAMES)}).value_counts(), "\n")
print("Validation class counts:")
print(pd.Series(yc_v.argmax(1)).map({i:c for i,c in enumerate(CLASS_NAMES)}).value_counts(), "\n")
del data['sig3_bin']

# 6. Build the multi-task model
inp = layers.Input(shape=(MAX_LEN,), name='seq_input')
x = layers.Embedding(len(vocab)+1, EMBED_DIM, mask_zero=True)(inp)
x = layers.Conv1D(256, 5, padding='same', activation='relu')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(256, 5, padding='same', activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
cls_branch = layers.Dense(128, activation='relu')(x)
cls_out = layers.Dense(len(CLASS_NAMES), activation='softmax', name='class_out')(cls_branch)
reg_branch = layers.Dense(128, activation='relu')(x)
reg_out = layers.Dense(len(SIGNAL_COLS), activation='linear', name='signal_out')(reg_branch)
model = models.Model(inputs=inp, outputs=[cls_out, reg_out])
model.compile(
    optimizer='adam',
    loss={'class_out':'categorical_crossentropy','signal_out':'mse'},
    loss_weights=LOSS_WTS,
    metrics={'class_out':'accuracy','signal_out':'mse'}
)
model.summary()

# 7. Train
ckpt = callbacks.ModelCheckpoint(
    filepath=str(MODEL_DIR/'best_model.h5'),
    save_best_only=True,
    monitor='val_class_out_accuracy',
    mode='max'
)
history = model.fit(
    X_tr, {'class_out': yc_tr, 'signal_out': ys_tr},
    validation_data=(X_v, {'class_out': yc_v, 'signal_out': ys_v}),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[ckpt], verbose=2
)
# persist history
history_path = MODEL_DIR / 'history.pkl'
joblib.dump(history.history, history_path)
print(f"Saved training history to {history_path}")

# 8. Save model & scaler
model.save(MODEL_DIR/'saved_model')
joblib.dump(scaler, MODEL_DIR/'signal_scaler.pkl')
print("Model and scaler saved.")

# 9. Save train/val CSVs
train_preds = model.predict(X_tr, verbose=0)
val_preds   = model.predict(X_v, verbose=0)
ys_tr_orig  = scaler.inverse_transform(ys_tr)
ys_v_orig   = scaler.inverse_transform(ys_v)
pr_tr_orig  = scaler.inverse_transform(train_preds[1])
pr_v_orig   = scaler.inverse_transform(val_preds[1])

df_tr = pd.DataFrame({
    'true_class': yc_tr.argmax(1), 'pred_class': train_preds[0].argmax(1),
    **{f'True_{c}': ys_tr_orig[:,i] for i,c in enumerate(SIGNAL_COLS)},
    **{f'Pred_{c}': pr_tr_orig[:,i] for i,c in enumerate(SIGNAL_COLS)}
})
df_tr.to_csv(DATA_DIR/'train.csv', index=False)

df_val = pd.DataFrame({
    'true_class': yc_v.argmax(1), 'pred_class': val_preds[0].argmax(1),
    **{f'True_{c}': ys_v_orig[:,i] for i,c in enumerate(SIGNAL_COLS)},
    **{f'Pred_{c}': pr_v_orig[:,i] for i,c in enumerate(SIGNAL_COLS)}
})
df_val.to_csv(DATA_DIR/'val.csv', index=False)
print("Training complete – CSVs saved to data folder.")
