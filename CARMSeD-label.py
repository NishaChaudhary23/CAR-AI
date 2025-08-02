#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:24:36 2025
@author: nishachaudhary

This script reads all Excel sheets under `input_folder`, 
classifies each row into one of the CARMSeD classes based 
on (Signal‑1_Label, Signal‑2_Label, Signal‑3_Label), and writes 
the augmented sheets out to a mirrored structure under `output_folder`.
"""

import pandas as pd
import os
from collections import defaultdict

# ---------------- CONFIG ----------------
input_folder  = "Final sequence output"
output_folder = "Final sequence output_with_classes"
# ----------------------------------------

print("=== CARMSeD Classification Pipeline ===")
print(f"Input  folder: {input_folder}")
print(f"Output folder: {output_folder}\n")

# Ensure output root exists
os.makedirs(output_folder, exist_ok=True)

# 27 valid combinations → CARMSeD class
label_combinations = {
    ('low','low','low')       : 'L-CARMSeD',
    ('low','low','middle')    : 'L-CARMSeD',
    ('low','low','high')      : 'H-CARMSeD',
    ('low','middle','low')    : 'L-CARMSeD',
    ('low','middle','middle') : 'L-CARMSeD',
    ('low','middle','high')   : 'H-CARMSeD',
    ('low','high','low')      : 'M-CARMSeD',
    ('low','high','middle')   : 'M-CARMSeD',
    ('low','high','high')     : 'H-CARMSeD',
    ('middle','low','low')    : 'L-CARMSeD',
    ('middle','low','middle') : 'L-CARMSeD',
    ('middle','low','high')   : 'H-CARMSeD',
    ('middle','middle','low') : 'L-CARMSeD',
    ('middle','middle','middle'): 'M-CARMSeD',
    ('middle','middle','high'): 'H-CARMSeD',
    ('middle','high','low')   : 'H-CARMSeD',
    ('middle','high','middle'): 'M-CARMSeD',
    ('middle','high','high')  : 'H-CARMSeD',
    ('high','low','low')      : 'M-CARMSeD',
    ('high','low','middle')   : 'M-CARMSeD',
    ('high','low','high')     : 'H-CARMSeD',
    ('high','middle','low')   : 'M-CARMSeD',
    ('high','middle','middle'): 'M-CARMSeD',
    ('high','middle','high')  : 'H-CARMSeD',
    ('high','high','low')     : 'M-CARMSeD',
    ('high','high','middle')  : 'M-CARMSeD',
    ('high','high','high')    : 'H-CARMSeD',
}

def classify_row(row):
    key = (row['Signal-1_Label'], row['Signal-2_Label'], row['Signal-3_Label'])
    return label_combinations.get(key, 'undefined')

# For per-file and total counts
class_distribution_per_file = {}
total_class_counts = defaultdict(int)

# Walk input tree
for root, dirs, files in os.walk(input_folder):
    print(f"Scanning folder: {root}")
    for filename in files:
        # skip hidden and non‑xlsx
        if filename.startswith('.') or not filename.lower().endswith('.xlsx'):
            continue

        in_path = os.path.join(root, filename)
        print(f"\n→ Processing file: {in_path}")

        try:
            df = pd.read_excel(in_path)
            n = len(df)
            print(f"  Rows read: {n}")

            # Drop any old 'class' columns to avoid duplication
            df = df.loc[:, ~df.columns.str.startswith('class')]

            # Check required label columns
            required = {'Signal-1_Label','Signal-2_Label','Signal-3_Label'}
            if not required.issubset(df.columns):
                missing = required - set(df.columns)
                print(f"  ⚠️ Missing columns {missing}; skipping file.")
                continue

            # Classify and tally
            df['class'] = df.apply(classify_row, axis=1)
            counts = df['class'].value_counts().to_dict()
            print(f"  Class distribution: {counts}")

            # Save per‑file and update totals
            class_distribution_per_file[filename] = counts
            for cls, c in counts.items():
                total_class_counts[cls] += c

            # Write out
            rel = os.path.relpath(root, input_folder)
            out_path = os.path.join(output_folder, rel, filename)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_excel(out_path, index=False)
            print(f"  Saved to: {out_path}")

        except Exception as e:
            print(f"  🚫 Error processing {filename}: {e}")

# Final summaries
print("\n=== Summary of Class Counts Per File ===")
for fname, dist in class_distribution_per_file.items():
    print(f"{fname}: {dist}")

print("\n=== Total Class Counts Across All Files ===")
for cls, total in total_class_counts.items():
    print(f"{cls}: {total}")

print("\nDone.")