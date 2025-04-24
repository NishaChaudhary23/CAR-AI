import pandas as pd
import os
from collections import defaultdict

# Define input and output root folder paths
input_root = '/Users/nishachaudhary/Documents/others/TA/CAR-AI/New-data-december/SEQUENCES_WITH_SIGNALS/input/'
output_root = '/Users/nishachaudhary/Documents/others/TA/CAR-AI/New-data-december/SEQUENCES_WITH_SIGNALS/results-april25/Final sequence output/'

print("=== Sequence Labeling Pipeline ===")
print(f"Input root : {input_root}")
print(f"Output root: {output_root}\n")

# Initialize counters
total_sequence_count = 0
main_folder_sequence_count = defaultdict(int)
main_folder_label_counts = defaultdict(lambda: {
    'Signal-1_Label': defaultdict(int),
    'Signal-2_Label': defaultdict(int),
    'Signal-3_Label': defaultdict(int),
})
total_label_counts = {
    'Signal-1_Label': defaultdict(int),
    'Signal-2_Label': defaultdict(int),
    'Signal-3_Label': defaultdict(int),
}

# 3‑bin labeling functions
def label_signal_1(value):
    if value < 0.5:
        return 'low'
    elif 0.5 <= value <= 1:
        return 'middle'
    else:  # value > 1
        return 'high'

def label_signal_2_3(value):
    if value < 0.25:
        return 'low'
    elif 0.25 <= value <= 0.5:
        return 'middle'
    else:  # value > 0.5
        return 'high'

def update_label_counts(label_counts, updates):
    for lbl, cnt in updates.items():
        label_counts[lbl] += cnt

def process_excel_file(input_file, output_file, main_folder):
    global total_sequence_count

    print(f"\n--- Processing file: {input_file} ---")
    df = pd.read_excel(input_file)
    n_rows = len(df)
    print(f"Read {n_rows} sequences")

    # update totals
    total_sequence_count += n_rows
    main_folder_sequence_count[main_folder] += n_rows

    # Label Signal‑1
    if 'Signal-1' in df.columns:
        df['Signal-1_Label'] = df['Signal-1'].apply(label_signal_1)
        counts1 = df['Signal-1_Label'].value_counts().to_dict()
        print(f"  Signal-1_Label distribution: {counts1}")
        update_label_counts(main_folder_label_counts[main_folder]['Signal-1_Label'], counts1)
        update_label_counts(total_label_counts['Signal-1_Label'], counts1)
    else:
        print(f"  ⚠️ Warning: 'Signal-1' column missing")

    # Label Signal‑2
    if 'Signal-2' in df.columns:
        df['Signal-2_Label'] = df['Signal-2'].apply(label_signal_2_3)
        counts2 = df['Signal-2_Label'].value_counts().to_dict()
        print(f"  Signal-2_Label distribution: {counts2}")
        update_label_counts(main_folder_label_counts[main_folder]['Signal-2_Label'], counts2)
        update_label_counts(total_label_counts['Signal-2_Label'], counts2)
    else:
        print(f"  ⚠️ Warning: 'Signal-2' column missing")

    # Label Signal‑3
    if 'Signal-3' in df.columns:
        df['Signal-3_Label'] = df['Signal-3'].apply(label_signal_2_3)
        counts3 = df['Signal-3_Label'].value_counts().to_dict()
        print(f"  Signal-3_Label distribution: {counts3}")
        update_label_counts(main_folder_label_counts[main_folder]['Signal-3_Label'], counts3)
        update_label_counts(total_label_counts['Signal-3_Label'], counts3)
    else:
        print(f"  ⚠️ Warning: 'Signal-3' column missing")

    # Ensure output directory
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Ensured output directory exists: {out_dir}")

    # Save
    df.to_excel(output_file, index=False)
    print(f"Saved labeled file to: {output_file}")

# Walk through input tree
for root, _, files in os.walk(input_root):
    print(f"\nScanning folder: {root}")
    for fname in files:
        if not fname.lower().endswith('.xlsx'):
            continue
        in_path = os.path.join(root, fname)
        rel = os.path.relpath(root, input_root)
        out_path = os.path.join(output_root, rel, fname)
        main_folder = os.path.normpath(rel).split(os.sep)[0]
        process_excel_file(in_path, out_path, main_folder)

# Final summary
print("\n=== All files processed ===")
print(f"Total sequences processed: {total_sequence_count}\n")

print("Per‑folder totals and label counts:")
for fld, seq_cnt in main_folder_sequence_count.items():
    print(f"\nFolder: {fld}")
    print(f"  Sequences: {seq_cnt}")
    for sig, lbls in main_folder_label_counts[fld].items():
        print(f"  {sig}: {dict(lbls)}")

print("\nOverall label counts across all folders:")
for sig, lbls in total_label_counts.items():
    print(f"  {sig}: {dict(lbls)}")
