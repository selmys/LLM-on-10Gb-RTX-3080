import json
import random

# Configuration
INPUT_FILE = "qa_dataset.jsonl"
TRAIN_FILE = "qa_train.jsonl"
VAL_FILE = "qa_val.jsonl"
TRAIN_SPLIT = 0.90  # 90% for training, 10% for validation

print(f"Reading {INPUT_FILE}...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

total_records = len(lines)
print(f"Total available rows found: {total_records}")

# Shuffle the records randomly so the order doesn't bias the splits
random.seed(42)  # Setting a seed makes the shuffle reproducible
random.shuffle(lines)

# Calculate split boundaries
split_idx = int(total_records * TRAIN_SPLIT)

train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

print(f"Splitting data: {len(train_lines)} training samples | {len(val_lines)} validation samples.")

# Write the training split
print(f"Saving to {TRAIN_FILE}...")
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

# Write the validation split
print(f"Saving to {VAL_FILE}...")
with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print("Dataset successfully split into Train and Validation sets!")
