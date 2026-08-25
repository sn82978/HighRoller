'''
SUPERSEDED, don't run this anymore.

random split, not the same one the rest of the models use now (see data_preparation.py,
which pulls the split from BaselineModels instead). keeping this around for history.

splits data/polymarket/btc_update_15m_candles_15s into a train folder, val folder, and test folder randomly.

60% test, 20% val, 20% test
'''

import os
import shutil
import random

DATA_DIR = "/Users/shreyanakum/Documents/HighRoller/data/polymarket/btc_updown_15m_candles_15s"
OUTPUT_BASE_DIR = "/Users/shreyanakum/Documents/HighRoller/QLearning/data"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

splits = ["train", "val", "test"]

split_dirs = {}
for split in splits:
    target_dir = os.path.join(OUTPUT_BASE_DIR, split)
    os.makedirs(target_dir, exist_ok=True)
    split_dirs[split] = target_dir

# ignore .parquet files
all_files = os.listdir(DATA_DIR)
# csv_files = [f for f in all_files if f.endswith(".csv") and not f.endswith(".parquet")]
csv_files = [f for f in all_files if f.endswith(".parquet") and not f.endswith(".csv")]
# csv_files = all_files
# print(len(csv_files))
# shuffle randomly
random.seed(42) # i set a seed for reproducible splits
random.shuffle(csv_files)

total_files = len(csv_files)
train_end = int(total_files * TRAIN_RATIO)
val_end = train_end + int(total_files*VAL_RATIO)

train_files = csv_files[:train_end]
val_files = csv_files[train_end:val_end]
test_files = csv_files[val_end:]

file_splits = {"train": train_files, "val": val_files, "test": test_files}

# make a copy of the .csv file
print(f"total CSV files found {total_files}")

for split, files in file_splits.items():
    dest_dir = split_dirs[split]
    print(f"{len(files)} files to '{dest_dir}'")
    for filename in files:
        src_path = os.path.join(DATA_DIR, filename)
        # move to ./QLearning/data/train, ./QLearning/data/test, ./QLearning/data/val
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)
