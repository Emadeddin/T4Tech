#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# -*- coding: utf-8 -*-
"""
T4Tech: Machine & Deep Learning for High-Fidelity Grid Anomaly Classification
-----------------------------------------------------------------------------

Project
  T4Tech provides open, reproducible datasets and reference tooling for training
  and evaluating ML/DL models on inverter-rich power systems using *time-domain*
  signals (4.8 kHz). The repository includes a Zone-2 training set and two
  unlabeled streaming event files for real-time benchmarking.

Datasets (repo root or ./data)
  • zone2_training_dataset.csv   — ~22 s, ≈105,768 rows @ 4,800 Hz
    - Features (14): MU23_{I23[a,b,c,n], V2_[a,b,c]n}, MU32_{I32[a,b,c,n], V32_[a,b,c]n}
    - Label: integer in {0..17} (see class map below)
  • event1.csv, event2.csv       — 6 s each @ 4,800 Hz (UNLABELED streams)

Class taxonomy (0..17)
  0  Normal Operation
  1  SLG A–N          2  SLG B–N         3  SLG C–N
  4  CT Ratio MU32    5  LL A–B          6  LL A–C          7  LL B–C
  8  CT Ratio MU23    9  DLG AB–N        10 DLG AC–N        11 DLG BC–N
  12 PT Ratio MU32    13 PT Ratio MU23   14 3φ AB–C         15 3φ ABC–N
  16 GPS Spoof MU32   17 GPS Spoof MU23

Intended use
  • Supervised training on zone2_training_dataset.csv
  • Unlabeled streaming evaluation on event1.csv / event2.csv
  • Optional one-cycle (80-sample) smoother + confidence threshold (e.g., τ=0.6)
    — choose *causal* (trailing) or *centered* and document the induced delay.

Authors & Affiliation
  Emad Abukhousa, Syed Sohail Feroz Syed Afroz, Fahad Alsaeed,
  Abdulaziz Qwbaiban, Saman Zonouz, A. P. Sakis Meliopoulos
  Power System Control and Automation Laboratory (PSCAL),
  School of ECE, Georgia Institute of Technology, Atlanta, USA

Contact
  emadak@gatech.edu, safroz7@gatech.edu, sakis.m@gatech.edu

License
  MIT License. Copyright (c) 2025 T4Tech contributors.

How to cite (IEEE)
  E. Abukhousa et al., “The Wisdom of the Crowd: High-Fidelity Classification of
  Cyber-Attacks and Faults in Power Systems Using Ensemble and Machine Learning,”
  in Proc. IEEE PES ISGT Middle East, Dubai, UAE, Nov. 23–26, 2025. (to appear)

  E. A. Abukhousa, S. S. F. S. Afroz, F. Alsaeed, A. Qwbaiban, and A. P. S. Meliopoulos,
  “Centralized Dynamic State Estimation Algorithm for Detecting and Distinguishing Faults
  and Cyber Attacks in Power Systems,” arXiv:2508.02102, 2025. doi:10.48550/arXiv.2508.02102

BibTeX (optional)
  @inproceedings{abukhousa2025wisdom,
    title={The Wisdom of the Crowd: High-Fidelity Classification of Cyber-Attacks and Faults
           in Power Systems Using Ensemble and Machine Learning},
    author={Abukhousa, Emadeldin A. and Syed Afroz, Syed Sohail Feroz and Alsaeed, Fahad
            and Qwbaiban, Abdulaziz and Zonouz, Saman and Meliopoulos, A. P. Sakis},
    booktitle={Proc. IEEE PES ISGT Middle East},
    address={Dubai, UAE},
    year={2025},
    note={to appear}
  }

  @article{abukhousa2025centralized,
    title={Centralized Dynamic State Estimation Algorithm for Detecting and Distinguishing
           Faults and Cyber Attacks in Power Systems},
    author={Abukhousa, Emadeldin A. and Syed Afroz, Syed Sohail Feroz and Alsaeed, Fahad
            and Qwbaiban, Abdulaziz and Meliopoulos, A. P. Sakis},
    journal={arXiv preprint arXiv:2508.02102},
    year={2025},
    doi={10.48550/arXiv.2508.02102}
  }

Disclaimer
  This dataset and code are provided “as is,” without warranties of any kind.
  Please verify applicability to your system, adhere to local safety/operational
  standards, and cite the works above when publishing results.

------------------------------------------------------------------------------
"""


import sys
from pathlib import Path
import pandas as pd

# ---------- Config ----------
# Filenames (root of repo)
TRAIN_PATH = Path("zone2_training_dataset.csv")
EVENT1_PATH = Path("event1.csv")
EVENT2_PATH = Path("event2.csv")


from pathlib import Path
import sys
import pandas as pd

# ----------------------- Configuration -----------------------

TRAIN_PATH = Path("zone2_training_dataset.csv")
EVENT1_PATH = Path("event1.csv")
EVENT2_PATH = Path("event2.csv")

# 18-class taxonomy (plus -1 for unseen)
CLASS_MAP = {
    0:  "Normal Operation",
    1:  "Single Line Fault A–N",
    2:  "Single Line Fault B–N",
    3:  "Single Line Fault C–N",
    4:  "CT Ratio Attack on MU32",
    5:  "Double Line Fault A–B",
    6:  "Double Line Fault A–C",
    7:  "Double Line Fault B–C",
    8:  "CT Ratio Attack on MU23",
    9:  "DLG Fault AB–N",
    10: "DLG Fault AC–N",
    11: "DLG Fault BC–N",
    12: "PT Ratio Attack on MU32",
    13: "PT Ratio Attack on MU23",
    14: "3 Lines Fault AB–C",
    15: "3 Lines Fault ABC–N",
    16: "GPS Spoofing on MU32",
    17: "GPS Spoofing on MU23",
    -1: "Unseen / Out-of-Zone"
}

# Column name candidates
TIME_CANDIDATES  = ["time","t","timestamp","Time","TIME","sec","seconds"]
LABEL_CANDIDATES = ["label","Labels","class","Class","y","target"]

# -------------------------------------------------------------

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}")
        sys.exit(1)

def find_col(cols, candidates):
    # exact then case-insensitive
    for c in candidates:
        if c in cols: return c
    lower = [c.lower() for c in cols]
    for c in candidates:
        c_low = c.lower()
        if c_low in lower:
            return cols[lower.index(c_low)]
    return None

def print_headers(df: pd.DataFrame, title: str):
    print(f"\n# Columns in {title}")
    for i, c in enumerate(df.columns):
        print(f"  {i:02d}: {c}")

def describe_time(df: pd.DataFrame, time_col: str):
    t = df[time_col]
    print("\n# Time column summary")
    print(f"  name: {time_col}")
    print(f"  rows: {len(t):,}")
    try:
        tmin, tmax = t.min(), t.max()
        print(f"  min:  {tmin}")
        print(f"  max:  {tmax}")
        if len(t) > 1:
            diffs = t.diff().dropna()
            dt_mean  = diffs.mean()
            dt_med   = diffs.median()
            fs = (1.0 / dt_mean) if dt_mean and dt_mean != 0 else float('nan')
            print(f"  mean Δt: {dt_mean} s   (median Δt: {dt_med} s)")
            if fs == fs:  # not NaN
                print(f"  estimated sampling rate: ~{fs:.2f} Hz")
    except Exception:
        pass

def attach_class_names(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    df = df.copy()
    def to_int_safe(v):
        try:    return int(v)
        except: return v
    df[label_col] = df[label_col].apply(to_int_safe)
    df["class_name"] = df[label_col].map(CLASS_MAP).fillna("UNKNOWN")
    return df

def print_label_by_time(df: pd.DataFrame, time_col: str, label_col: str, head=15):
    tmp = df[[time_col, label_col]].copy()
    tmp["class_name"] = tmp[label_col].map(CLASS_MAP).fillna("UNKNOWN")
    print(f"\n# First {head} rows: time, label, class_name")
    print(tmp.head(head).to_string(index=False))

def analyze_label_periods(df: pd.DataFrame, time_col: str, label_col: str):
    """
    Analyze time periods for each label in the training dataset.
    Creates a table showing start time, end time, duration, label, and class name for each period.
    """
    print(f"\n# Label Time Period Analysis")
    print("="*80)
    
    # Sort by time to ensure proper ordering
    df_sorted = df.sort_values(time_col).copy()
    
    # Find where labels change
    label_changes = df_sorted[label_col] != df_sorted[label_col].shift(1)
    
    # Get indices where label changes occur (including first row)
    change_indices = df_sorted[label_changes].index.tolist()
    
    periods = []
    
    for i, start_idx in enumerate(change_indices):
        # Determine end index for this period
        if i < len(change_indices) - 1:
            end_idx = change_indices[i + 1] - 1
        else:
            end_idx = df_sorted.index[-1]
        
        # Get period information
        start_time = df_sorted.loc[start_idx, time_col]
        end_time = df_sorted.loc[end_idx, time_col]
        duration = end_time - start_time
        label = df_sorted.loc[start_idx, label_col]
        class_name = df_sorted.loc[start_idx, "class_name"]
        num_samples = end_idx - start_idx + 1
        
        periods.append({
            'Period': i + 1,
            'Start Time': start_time,
            'End Time': end_time,
            'Duration (s)': duration,
            'Label': label,
            'Class Name': class_name,
            'Samples': num_samples
        })
    
    # Create DataFrame for better formatting
    periods_df = pd.DataFrame(periods)
    
    # Format the table for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(periods_df.to_string(index=False))
    
    # Summary statistics
    print(f"\n# Summary Statistics")
    print(f"Total periods: {len(periods)}")
    print(f"Total time span: {periods_df['Duration (s)'].sum():.2f} seconds")
    print(f"Average period duration: {periods_df['Duration (s)'].mean():.2f} seconds")
    print(f"Shortest period: {periods_df['Duration (s)'].min():.2f} seconds")
    print(f"Longest period: {periods_df['Duration (s)'].max():.2f} seconds")
    
    # Label distribution summary
    print(f"\n# Label Distribution in Periods")
    label_summary = periods_df.groupby(['Label', 'Class Name']).agg({
        'Duration (s)': ['count', 'sum', 'mean'],
        'Samples': 'sum'
    }).round(2)
    label_summary.columns = ['Occurrences', 'Total Duration (s)', 'Avg Duration (s)', 'Total Samples']
    print(label_summary.to_string())
    
    # Save periods table
    periods_output = Path("label_time_periods.csv")
    periods_df.to_csv(periods_output, index=False)
    print(f"\nSaved detailed periods table: {periods_output}")
    
    return periods_df

def main():
    # ===== TRAINING (labeled) =====
    print("=== TRAINING DATASET ===")
    train_df = load_csv(TRAIN_PATH)
    print_headers(train_df, TRAIN_PATH.name)

    time_col  = find_col(train_df.columns, TIME_CANDIDATES)
    label_col = find_col(train_df.columns, LABEL_CANDIDATES)

    if time_col is None:
        print("\nWARNING: Could not find time column automatically. "
              f"Tried {TIME_CANDIDATES}. Using first column as time for display.")
        time_col = train_df.columns[0]

    if label_col is None:
        print("\nERROR: Could not find a label column in the training dataset. "
              f"Tried {LABEL_CANDIDATES}. Please rename your label column or update LABEL_CANDIDATES.")
        sys.exit(2)

    describe_time(train_df, time_col)
    train_df = attach_class_names(train_df, label_col)
    print_label_by_time(train_df, time_col, label_col, head=15)
    
    # NEW: Analyze label time periods
    periods_df = analyze_label_periods(train_df, time_col, label_col)

    out_train = Path("training_time_label_class.csv")
    train_df.to_csv(out_train, columns=[time_col, label_col, "class_name"], index=False)
    print(f"Saved: {out_train}")

    # ===== EVENT 1 (unlabeled stream) =====
    print("\n\n=== EVENT 1 (unlabeled stream) ===")
    e1 = load_csv(EVENT1_PATH)
    print_headers(e1, EVENT1_PATH.name)
    e1_time = find_col(e1.columns, TIME_CANDIDATES) or e1.columns[0]
    describe_time(e1, e1_time)

    # Save a light-weight time index for reference (no labels)
    e1_out = Path("event1_time.csv")
    e1[[e1_time]].to_csv(e1_out, index=False)
    print(f"Saved: {e1_out}")

    # ===== EVENT 2 (unlabeled stream) =====
    print("\n\n=== EVENT 2 (unlabeled stream) ===")
    e2 = load_csv(EVENT2_PATH)
    print_headers(e2, EVENT2_PATH.name)
    e2_time = find_col(e2.columns, TIME_CANDIDATES) or e2.columns[0]
    describe_time(e2, e2_time)

    e2_out = Path("event2_time.csv")
    e2[[e2_time]].to_csv(e2_out, index=False)
    print(f"Saved: {e2_out}")

    print("\nDone.")

if __name__ == "__main__":
    main()