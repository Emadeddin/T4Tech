

 <p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Emadeddin/T4Tech/blob/main/CS7641_ML_Project_Logo.png?raw=true">
  <img src="https://github.com/Emadeddin/T4Tech/blob/main/CS7641_ML_Project_Logo.png?raw=true" alt="Description of image" style="width: 300; height: auto;">
</picture>
</p>

<h1 align="center">T4Tech: Machine & Deep Learning for High-Fidelity Grid Anomaly Classification</h1>

T4Tech provides open, reproducible datasets for training and evaluating ML/DL models on inverter-rich power systems using **time-domain signals** sampled at **4.8 kHz**. It includes a **Zone-2 training dataset** and **two streaming event datasets** designed for benchmarking real-time classification pipelines.

---

## 📦 Datasets

### 🧪 Training (Zone 2)
- **File:** `data/single-zone/zone2_training_dataset.csv`
- **Sampling rate:** 4,800 Hz (80 samples/cycle @ 60 Hz)
- **Duration:** ~22.0 s (≈105,768 samples)
- **Features (14):** 3ϕ currents + neutral and 3ϕ voltages from **MU23** and **MU32**  
  *(Exact list in `data/schemas/single_zone_features.csv`.)*
- **Classes (18):** 0..17  
  *(Map in `data/schemas/class_map.csv`)*

### ⚡ Streaming Events (Zone 2)
- **Event 1:** `data/single-zone/streaming/event1.csv`  
  1.00–1.20s: SLG A–N (1) • 2.00–2.20s: LL B–C (7) • 3.00–3.20s: DLG AC–N (10) • 4.00–4.20s: CT Ratio MU32 (4) • 5.00–5.20s: PT Ratio MU23 (13)
- **Event 2:** `data/single-zone/streaming/event2.csv`  
  1.00–1.20s: ABC (14) • 2.00–2.20s: ABC–G (15) • 4.00–4.20s: GPS Spoof MU32 (16) • 5.00–5.20s: SLG in another zone → **treat as unseen (−1)**

---
# T4Tech Power System Dataset Analysis

This repository contains an analysis of the T4Tech power system cybersecurity dataset, which includes electrical grid measurements and various fault/attack scenarios for Zone 2 operations.

## 📊 Dataset Overview

The dataset consists of three main components:
- **Training Dataset**: Labeled data with 18 different operational states
- **Event 1**: Unlabeled streaming data for testing
- **Event 2**: Unlabeled streaming data for testing

## 📁 Dataset Structure

### Training Dataset (`zone2_training_dataset.csv`)
- **Total Samples**: 105,768 observations
- **Time Span**: 21,999,760 microseconds (~22 seconds)
- **Sampling Rate**: ~4,808 Hz (208-microsecond intervals)
- **Features**: 15 sensor measurements + 1 label column

#### Sensor Measurements
- **MU23 Measurements**: Current (I23_a/b/c/n) and Voltage (V2_an/bn/cn)
- **MU32 Measurements**: Current (I32_a/b/c/n) and Voltage (V32_an/bn/cn)

### Event Datasets (`event1.csv`, `event2.csv`)
- **Total Samples**: 28,800 observations each
- **Time Span**: ~6 seconds each
- **Sampling Rate**: ~4,800 Hz (208-microsecond intervals, similar to training)
- **Features**: 14 sensor measurements (no labels)

## 🏷️ Label Classification System

The training dataset includes 18 different operational states:

| Label | Class Name | Description |
|-------|------------|-------------|
| 0 | Normal Operation | Standard grid operation |
| 1 | Single Line Fault A–N | Phase A to neutral fault |
| 2 | Single Line Fault B–N | Phase B to neutral fault |
| 3 | Single Line Fault C–N | Phase C to neutral fault |
| 4 | CT Ratio Attack on MU32 | Current transformer attack |
| 5 | Double Line Fault A–B | Phase A to B fault |
| 6 | Double Line Fault A–C | Phase A to C fault |
| 7 | Double Line Fault B–C | Phase B to C fault |
| 8 | CT Ratio Attack on MU23 | Current transformer attack |
| 9 | DLG Fault AB–N | Double line to ground fault AB-N |
| 10 | DLG Fault AC–N | Double line to ground fault AC-N |
| 11 | DLG Fault BC–N | Double line to ground fault BC-N |
| 12 | PT Ratio Attack on MU32 | Potential transformer attack |
| 13 | PT Ratio Attack on MU23 | Potential transformer attack |
| 14 | 3 Lines Fault AB–C | Three-phase fault AB-C |
| 15 | 3 Lines Fault ABC–N | Three-phase fault ABC-N |
| 16 | GPS Spoofing on MU32 | GPS timing attack |
| 17 | GPS Spoofing on MU23 | GPS timing attack |

## 📈 Temporal Analysis Results

### Label Period Distribution

The training dataset contains **35 distinct time periods** alternating between normal operations and various fault/attack scenarios:

| Period | Start Time (μs) | End Time (μs) | Duration (μs) | Duration (s) | Label | Class Name |
|--------|-----------------|---------------|---------------|--------------|-------|------------|
| 1 | 0 | 999,856 | 999,856 | 1.00 | 0 | Normal Operation |
| 2 | 1,000,064 | 1,499,888 | 499,824 | 0.50 | 1 | Single Line Fault A–N |
| 3 | 1,500,096 | 1,999,920 | 499,824 | 0.50 | 0 | Normal Operation |
| 4 | 2,000,128 | 2,499,952 | 499,824 | 0.50 | 2 | Single Line Fault B–N |
| 5 | 2,500,160 | 2,999,984 | 499,824 | 0.50 | 0 | Normal Operation |
| ... | ... | ... | ... | ... | ... | ... |
| 35 | 21,500,144 | 21,999,760 | 499,616 | 0.50 | 0 | Normal Operation |

> 📋 *Complete period table available in `label_time_periods.csv`*

### Key Statistics

- **Total Periods**: 35
- **Total Time Span**: 21,992,688 microseconds (~22 seconds)
- **Average Period Duration**: 628,363 microseconds (~0.63 seconds)
- **Shortest Period**: 499,616 microseconds (~0.50 seconds)
- **Longest Period**: 1,499,776 microseconds (~1.50 seconds)

### Class Distribution

| Label | Class Name | Occurrences | Total Duration (μs) | Total Duration (s) | Avg Duration (s) | Total Samples |
|-------|------------|-------------|---------------------|-------------------|------------------|---------------|
| 0 | Normal Operation | 18 | 13,496,304 | 13.50 | 0.75 | 64,903 |
| 1-17 | Various Faults/Attacks | 1 each | ~499,800 each | ~0.50 each | ~0.50 each | ~2,404 each |

## 🔍 Data Characteristics

### Training Data Pattern
- **Balanced Design**: Each fault/attack type appears exactly once
- **Consistent Duration**: Most fault periods last ~0.5 seconds
- **Normal Operation Baseline**: Accounts for ~61% of total samples
- **Sequential Structure**: Alternating pattern between normal and fault states
- **High-Frequency Sampling**: ~4,808 Hz providing detailed temporal resolution
- **Rapid Transitions**: Each fault/attack scenario captured in sub-second intervals

### Event Data Pattern
- **Similar Sampling Rate**: ~4,800 Hz, consistent with training data
- **Extended Duration**: 6-second windows (much longer than training periods)
- **Unlabeled**: Designed for anomaly detection and classification testing
- **Real-time Simulation**: Represents continuous monitoring scenarios

### Time Scale Insights
- **Microsecond Precision**: Time stamps in microseconds enable precise fault detection
- **Sub-second Events**: Most fault conditions last only 0.5 seconds
- **Rapid Sampling**: ~208 microsecond intervals between measurements
- **Event Detection Window**: Training periods are much shorter than typical event analysis windows

## 🛠️ Analysis Tools

The repository includes a comprehensive data exploration script (`explore.py`) that provides:

- **Automatic column detection** for time and label columns
- **Statistical summaries** of temporal characteristics
- **Period analysis** with start/end times for each label
- **Class distribution** statistics
- **Export functionality** for further analysis

### Usage
```bash
python explore.py
```

### Generated Files
- `training_time_label_class.csv` - Time series with labels and class names
- `label_time_periods.csv` - Detailed period analysis table
- `event1_time.csv` - Event 1 time index
- `event2_time.csv` - Event 2 time index

## 📊 Dataset Applications

This dataset is suitable for:

- **Power System Cybersecurity Research**
- **Anomaly Detection** in electrical grids
- **Time Series Classification** algorithms
- **Real-time Fault Detection** systems
- **Cyber-Physical Security** analysis
- **Machine Learning** for power system monitoring

## 🚀 Getting Started

1. Clone this repository
2. Ensure you have the required CSV files:
   - `zone2_training_dataset.csv`
   - `event1.csv`
   - `event2.csv`
3. Run the analysis script: `python explore.py`
4. Explore the generated analysis files and results

## 📋 Requirements

- Python 3.x
- pandas
- pathlib (standard library)

---

*This analysis provides insights into the high-frequency temporal structure and characteristics of the T4Tech power system cybersecurity dataset. The microsecond-precision timestamps and sub-second fault durations make this dataset particularly valuable for rapid fault detection and real-time grid security research.*
