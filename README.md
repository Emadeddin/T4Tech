

 <p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Emadeddin/T4Tech/blob/main/CS7641_ML_Project_Logo.png?raw=true">
  <img src="https://github.com/Emadeddin/T4Tech/blob/main/CS7641_ML_Project_Logo.png?raw=true" alt="Description of image" style="width: 300; height: auto;">
</picture>
</p>

<h1 align="center">T4Tech: Machine & Deep Learning for High-Fidelity Grid Anomaly Classification</h1>

T4Tech provides open, reproducible datasets for training and evaluating ML/DL models on inverter-rich power systems using **time-domain signals** sampled at **4.8 kHz**. It includes a **Zone-2 training dataset** and **two streaming event datasets** designed for benchmarking real-time classification pipelines.

---

# T4Tech Power System Cybersecurity Dataset

## IEEE Citation Format

**Paper Reference:**

E. Abukhousa, S. S. F. Syed Afroz, F. Alsaeed, A. Qwbaiban, S. Zonouz, and A. P. S. Meliopoulos, "The Wisdom of the Crowd: High-Fidelity Classification of Cyber-Attacks and Faults in Power Systems Using Ensemble and Machine Learning," to be published in *Proc. IEEE PES Innovative Smart Grid Technologies (ISGT) Middle East*, Dubai, UAE, Nov. 23-26, 2025.

## Dataset Information

This high-fidelity power system cybersecurity dataset was developed as part of the research presented in the above paper. The dataset provides comprehensive measurements from a realistic digital substation simulation, designed to support machine learning research in power system fault detection and cybersecurity.

### Dataset Generation Details

- **Software Platform**: WinIGS (Integrated Grounding System Analysis)
- **Institution**: Power System Control and Automation Laboratory (PSCAL), Georgia Institute of Technology
- **Simulation Environment**: High-fidelity electromagnetic transient (EMT) simulations
- **Sampling Rate**: 4.8 kHz (80 samples per cycle at 60 Hz)
- **Time Resolution**: 208 microseconds
- **Data Format**: COMTRADE (Common Format for Transient Data Exchange)

### WinIGS Software Information

WinIGS is a proprietary power system analysis and grounding simulation software developed by Advanced Grounding Concepts (AGC). The software enables detailed electromagnetic transient simulations that capture the dynamic interactions between synchronous generators and inverter-based resources (IBRs) under realistic operating conditions.

**Software Details:**
- **Name**: WinIGS Integrated Grounding System Analysis for Windows
- **Version**: 8.1.5
- **Developer**: Advanced Grounding Concepts (AGC), Alpharetta, GA, USA
- **Website**: https://ap-concepts.com/

### Laboratory Information

**Power System Control and Automation Laboratory (PSCAL)**
- **Institution**: School of Electrical and Computer Engineering, Georgia Institute of Technology
- **Location**: Atlanta, GA, USA
- **Website**: https://pscal.ece.gatech.edu/
- **Research Focus**: Power system protection, control, cybersecurity, and grid modernization

## Open Source Research Initiative

This dataset is made publicly available to support and advance research in power system cybersecurity and machine learning applications. We encourage the research community to utilize this high-fidelity dataset for:

- Power system fault classification algorithms
- Cybersecurity anomaly detection methods
- Machine learning model development and benchmarking
- Real-time protection system research
- Cyber-physical security analysis

### Citation Request

If you use this dataset in your research, please cite our work using the IEEE format provided above. Your citations help support continued research and dataset development in this critical area.

### Dataset Contents

The dataset includes three main components:

1. **Training Dataset** (`zone2_training_dataset.csv`)
   - Duration: 22 seconds (21,999,760 microseconds)
   - Samples: 105,768 observations
   - Classes: 18 (1 normal + 17 fault/attack scenarios)
   - Features: 14 electrical measurements + 1 label

2. **Event 1** (`event1.csv`)
   - Duration: 6 seconds
   - Samples: 28,800 observations
   - Features: 14 electrical measurements (unlabeled)

3. **Event 2** (`event2.csv`)
   - Duration: 6 seconds
   - Samples: 28,800 observations
   - Features: 14 electrical measurements (unlabeled)

### Measurement Channels

The dataset includes voltage and current measurements from two Merging Units (MUs):

**MU23 Measurements:**
- Three-phase currents: MU23_I23_a, MU23_I23_b, MU23_I23_c, MU23_I23_n
- Three-phase voltages: MU23_V2_an, MU23_V2_bn, MU23_V2_cn

**MU32 Measurements:**
- Three-phase currents: MU32_I32_a, MU32_I32_b, MU32_I32_c, MU32_I32_n
- Three-phase voltages: MU32_V32_an, MU32_V32_bn, MU32_V32_cn

### Fault and Attack Scenarios

The training dataset includes 17 distinct anomaly classes:
- Single line faults (A-N, B-N, C-N)
- Double line faults (A-B, A-C, B-C)
- Double line-to-ground faults (AB-N, AC-N, BC-N)
- Three-phase faults (AB-C, ABC-N)
- CT ratio attacks on MU32 and MU23
- PT ratio attacks on MU32 and MU23
- GPS spoofing attacks on MU32 and MU23

## Publication Venue

**IEEE PES Innovative Smart Grid Technologies (ISGT) Middle East**
- **Date**: November 23-26, 2025
- **Location**: Dubai, UAE
- **Organizers**: IEEE Power & Energy Society (PES) and University of Dubai
- **Conference Focus**: Smart grid technologies, renewable integration, and grid modernization

## Contact Information

For questions about the dataset or research collaboration opportunities, please contact:

**Primary Authors:**
- Emad Abukhousa (emadak@gatech.edu)
- Syed Sohail Feroz Syed Afroz (safroz7@gatech.edu)
- A.P. Sakis Meliopoulos (sakis.m@gatech.edu)

**Institution:**
School of Electrical and Computer Engineering  
Georgia Institute of Technology  
Atlanta, GA, USA

---

*This dataset represents a significant contribution to the power system cybersecurity research community. We encourage researchers worldwide to utilize this resource and contribute to the advancement of secure and resilient power systems through machine learning and artificial intelligence applications.*
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
