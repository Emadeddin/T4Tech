
- **Sampling rate**: 4,800 Hz (80 samples/cycle @ 60 Hz)
- **Single-zone (PZ-2)**: 14 features, 18 classes, 105,768 samples (22.04 s)
- **Multi-zone (PZ-1..PZ-5)**: 70 features, labels (Zone, Class, Device), 528,840 samples (110.18 s)

See `docs/dataset_spec.md` and `docs/events.md` for full details.
""")
os.makedirs(os.path.join(root, "data"), exist_ok=True)
with open(os.path.join(root, "data", "README.md"), "w") as f:
    f.write(data_readme)

# Docs
docs_dir = os.path.join(root, "docs")
os.makedirs(docs_dir, exist_ok=True)

events_md = textwrap.dedent("""
# Streaming Events

All times are in seconds; sampling at 4,800 Hz (80 samples/cycle).

## Event Stream 1 (5 anomalies)
1.00–1.20: SLG A–N (Class 1)  
2.00–2.20: LL B–C (Class 7)  
3.00–3.20: DLG AC–N (Class 10)  
4.00–4.20: CT Ratio Attack MU32 (Class 4)  
5.00–5.20: PT Ratio Attack MU23 (Class 13)

## Event Stream 2 (4 anomalies + one out-of-zone)
1.00–1.20: 3φ Fault ABC (Class 14)  
2.00–2.20: 3φ+G Fault ABC–G (Class 15)  
4.00–4.20: GPS Spoofing MU32 (Class 16)  
5.00–5.20: SLG fault in another zone (unlabeled in single-zone set; treat as unseen/−1)
""")
with open(os.path.join(docs_dir, "events.md"), "w") as f:
    f.write(events_md)

dataset_spec = textwrap.dedent("""
# Dataset Specification

**Origin**: High-fidelity WinIGS EMT simulations (PSCAL, Georgia Tech).  
**Formats**: CSV/COMTRADE-derived time-domain samples.  
**Sampling**: 4,800 Hz (Δt = 208 µs).

## Single-Zone Training (PZ-2)
- Samples: 105,768 (22.04 s)
- Features: 14 (3I+N, 3V per MU × 2 MUs: MU23, MU32)
- Classes: 18 (see `data/schemas/class_map.csv`)
- Label column: `label` (0..17)

## Multi-Zone Training (PZ-1..PZ-5)
- Samples: 528,840 (110.18 s)
- Features: 70 (3I+N, 3V × 10 MUs)
- Labels: `Zone` (1..5), `Class` (0..17), `Device` (codes per MU id)

## Preprocessing
- Remove timestamp to avoid leakage
- z-score per-channel using train-set μ, σ
- Optional: 80-sample windows for temporal models
- Class balancing as needed for Normal vs minority anomalies
""")
with open(os.path.join(docs_dir, "dataset_spec.md"), "w") as f:
    f.write(dataset_spec)

# README
readme = textwrap.dedent("""
<p align="center">
  <img src="assets/header.png" alt="T4Tech" width="85%"/>
</p>

# T4Tech: Machine & Deep Learning for High-Fidelity Grid Anomaly Classification

**T4Tech** provides open, reproducible datasets and code specs for training and streaming evaluation of ML/DL
models on inverter-rich power systems. It accompanies two complementary papers: an accuracy/robustness study
on high-fidelity streaming classification and a follow-up benchmark focusing on deployability.

## Quick Start
```bash
git clone <your-repo-url> t4tech
cd t4tech
# Data layout and specs
cat data/README.md
cat docs/dataset_spec.md
