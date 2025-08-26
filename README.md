

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

## 🧰 How to Use

```bash
# clone your repo (example)
git clone https://github.com/<you>/T4Tech-HighFidelity-Grid-ML.git
cd T4Tech-HighFidelity-Grid-ML

# explore schemas
cat data/schemas/class_map.csv
cat data/schemas/single_zone_features.csv

# typical workflow (pseudo)
python train.py --data data/single-zone/zone2_training_dataset.csv
python eval_stream.py --data data/single-zone/streaming/event1.csv --tau 0.6 --smooth 80
