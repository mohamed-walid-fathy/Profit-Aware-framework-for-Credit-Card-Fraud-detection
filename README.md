# Profit-Aware AI Framework for Credit Card Fraud Detection

> **Thesis Project — New Giza University (NGU), 2026**  
> A hybrid Machine Learning + Reinforcement Learning system that makes financially-optimal fraud decisions by dynamically adapting to transaction amount, reviewer capacity, and business cost structure.

---

## 📄 Thesis

The full thesis PDF is included in this repository: [`thesis.pdf`](./thesis.pdf)

---

## 🧠 What This Project Does

Most fraud detection systems are optimised for statistical metrics like AUC or F1. This framework is different — it is optimised for **profit**.

Rather than predicting fraud as a binary classification problem, the system learns a three-way decision policy:

```
fi < T_low(A)              →  Accept   (let the transaction through)
T_low(A) ≤ fi < T_high(A)  →  Review   (send to a human analyst)
fi ≥ T_high(A)              →  Reject   (block the transaction)
```

Where `fi` is the fraud probability score and `T_low`, `T_high` are **amount-aware thresholds** that shift dynamically with transaction size. A suspicious $500 transaction and a suspicious $5 transaction are not treated the same — the system knows the review is worth more for one than the other.

Performance is measured using **Profit Gain (PG)**:

$$PG = \frac{\$_{model} - \$_{no\ fraud\ management}}{\$_{oracle} - \$_{no\ fraud\ management}}$$

A PG of 0 means the system adds nothing over doing nothing. A PG of 1 means it captures the entire theoretically achievable gain from fraud detection.

---

## 🏗️ System Architecture

```
Transaction Data
      │
      ▼
Preprocessing & Feature Engineering
(UID features, time features, frequency encodings, AE anomaly features)
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
CatBoost Classifier                   Autoencoder (AE)
(raw categoricals + numeric features)  (trained on legitimate txns only)
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
          Hybrid Score Blending
          α · CatBoost + (1−α) · AE_pct
          (α selected by validation profit, not AUC)
                     │
                     ▼
          Score Normalisation [0, 1]
                     │
                     ▼
          ┌─────────────────────┐
          │   PPO RL Agent      │  ←── Cost Function (8-outcome, amount-aware)
          │  Learns: base_low,  │
          │  slope_low,         │
          │  base_high,         │
          │  slope_high         │
          └─────────────────────┘
                     │
                     ▼
     Amount-Aware Decision Engine
     T_low(A) = base_low + slope_low × z(A)
     T_high(A) = base_high + slope_high × z(A)
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      Accept       Review       Reject
                     │
                     ▼
          Capacity Constraint
          (daily reviewer budget)
                     │
                     ▼
          Human Analyst Review
                     │
                     ▼
          Monitoring Dashboard
          (decisions · risk scores · financial metrics · drift detection)
```

---

## 💡 Key Design Decisions

### Hybrid Classifier
CatBoost is used as the primary supervised classifier because it handles raw categorical features natively (card type, email domain, device type, etc.) without manual frequency encoding — which was shown to introduce distributional drift. Its output is blended with an Autoencoder anomaly score:

```
hybrid_score = α · CatBoost_sqrt_minmax + (1 − α) · AE_percentile
```

The blending weight `α` is selected by maximising **validation profit**, not AUC — a deliberate design choice that aligns model selection with the actual evaluation objective.

### Amount-Aware Thresholds
The single most important architectural change from the initial design. The RL agent learns four parameters instead of two fixed scalars:

| Parameter | Meaning |
|---|---|
| `base_low` | Accept/Review boundary at the median transaction amount |
| `base_high` | Review/Reject boundary at the median transaction amount |
| `slope_low` | How aggressively the accept boundary lowers for large transactions |
| `slope_high` | How aggressively the reject boundary lowers for large transactions |

A negative `slope_low` means the agent becomes more willing to review expensive transactions at lower suspicion scores — financially correct because the review cost ($3.50) is trivial relative to the fraud loss on a large transaction.

### Extended Cost Function
The cost function models all eight combinations of decision × ground truth, including a **reviewer accuracy parameter** (`prev`) that governs the probability-weighted expected loss when a transaction enters review:

| Decision | Ground Truth | Cost |
|---|---|---|
| Accept | Fraud | `flm × A` |
| Accept | Legitimate | `−pr × A` |
| Reject | Fraud | `0` |
| Reject | Legitimate | `pr × A × ltv` |
| Review | Fraud | `rc + (1−prev) × flm × A` |
| Review | Legitimate | `rc + pr × A × ((1−prev) × ltv − prev)` |

The review cost decomposes as: a flat fee `rc`, plus a probability-weighted expected loss from incorrect decisions, minus a revenue credit for correctly-cleared legitimate transactions. This prevents the agent from treating review as a cost-free escape hatch.

### Capacity-Constrained Sequential RL
A separate RL environment models a **hard daily reviewer budget**. Overflow transactions are resolved via a profit-optimal binary fallback rule rather than unlimited escalation. This was the key finding for deployment realism — the optimal static threshold under unlimited review degrades significantly when capacity is enforced, but the amount-aware policy is substantially more robust to budget constraints.

---

## 📊 Results

| Method | Profit Gain (PG) | Notes |
|---|---|---|
| Baseline (th=0.5) | ~0.00 | No optimisation |
| Grid Search | ~0.75 | Exhaustive threshold sweep |
| PPO Fixed Threshold | ~0.79 | Single (T_low, T_high) pair |
| PPO Amount-Continuous | **~0.82** | Four-parameter amount-aware policy |

> Results on the IEEE-CIS Fraud Detection dataset (Kaggle). Full results including per-expert breakdown, capacity sweep, and forward-chaining validation are in [`thesis.pdf`](./thesis.pdf).

---

## 📁 Repository Structure

```
├── thesis.pdf                          # Full thesis document
│
├── notebooks/
│   ├── GPPPO_expert_cluster.ipynb      # Expert cluster model (per-band LGB + unified PPO)
│   ├── GPPPO_A_your_code.ipynb         # LGB + AE + catch-bonus PPO
│   ├── GPPPO_B_friend_code.ipynb       # Hybrid blend variant
│   ├── GPPPO_v3_catch_bonus.ipynb      # Catch-bonus reward formulation (β=0.38)
│   └── GPPPO_hybrid_fixed_all20.ipynb  # Final fixed version (all 20 issues resolved)
│
├── dashboard/
│   └── fraud_explorer.py               # Streamlit amount distribution explorer
│
├── preproc_cache/                      # Auto-generated after first run (gitignored)
│   ├── train_preprocessed.csv
│   ├── val_preprocessed.csv
│   ├── test_preprocessed.csv
│   └── preproc_meta.pkl
│
├── expert_cache/                       # Auto-generated expert model artifacts
├── ae_cache/                           # Auto-generated AE features
├── lstm_cache/                         # Auto-generated LSTM vectors
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies:

```
pandas numpy matplotlib seaborn scipy scikit-learn
tensorflow lightgbm catboost torch
streamlit tqdm joblib
```

### Data

This project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/competitions/ieee-fraud-detection/data) from Kaggle.

Download and place the files as follows:

```
ieee-fraud-detection/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### Running the Main Notebook

Open `notebooks/GPPPO_hybrid_fixed_all20.ipynb` and run cells in order. On the first run, preprocessing will execute in full and save CSVs to `preproc_cache/`. All subsequent runs load directly from cache — typically saving 10–15 minutes per session.

### Running the Amount Distribution Explorer

```bash
cd dashboard
streamlit run fraud_explorer.py
```

Make sure to launch from the directory containing your `preproc_cache/` or `ieee-fraud-detection/` folder.

---

## ⚙️ Configuration

The key parameters to adjust are in the early cells of the main notebook:

```python
# Cost function parameters
COST_PARAMS = dict(
    pr   = 0.02,   # profit rate per transaction
    ltv  = 5,      # lifetime value multiplier
    flm  = 3,      # fraud loss multiplier
    rc   = 3.5,    # review cost ($)
    prev = 0.90,   # reviewer accuracy
)

# Expert cluster splits (from density explorer)
EXPERT_SPLITS = [50.0, 200.0]   # 2 splits → 3 expert models

# PPO training
BETA        = 0.38   # catch-bonus reward coefficient
N_EPISODES  = 40000
SEED        = 42
```

---

## 📐 Mathematical Summary

**Hybrid fraud score:**
$$s_i = \alpha \cdot \text{CatBoost}_{mm}(x_i) + (1 - \alpha) \cdot \text{AE}_{pct}(x_i)$$

**Amount-aware thresholds:**
$$T_{low}(A_i) = base_{low} + slope_{low} \cdot z(A_i)$$
$$T_{high}(A_i) = base_{high} + slope_{high} \cdot z(A_i)$$

**Normalised log-amount:**
$$z(A_i) = \frac{\log(1 + A_i) - \mu_{\log A}}{\sigma_{\log A}}$$

**Catch-bonus reward:**
$$R = -\text{Cost} + \beta \sum_{i \in \text{TP-Reject}} A_i$$

**Profit Gain:**
$$PG = \frac{\$_{model} - \$_{no\ fraud}}{\$_{oracle} - \$_{no\ fraud}}$$

---

## 👥 Team

| Name | Role |
|---|---|
| [Mohamed Walid] | RL design, cost function formulation, PPO training, amount-aware policy |
| [Mohamed Osama] | Data processing, Autoencoder training, Capacity Layer implementation|
| [Ammar Ahmed] | KDE training |
| [Aly Osman] | Random Forest Training | 
| [Jana Sameh] | CNN Training, Dashboard Design and implementation | 
| [Abdulrahman Amr] LOF Training, Dashboard Design |


**Supervisor:** [Dr.Mariam Nabil] — Assistant Professor, New Giza University

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@thesis{yourname2025fraud,
  title     = {Profit-Aware AI Framework for Credit Card Fraud Detection},
  author    = {Mohammad Ossama, Ammar Ahmed, Aly Osman, Abdulrahman Amr, Jana Sameh, Mohamed Walid},
  year      = {2026},
  school    = {New Giza University},
  type      = {Bachelor's Thesis}
}
```

---

## 📄 License

This project is released under the MIT License. See [`LICENSE`](./LICENSE) for details.

---

<p align="center">
  <i>Built with equal parts gradient descent and stubbornness.</i>
</p>
