# GENCO Challenge — Prediction Error Analysis

This repository contains analysis scripts for the GENCO Challenge, focusing on active-power prediction errors in the IEEE-14 and IEEE-30 cases.

## Repository structure

```text
genco-challenge-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── analyze_case30_buses.py
│   ├── compare_case14_case30.py
│   └── analyze_active_power_compensation.py
├── figures/                 # Curated figures for the final submission
├── docs/                    # Methodology and notes
└── outputs/                 # Local datasets and generated intermediate results (not tracked)
    ├── case14_seed42/
    └── case30_seed42/
```

## Data placement

Place the prediction files locally at these paths:

- `outputs/case14_seed42/case14_ieee_predictions.parquet`
- `outputs/case30_seed42/case30_ieee_predictions.parquet`

The parquet files and generated intermediate outputs are excluded from Git by default. Only publish datasets if the challenge rules and data licence explicitly allow redistribution. The final, selected figures can be copied to `figures/` and committed.

## Setup

Use Python 3.10 or newer if compatible with your environment.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the analyses

Run commands from the repository root so the relative data paths resolve correctly.

```bash
python src/compare_case14_case30.py
python src/analyze_active_power_compensation.py
python src/analyze_case30_buses.py
```

The scripts write intermediate tables and other analysis outputs beneath `outputs/`. The detailed IEEE-30 script is exploratory and contains a broader set of analyses; the other two scripts focus on comparing active-power errors and their compensation across IEEE-14 and IEEE-30.

## Important interpretation note

The sum of signed generator active-power prediction errors is an aggregate prediction-error measure. It should not automatically be described as the physical power-balance error of the network. Bus numbering and the meaning of target/prediction columns must be checked against the challenge data definition before drawing physical conclusions.

## Current focus

The primary analysis investigates whether prediction errors in active generation at buses 0 and 1 tend to have opposite signs and partially cancel when summed. Apparent power and the challenge's MRES metric may be added only after their definitions and data availability have been verified.
