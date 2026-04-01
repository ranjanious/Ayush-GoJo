# Stochastic Modeling of Interest Rates and Its Impact on Bond Portfolio Risk

Ayush Ranjan and Gaurang Jinka  |  April to Fall 2026

## Overview

This repository contains all simulation code, calibration scripts,
and figure-generation code for the research paper submitted to arXiv
in the q-fin.RM category.

## Requirements

- Python 3.11+ (developed on 3.14.3)
- A free FRED API key from fred.stlouisfed.org/docs/api/api_key.html

## Setup

```bash
python3 -m venv venv_rates
source venv_rates/bin/activate   # Windows: venv_rates\Scripts\activate.bat
pip install -r requirements.txt
export FRED_API_KEY="your_key_here"
```

## Reproducing the data

```bash
python3 data_prep.py
```

Downloads FRED yield series and saves data/processed/fred_yields.parquet.

## Repository structure

```
data/processed/   -- cleaned yield data (committed)
data/raw/         -- source files, not committed (regenerate via data_prep.py)
models/shared/    -- pricing and risk metric functions used by all models
models/dbm/       -- Model A: Drifted Brownian Motion
models/vasicek/   -- Model B: Vasicek
models/cir/       -- Model C: Cox-Ingersoll-Ross
models/hw/        -- Model D: Hull-White
figures/          -- publication-quality PDF figures
paper/            -- LaTeX source for arXiv
```

## Bond Portfolio

| Parameter | Bond A | Bond B |
|-----------|--------|--------|
| Maturity | 2 years | 10 years |
| Coupon rate | 3.68% | 4.20% |
| Semiannual coupon | $1.84 | $2.10 |
| Initial price | $100 (par) | $100 (par) |
| Portfolio weight | 50% | 50% |

## Models

1. **DBM** (Drifted Brownian Motion) -- random walk baseline
2. **Vasicek** -- mean-reverting, Gaussian, closed-form bond prices
3. **CIR** (Cox-Ingersoll-Ross) -- mean-reverting, non-negative rates
4. **Hull-White** -- yield-curve fitting via time-varying drift

## Source Data

U.S. Treasury Daily Par Yield Curve CMT Rates, March 17, 2026.
FRED series: DGS3MO, DGS2, DGS10, T10Y2Y.
