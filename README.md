# ETF NAV & Fixed Income Risk Analytics Engine

A Python-based analytics tool that computes synthetic NAV for a fixed income ETF basket, tracks premium/discount to market price, and runs DV01/duration risk across constituent bonds — replicating core workflows of an ETF trading desk.

## Features

- **Synthetic NAV Computation** — DCF-based bond pricing across a weighted constituent basket (Treasuries + IG credit + Agency MBS)
- **Premium / Discount Tracking** — Compares synthetic NAV to live ETF market price (AGG); signals creation/redemption opportunities for Authorised Participants
- **DV01 & Modified Duration Risk** — Per-constituent and portfolio-level interest rate sensitivity
- **Historical NAV Series** — Rolling NAV computation over a configurable lookback window
- **Desk-ready Dashboard** — 4-panel dark-mode visualisation: NAV vs market price, premium/discount bars, DV01 heatmap, duration trend

## Architecture

```
etf_nav_engine/
├── nav_engine.py      # Core: bond pricing, NAV computation, yield fetch, P/D analysis
├── visualiser.py      # Dashboard: 4-panel matplotlib output
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Run the analytics engine (CLI output)
python nav_engine.py

# Generate the dashboard PNG
python visualiser.py
```

## Methodology

### Bond Pricing
Each constituent is priced using standard DCF:

$$P = \sum_{t=1}^{n} \frac{C}{(1 + y/f)^t} + \frac{F}{(1 + y/f)^n}$$

where `C` = periodic coupon, `y` = YTM, `f` = coupon frequency, `F` = face value.

### Modified Duration & DV01
```
Modified Duration = Macaulay Duration / (1 + ytm/freq)
DV01 = Modified Duration × Price × 0.0001
```

### Synthetic NAV
```
NAV per share = weighted-average clean price, expressed per $100 of face value
```

### Premium / Discount
```
P/D (%) = (Market Price - NAV) / NAV × 100
Positive → Creation opportunity (AP buys basket, delivers to ETF, receives shares to sell)
Negative → Redemption opportunity (AP buys ETF shares, redeems for basket)
```

## Yield Data
Treasury yields fetched via `yfinance` (^FVX, ^TNX, ^TYX). Credit spreads for IG and MBS constituents are applied on top of the relevant Treasury benchmark. In a production environment these would be sourced from Bloomberg or a fixed income data vendor.

## Tech Stack
Python · NumPy · Pandas · yfinance · Matplotlib

## Author
Nikhil Bhalla — [LinkedIn](https://linkedin.com/in/nikhil-bhalla-383186208)
