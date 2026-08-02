"""
ETF NAV & Fixed Income Risk Analytics Engine
=============================================
Computes synthetic NAV for a fixed income ETF basket, tracks premium/discount
to market price, and runs DV01/duration risk across constituents.

Author: Nikhil Bhalla
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Bond pricing utilities
# ---------------------------------------------------------------------------

def bond_price(face: float, coupon_rate: float, ytm: float,
               years_to_maturity: float, freq: int = 2) -> float:
    """
    Compute the clean price of a fixed-coupon bond using discounted cash flows.

    Parameters
    ----------
    face            : Face/par value of the bond
    coupon_rate     : Annual coupon rate (e.g. 0.04 for 4%)
    ytm             : Yield to maturity (annual, e.g. 0.05 for 5%)
    years_to_maturity: Time to maturity in years
    freq            : Coupon frequency per year (2 = semi-annual)

    Returns
    -------
    float : Clean bond price
    """
    periods = int(years_to_maturity * freq)
    coupon = face * coupon_rate / freq
    ytm_per_period = ytm / freq

    if periods == 0:
        return face

    times = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face  # principal at maturity

    price = np.sum(cash_flows / (1 + ytm_per_period) ** times)
    return price


def modified_duration(face: float, coupon_rate: float, ytm: float,
                      years_to_maturity: float, freq: int = 2) -> float:
    """
    Compute modified duration of a bond (Macaulay duration / (1 + ytm/freq)).
    """
    periods = int(years_to_maturity * freq)
    if periods == 0:
        return 0.0

    coupon = face * coupon_rate / freq
    ytm_per_period = ytm / freq
    times = np.arange(1, periods + 1)
    cash_flows = np.full(periods, coupon)
    cash_flows[-1] += face

    pv_cf = cash_flows / (1 + ytm_per_period) ** times
    price = np.sum(pv_cf)
    macaulay = np.sum(times * pv_cf) / price / freq  # in years
    mod_dur = macaulay / (1 + ytm_per_period)
    return mod_dur


def dv01(face: float, coupon_rate: float, ytm: float,
         years_to_maturity: float, freq: int = 2) -> float:
    """
    DV01 (Dollar Value of a Basis Point): price sensitivity to 1bp YTM shift.
    DV01 = Modified Duration * Price * 0.0001
    """
    price = bond_price(face, coupon_rate, ytm, years_to_maturity, freq)
    mod_dur = modified_duration(face, coupon_rate, ytm, years_to_maturity, freq)
    return mod_dur * price * 0.0001


# ---------------------------------------------------------------------------
# ETF constituent data
# ---------------------------------------------------------------------------

# Simulated AGG-like (iShares Core US Aggregate Bond ETF) basket
# In production this would be loaded from Bloomberg or iShares holdings CSV
#
# EDIT: added a 7th constituent, Agency MBS Pass-Through, at a 23% weight —
# AGG's real portfolio is roughly 23% mortgage-backed securities, and the
# original 6-bond basket had none at all. The other six weights were trimmed
# proportionally to make room for it. "maturity_years": 7.0 for the MBS row
# is a weighted-average-life (WAL) proxy, not the bond's stated 30-year
# term — a 30-year pass-through amortizes and prepays, so its effective
# price sensitivity behaves much more like a ~7-year bullet bond than an
# actual 30-year one. Everything else below is unchanged.
ETF_BASKET = [
    {"ticker": "US2Y",  "label": "US 2Y Treasury",   "face": 1_000_000, "coupon": 0.0456, "maturity_years": 2.0,  "weight": 0.14},
    {"ticker": "US5Y",  "label": "US 5Y Treasury",   "face": 1_000_000, "coupon": 0.0425, "maturity_years": 5.0,  "weight": 0.17},
    {"ticker": "US10Y", "label": "US 10Y Treasury",  "face": 1_000_000, "coupon": 0.0438, "maturity_years": 10.0, "weight": 0.19},
    {"ticker": "US30Y", "label": "US 30Y Treasury",  "face": 1_000_000, "coupon": 0.0450, "maturity_years": 30.0, "weight": 0.12},
    {"ticker": "IG_A",  "label": "IG Corp A-rated",  "face": 1_000_000, "coupon": 0.0520, "maturity_years": 7.0,  "weight": 0.09},
    {"ticker": "IG_BBB","label": "IG Corp BBB-rated","face": 1_000_000, "coupon": 0.0580, "maturity_years": 5.0,  "weight": 0.06},
    {"ticker": "MBS",   "label": "Agency MBS Pass-Through", "face": 1_000_000, "coupon": 0.0450, "maturity_years": 7.0, "weight": 0.23},
]

TOTAL_SHARES = 50_000_000  # ETF shares outstanding


# ---------------------------------------------------------------------------
# Yield curve fetch (FRED proxies via yfinance Treasury ETFs)
# ---------------------------------------------------------------------------

YIELD_PROXIES = {
    "US2Y":   "^IRX",   # 13-week as proxy; ideally use FRED DGS2
    "US5Y":   "^FVX",   # 5-year Treasury yield
    "US10Y":  "^TNX",   # 10-year Treasury yield
    "US30Y":  "^TYX",   # 30-year Treasury yield
    "IG_A":   "^TNX",   # use 10Y + spread as proxy
    "IG_BBB": "^TNX",   # use 10Y + spread as proxy
    "MBS":    "^TNX",   # use 10Y + OAS-like spread as proxy
}

CREDIT_SPREADS = {
    "IG_A":    0.0085,   # ~85bps over Treasury
    "IG_BBB":  0.0150,   # ~150bps over Treasury
    "MBS":     0.0035,   # ~35bps OAS-like spread over Treasury
}


def fetch_yields(lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetch recent yield data for Treasury benchmarks via yfinance.
    Returns a DataFrame of yields indexed by date.
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days)

    tickers = list(set(YIELD_PROXIES.values()))
    raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # yfinance returns Treasury yields as percentages — convert to decimals
    yields = raw / 100.0
    yields.dropna(how="all", inplace=True)
    return yields


def get_latest_ytm(yields_df: pd.DataFrame, ticker: str) -> float:
    """
    Resolve the latest YTM for a constituent, adding credit spread if applicable.
    """
    proxy = YIELD_PROXIES[ticker]
    if proxy not in yields_df.columns:
        return 0.045  # fallback
    base_yield = float(yields_df[proxy].dropna().iloc[-1])
    spread = CREDIT_SPREADS.get(ticker, 0.0)
    return base_yield + spread


# ---------------------------------------------------------------------------
# NAV computation
# ---------------------------------------------------------------------------

def compute_nav(yields_df: pd.DataFrame) -> dict:
    """
    Compute synthetic ETF NAV from constituent bond prices.

    NAV = Sum(constituent bond price * weight) / share_ratio
    Returns a dict with per-constituent detail and aggregate NAV.
    """
    results = []
    total_portfolio_value = 0.0
    total_dv01 = 0.0
    total_duration_contrib = 0.0

    for bond in ETF_BASKET:
        ytm = get_latest_ytm(yields_df, bond["ticker"])
        price = bond_price(bond["face"], bond["coupon"], ytm, bond["maturity_years"])
        mod_dur = modified_duration(bond["face"], bond["coupon"], ytm, bond["maturity_years"])
        dv01_val = dv01(bond["face"], bond["coupon"], ytm, bond["maturity_years"])

        market_value = price * bond["weight"]
        total_portfolio_value += market_value
        total_dv01 += dv01_val * bond["weight"]
        total_duration_contrib += mod_dur * bond["weight"]

        results.append({
            "Label":            bond["label"],
            "Coupon (%)":       round(bond["coupon"] * 100, 2),
            "YTM (%)":          round(ytm * 100, 3),
            "Maturity (yrs)":   bond["maturity_years"],
            "Clean Price":      round(price, 2),
            "Mod Duration":     round(mod_dur, 3),
            "DV01 ($)":         round(dv01_val, 2),
            "Weight":           bond["weight"],
            "Wtd Mkt Value":    round(market_value, 2),
        })

    constituents_df = pd.DataFrame(results)

    # EDIT: NAV scaling fixed so the price is comparable to AGG's real
    # trading range. Previously: total_portfolio_value / TOTAL_SHARES *
    # 1_000_000, an arbitrary shares-outstanding constant that produced a
    # NAV around $19,990 with no relationship to AGG's real ~$95-100 price.
    # Each constituent's "face" is 1,000,000, so total_portfolio_value is a
    # weight-averaged price already scaled to a $1,000,000 face bond.
    # Dividing by 10,000 re-expresses that as the standard "price per $100
    # of face value" bond-quoting convention, which lands directly in
    # AGG's real trading range.
    nav_per_share = total_portfolio_value / 10_000

    return {
        "constituents":        constituents_df,
        "nav_per_share":       round(nav_per_share, 4),
        "portfolio_value":     round(total_portfolio_value, 2),
        "portfolio_dv01":      round(total_dv01, 2),
        "portfolio_duration":  round(total_duration_contrib, 3),
        "as_of":               yields_df.index[-1].strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Premium / Discount tracking
# ---------------------------------------------------------------------------

def fetch_etf_market_price(etf_ticker: str = "AGG", lookback_days: int = 30) -> pd.Series:
    """
    Fetch market close prices for the ETF (used to compute premium/discount
    to NAV). Uses AGG (iShares Core US Aggregate Bond ETF) as the reference
    ETF.
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    data = yf.download(etf_ticker, start=start, end=end, progress=False)["Close"]
    return data.dropna()


def compute_premium_discount(market_price: float, nav: float) -> dict:
    """
    Compute premium (+) or discount (-) of ETF market price vs NAV.
    Positive = ETF trading above NAV (creation opportunity for APs).
    Negative = ETF trading below NAV (redemption opportunity for APs).
    """
    diff = market_price - nav
    pct = (diff / nav) * 100
    signal = "PREMIUM — Creation opportunity" if pct > 0 else "DISCOUNT — Redemption opportunity"
    return {
        "Market Price":    round(market_price, 4),
        "Synthetic NAV":   round(nav, 4),
        "Difference ($)":  round(diff, 4),
        "Premium/Disc (%)":round(pct, 4),
        "Signal":          signal,
    }


# ---------------------------------------------------------------------------
# Historical NAV series (rolling computation over lookback window)
# ---------------------------------------------------------------------------

def compute_historical_nav(yields_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll through each date in yields_df and compute a NAV estimate.
    Returns a time series of synthetic NAV per share.
    """
    records = []
    for date in yields_df.index:
        day_yields = yields_df.loc[[date]]
        nav_data = compute_nav(day_yields)
        records.append({
            "Date":    date,
            "NAV":     nav_data["nav_per_share"],
            "DV01":    nav_data["portfolio_dv01"],
            "Duration":nav_data["portfolio_duration"],
        })
    return pd.DataFrame(records).set_index("Date")


# ---------------------------------------------------------------------------
# Entry point (CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  ETF NAV & Fixed Income Risk Analytics Engine")
    print("=" * 65)

    print("\n[1] Fetching yield curve data...")
    yields = fetch_yields(lookback_days=30)
    print(f"    Data fetched: {yields.index[0].date()} → {yields.index[-1].date()}")

    print("\n[2] Computing Synthetic ETF NAV...")
    nav_result = compute_nav(yields)

    print(f"\n    As of          : {nav_result['as_of']}")
    print(f"    Synthetic NAV  : ${nav_result['nav_per_share']:.4f} per share")
    print(f"    Portfolio DV01 : ${nav_result['portfolio_dv01']:.2f}")
    print(f"    Portfolio Dur  : {nav_result['portfolio_duration']:.3f} years")

    print("\n[3] Constituent Risk Breakdown:")
    print(nav_result["constituents"].to_string(index=False))

    print("\n[4] Fetching ETF market price (AGG)...")
    mkt_prices = fetch_etf_market_price("AGG", lookback_days=5)
    latest_mkt = float(mkt_prices.iloc[-1])

    pd_result = compute_premium_discount(latest_mkt, nav_result["nav_per_share"])
    print(f"\n[5] Premium / Discount Analysis:")
    for k, v in pd_result.items():
        print(f"    {k:<22}: {v}")

    print("\n[6] Building historical NAV series...")
    hist = compute_historical_nav(yields)
    print(f"    {len(hist)} trading days computed.")
    print(hist.tail(5).to_string())

    print("\n[Done] Run visualiser.py for charts and dashboard output.")
