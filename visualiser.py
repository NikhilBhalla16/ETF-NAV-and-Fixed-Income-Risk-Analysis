"""
ETF NAV Analytics — Visualiser
================================
Generates desk-ready charts:
  1. Historical synthetic NAV vs ETF market price
  2. Premium / Discount over time
  3. Constituent DV01 risk heatmap
  4. Portfolio duration trend

Run after nav_engine.py has been verified.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from nav_engine import (
    fetch_yields,
    compute_nav,
    compute_historical_nav,
    fetch_etf_market_price,
    ETF_BASKET,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
})

ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
ORANGE   = "#d29922"
PURPLE   = "#bc8cff"
GREY     = "#8b949e"


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def load_data(lookback: int = 30):
    yields     = fetch_yields(lookback_days=lookback)
    hist_nav   = compute_historical_nav(yields)
    mkt_prices = fetch_etf_market_price("AGG", lookback_days=lookback)

    # Align on common dates
    merged = hist_nav.join(mkt_prices.rename("Market Price"), how="inner")

    # Compute premium/discount series
    merged["Prem/Disc (%)"] = (merged["Market Price"] - merged["NAV"]) / merged["NAV"] * 100

    # Latest constituent snapshot
    latest_nav = compute_nav(yields)

    return merged, latest_nav


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def plot_nav_vs_market(ax, df: pd.DataFrame):
    ax.plot(df.index, df["NAV"],          color=ACCENT,  lw=1.6, label="Synthetic NAV")
    ax.plot(df.index, df["Market Price"], color=ORANGE,  lw=1.6, label="Market Price (AGG)", linestyle="--")
    ax.fill_between(df.index,
                    df["NAV"], df["Market Price"],
                    where=df["Market Price"] >= df["NAV"],
                    alpha=0.15, color=GREEN,  label="Premium")
    ax.fill_between(df.index,
                    df["NAV"], df["Market Price"],
                    where=df["Market Price"] < df["NAV"],
                    alpha=0.15, color=RED,    label="Discount")
    ax.set_title("Synthetic NAV vs Market Price")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def plot_premium_discount(ax, df: pd.DataFrame):
    colors = [GREEN if v >= 0 else RED for v in df["Prem/Disc (%)"]]
    ax.bar(df.index, df["Prem/Disc (%)"], color=colors, width=0.6, alpha=0.85)
    ax.axhline(0, color=GREY, lw=0.8, linestyle="--")
    ax.set_title("Daily Premium / Discount to NAV (%)")
    ax.set_ylabel("Prem / Disc (%)")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Annotate latest value
    last_val = df["Prem/Disc (%)"].iloc[-1]
    clr = GREEN if last_val >= 0 else RED
    ax.annotate(f"Latest: {last_val:+.3f}%",
                xy=(df.index[-1], last_val),
                xytext=(-45, 12), textcoords="offset points",
                fontsize=7, color=clr,
                arrowprops=dict(arrowstyle="->", color=clr, lw=0.8))


def plot_dv01_heatmap(ax, latest_nav: dict):
    df = latest_nav["constituents"][["Label", "DV01 ($)", "Mod Duration", "Weight"]].copy()
    df = df.sort_values("DV01 ($)", ascending=True)

    bars = ax.barh(df["Label"], df["DV01 ($)"],
                   color=[plt.cm.RdYlGn_r(w) for w in np.linspace(0.1, 0.85, len(df))],
                   height=0.55, edgecolor="#21262d")

    for bar, val in zip(bars, df["DV01 ($)"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=6.5, color="#c9d1d9")

    ax.set_title("Constituent DV01 Risk ($) — per Unit Weight")
    ax.set_xlabel("DV01 ($)")
    ax.grid(True, axis="x")
    ax.set_xlim(0, df["DV01 ($)"].max() * 1.25)


def plot_duration_trend(ax, df: pd.DataFrame):
    ax.plot(df.index, df["Duration"], color=PURPLE, lw=1.6)
    ax.fill_between(df.index, df["Duration"].min() * 0.998, df["Duration"],
                    alpha=0.15, color=PURPLE)
    ax.set_title("Portfolio Modified Duration (years)")
    ax.set_ylabel("Duration (yrs)")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))


# ---------------------------------------------------------------------------
# Master dashboard
# ---------------------------------------------------------------------------

def build_dashboard(output_path: str = "etf_nav_dashboard.png"):
    print("Fetching data...")
    merged, latest_nav = load_data(lookback=30)

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        f"ETF NAV & Fixed Income Risk Analytics Dashboard   |   As of {latest_nav['as_of']}   |   "
        f"Synthetic NAV: ${latest_nav['nav_per_share']:.4f}   |   "
        f"Portfolio DV01: ${latest_nav['portfolio_dv01']:,.2f}   |   "
        f"Duration: {latest_nav['portfolio_duration']:.3f}y",
        fontsize=9, color="#c9d1d9", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_nav_vs_market(ax1, merged)
    plot_premium_discount(ax2, merged)
    plot_dv01_heatmap(ax3, latest_nav)
    plot_duration_trend(ax4, merged)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Dashboard saved → {output_path}")
    return output_path


if __name__ == "__main__":
    build_dashboard("etf_nav_dashboard.png")
