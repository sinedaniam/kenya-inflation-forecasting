"""Forecasting Kenya's annual CPI inflation.

Pulls the World Bank series FP.CPI.TOTL.ZG, compares simple benchmarks with
ARIMA models in a rolling-origin backtest, then forecasts three years ahead.

Run from the repo root:  python src/analysis.py
"""
import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kenya_inflation.csv"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"
INDICATOR = "FP.CPI.TOTL.ZG"
BACKTEST_START = 2005
HORIZON = 3
FINAL_START = 1996  # post-liberalisation regime; excludes the early-1990s spike


def load_series() -> pd.Series:
    """Download the series once, then reuse the cached CSV."""
    if not DATA.exists():
        url = f"https://api.worldbank.org/v2/country/KEN/indicator/{INDICATOR}"
        rows = requests.get(url, params={"format": "json", "per_page": 200}, timeout=30).json()[1]
        df = pd.DataFrame({"year": int(r["date"]), "inflation": r["value"]} for r in rows)
        DATA.parent.mkdir(exist_ok=True)
        df.dropna().sort_values("year").to_csv(DATA, index=False)
    df = pd.read_csv(DATA)
    return df.set_index("year")["inflation"]


def select_arima(y: pd.Series) -> tuple:
    """Pick (p, d, q) by AIC over a small grid."""
    best, best_aic = (1, 0, 0), np.inf
    for order in itertools.product(range(3), range(2), range(3)):
        try:
            aic = ARIMA(y.values, order=order).fit().aic
        except Exception:
            continue
        if aic < best_aic:
            best, best_aic = order, aic
    return best


def one_step_forecasts(y: pd.Series, order: tuple) -> pd.DataFrame:
    """Expanding-window, one-year-ahead forecasts from each model."""
    out = []
    for year in y.index[y.index >= BACKTEST_START]:
        train = y[y.index < year]
        out.append({
            "year": year,
            "actual": y[year],
            "Naive (last year)": train.iloc[-1],
            "Historical mean": train.mean(),
            "AR(1)": ARIMA(train.values, order=(1, 0, 0)).fit().forecast(1)[0],
            f"ARIMA{order}": ARIMA(train.values, order=order).fit().forecast(1)[0],
        })
    return pd.DataFrame(out).set_index("year")


def score(bt: pd.DataFrame) -> pd.DataFrame:
    err = bt.drop(columns="actual").sub(bt["actual"], axis=0)
    return pd.DataFrame({
        "RMSE": np.sqrt((err**2).mean()),
        "MAE": err.abs().mean(),
    }).round(2).sort_values("RMSE")


def main():
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    y = load_series()
    print(f"Observations: {len(y)} ({y.index.min()}-{y.index.max()})")

    adf_p = adfuller(y.values)[1]
    print(f"ADF test p-value: {adf_p:.3f}")

    order = select_arima(y[y.index < BACKTEST_START])
    print(f"ARIMA order chosen on pre-{BACKTEST_START} data: {order}")

    bt = one_step_forecasts(y, order)
    scores = score(bt)
    print("\nOne-step-ahead backtest", f"{BACKTEST_START}-{y.index.max()}:\n", scores)
    scores.to_csv(RESULTS / "backtest_scores.csv")
    bt.round(2).to_csv(RESULTS / "backtest_forecasts.csv")

    # Final model on the recent regime: the time-series model with the lower backtest RMSE.
    models = {"AR(1)": (1, 0, 0), f"ARIMA{order}": order}
    final_order = models[min(models, key=lambda m: scores.loc[m, "RMSE"])]
    fit = ARIMA(y[y.index >= FINAL_START].values, order=final_order).fit()
    fc = fit.get_forecast(HORIZON)
    years = np.arange(y.index.max() + 1, y.index.max() + 1 + HORIZON)
    ci80, ci95 = fc.conf_int(alpha=0.20), fc.conf_int(alpha=0.05)
    forecast = pd.DataFrame({
        "year": years,
        "forecast": fc.predicted_mean,
        "lo80": ci80[:, 0], "hi80": ci80[:, 1],
        "lo95": ci95[:, 0], "hi95": ci95[:, 1],
    }).round(2)
    forecast.to_csv(RESULTS / "forecast.csv", index=False)
    print(f"\nForecast with ARIMA{final_order}:\n", forecast.to_string(index=False))

    # Figure 1: history and forecast fan.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y.index, y.values, color="#1f4e79", lw=1.8, label="Actual")
    ax.plot(years, forecast["forecast"], color="#c0504d", lw=2, label=f"ARIMA{final_order} forecast")
    ax.fill_between(years, forecast["lo95"], forecast["hi95"], color="#c0504d", alpha=0.12, label="95% interval")
    ax.fill_between(years, forecast["lo80"], forecast["hi80"], color="#c0504d", alpha=0.25, label="80% interval")
    ax.axhspan(2.5, 7.5, color="grey", alpha=0.08, label="CBK target band (5% ± 2.5)")
    ax.set(title="Kenya: annual CPI inflation and 3-year forecast", ylabel="% change", xlabel="")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "inflation_forecast.png", dpi=150)

    # Figure 2: backtest forecasts vs actual.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(bt.index, bt["actual"], color="black", lw=2, label="Actual")
    for col, c in zip(bt.columns[1:], ["#7f7f7f", "#bcbd22", "#2ca02c", "#c0504d"]):
        ax.plot(bt.index, bt[col], lw=1.3, color=c, label=f"{col} (RMSE {scores.loc[col, 'RMSE']})")
    ax.set(title="One-year-ahead forecasts in the backtest", ylabel="% change")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "backtest.png", dpi=150)


if __name__ == "__main__":
    main()
