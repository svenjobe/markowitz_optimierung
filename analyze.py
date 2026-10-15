#!/usr/bin/env python3
"""
Markowitz-Optimierung mit Out-of-Sample-Test.

Ablauf:
1. Kurse laden (bis zum festen Enddatum in src/config.py), Renditen berechnen
2. Optimierung auf den Trainingsdaten: geschlossene Loesung, Long-only-Loesung,
   effizienter Rand
3. Out-of-Sample-Test der Strategien (statisch und Walk-Forward)
4. Vergleich mit 5.000 zufaelligen Portfolios

Aufruf:
    python analyze.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src import config
from src.backtest import (
    make_strategies, percentile_rank, performance_table, random_portfolio_sharpes,
    split_train_test, static_backtest, walk_forward_all,
)
from src.data_loader import daily_returns, load_prices
from src.optimizer import (
    annual_stats, condition_number, efficient_frontier, min_variance,
    min_variance_closed_form, portfolio_performance, random_weights, shrink_covariance,
)
from src.visualize import (
    plot_correlation, plot_frontier, plot_rank_histogram, plot_wealth, plot_weights,
)


def pct_table(table: pd.DataFrame) -> pd.DataFrame:
    """Prozentwerte fuer die Konsolenausgabe."""
    out = table.copy()
    for col in ("Rendite p.a.", "Volatilitaet p.a.", "Max. Drawdown"):
        out[col] = (out[col] * 100).round(1).astype(str) + " %"
    out["Sharpe Ratio"] = out["Sharpe Ratio"].round(2)
    return out


def main() -> None:
    out = config.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    rf, cap = config.RISK_FREE_RATE, config.MAX_WEIGHT

    # 1. Daten
    prices = load_prices()
    returns = daily_returns(prices)
    train, test = split_train_test(returns, config.TRAIN_END)
    print(f"Daten: {returns.index[0]:%d.%m.%Y} bis {returns.index[-1]:%d.%m.%Y}, {len(returns)} Handelstage")
    print(f"Training: {len(train)} Tage bis {train.index[-1]:%d.%m.%Y}, Test: {len(test)} Tage ab {test.index[0]:%d.%m.%Y}")

    # 2. Optimierung auf den Trainingsdaten
    mu, cov = annual_stats(train)
    cov_shrunk = shrink_covariance(cov, config.SHRINKAGE)
    print(f"\nKondition der Kovarianzmatrix: {condition_number(cov):.0f} "
          f"(nach Schrumpfung mit {config.SHRINKAGE:.0%}: {condition_number(cov_shrunk):.0f})")

    w_closed = pd.Series(min_variance_closed_form(cov), index=returns.columns)
    w_long = pd.Series(min_variance(cov, cap), index=returns.columns)
    compare = pd.DataFrame({"Geschlossen (Leerverkauf erlaubt)": w_closed, f"Long-only (max. {cap:.0%})": w_long})
    print("\nMinimum-Varianz-Portfolio: geschlossene Loesung gegen Long-only")
    print((compare * 100).round(1).to_string())
    compare.to_csv(os.path.join(out, "min_varianz_vergleich.csv"))

    strategies = make_strategies(rf, cap, config.SHRINKAGE)
    weights, in_sample, out_sample = static_backtest(returns, config.TRAIN_END, strategies)
    weights.to_csv(os.path.join(out, "gewichte.csv"))
    print("\nGewichte (%):")
    print((weights * 100).round(1).to_string())

    frontier = efficient_frontier(mu, cov, config.N_FRONTIER_POINTS, cap)
    rand_w = random_weights(len(mu), config.N_RANDOM_PORTFOLIOS, cap, config.SEED)
    cloud = pd.DataFrame({
        "return": rand_w @ mu.to_numpy(),
        "vol": np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov.to_numpy(), rand_w)),
    })
    assets = pd.DataFrame({"return": mu, "vol": np.sqrt(np.diag(cov.to_numpy()))}, index=mu.index)
    points = {name: portfolio_performance(weights[name], mu, cov) for name in ("1/N", "Min-Varianz", "Max-Sharpe")}
    plot_frontier(cloud, frontier, assets, points,
                  f"Effizienter Rand (Trainingsdaten bis {train.index[-1]:%d.%m.%Y})",
                  os.path.join(out, "effizienter_rand.png"))
    plot_weights(weights, os.path.join(out, "gewichte.png"))
    plot_correlation(train, os.path.join(out, "korrelation.png"))

    # 3. Out-of-Sample
    in_table = performance_table(in_sample, rf)
    out_table = performance_table(out_sample, rf)
    print("\nIn-Sample (Training):")
    print(pct_table(in_table).to_string())
    print("\nOut-of-Sample (Test):")
    print(pct_table(out_table).to_string())
    in_table.to_csv(os.path.join(out, "kennzahlen_training.csv"))
    out_table.to_csv(os.path.join(out, "kennzahlen_test.csv"))
    plot_wealth(out_sample, f"Out-of-Sample: Wertentwicklung ab {test.index[0]:%d.%m.%Y}",
                os.path.join(out, "wertentwicklung_test.png"))

    wf = walk_forward_all(returns, strategies, config.WF_TRAIN_DAYS, config.WF_REBALANCE_DAYS)
    wf_table = performance_table(wf, rf)
    print(f"\nWalk-Forward (Fenster {config.WF_TRAIN_DAYS} Tage, Neuoptimierung alle {config.WF_REBALANCE_DAYS} Tage):")
    print(pct_table(wf_table).to_string())
    wf_table.to_csv(os.path.join(out, "kennzahlen_walk_forward.csv"))
    plot_wealth(wf, f"Walk-Forward: Wertentwicklung ab {wf.index[0]:%d.%m.%Y}",
                os.path.join(out, "wertentwicklung_walk_forward.png"))

    # 4. Zufallsportfolios im Testzeitraum
    random_sharpes = random_portfolio_sharpes(test, rand_w, rf)
    strategy_sharpes = out_table["Sharpe Ratio"].to_dict()
    ranks = {name: percentile_rank(value, random_sharpes) for name, value in strategy_sharpes.items()}
    rank_table = pd.DataFrame({"Sharpe Ratio (Test)": strategy_sharpes, "Rang unter Zufallsportfolios": ranks})
    print("\nRang im Test (1,00 = besser als alle 5.000 Zufallsportfolios):")
    print(rank_table.round(2).to_string())
    rank_table.to_csv(os.path.join(out, "zufallsvergleich.csv"))
    plot_rank_histogram(random_sharpes, strategy_sharpes, os.path.join(out, "zufallsvergleich.png"))

    print(f"\nFertig. Ergebnisse in '{out}/'.")


if __name__ == "__main__":
    main() 