"""
Out-of-Sample-Auswertung der Strategien.

Statischer Test: Gewichte werden nur mit Daten bis TRAIN_END bestimmt und dann
unveraendert auf den Testzeitraum angewendet.
Walk-Forward: Gewichte werden regelmaessig mit einem rollierenden Fenster neu
bestimmt, und es werden immer nur Daten vor dem Anlagezeitpunkt verwendet.

Annahme: Die Gewichte werden taeglich auf ihren Zielwert zurueckgesetzt
(konstante Gewichte), Transaktionskosten bleiben unberuecksichtigt.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.optimizer import annual_stats, max_sharpe, min_variance, shrink_covariance

Strategy = Callable[[pd.DataFrame], np.ndarray]


# ---------------------------------------------------------------------------
# Strategien: jede bekommt die Trainings-Renditen und liefert Gewichte
# ---------------------------------------------------------------------------

def equal_weight(train: pd.DataFrame) -> np.ndarray:
    """1/N: alle Titel gleich gewichtet, keine Schaetzung noetig."""
    n = train.shape[1]
    return np.full(n, 1.0 / n)


def make_strategies(rf: float, max_weight: float, shrinkage: float) -> dict[str, Strategy]:
    """Die verglichenen Strategien als Name -> Funktion."""

    def min_var(train: pd.DataFrame) -> np.ndarray:
        _, cov = annual_stats(train)
        return min_variance(cov, max_weight)

    def tangency(train: pd.DataFrame) -> np.ndarray:
        mu, cov = annual_stats(train)
        return max_sharpe(mu, cov, rf, max_weight)

    def tangency_shrunk(train: pd.DataFrame) -> np.ndarray:
        mu, cov = annual_stats(train)
        return max_sharpe(mu, shrink_covariance(cov, shrinkage), rf, max_weight)

    return {
        "1/N": equal_weight,
        "Min-Varianz": min_var,
        "Max-Sharpe": tangency,
        "Max-Sharpe (Sigma geschrumpft)": tangency_shrunk,
    }


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def split_train_test(returns: pd.DataFrame, train_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trennt die Renditen am Datum train_end (dieses Datum gehoert noch zum Training)."""
    cut = pd.Timestamp(train_end)
    train, test = returns.loc[:cut], returns.loc[cut + pd.Timedelta(days=1):]
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Trainings- oder Testzeitraum ist leer. TRAIN_END pruefen.")
    return train, test


def portfolio_returns(returns: pd.DataFrame, weights) -> pd.Series:
    """Tagesrendite eines Portfolios mit konstanten Gewichten."""
    return returns @ np.asarray(weights, dtype=float)


def realized_performance(daily: pd.Series, rf: float, periods: int = 252) -> dict[str, float]:
    """Tatsaechlich realisierte Kennzahlen einer Tagesrenditen-Reihe."""
    n = daily.count()
    ann_ret = float((1.0 + daily).prod() ** (periods / n) - 1.0)
    ann_vol = float(daily.std(ddof=1) * np.sqrt(periods))
    wealth = (1.0 + daily).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "Rendite p.a.": ann_ret,
        "Volatilitaet p.a.": ann_vol,
        "Sharpe Ratio": (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan"),
        "Max. Drawdown": max_dd,
    }


def performance_table(daily: pd.DataFrame, rf: float) -> pd.DataFrame:
    """Kennzahlen je Strategie (eine Zeile pro Strategie)."""
    rows = {name: realized_performance(daily[name], rf) for name in daily.columns}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Statischer Out-of-Sample-Test
# ---------------------------------------------------------------------------

def static_backtest(
    returns: pd.DataFrame, train_end: str, strategies: dict[str, Strategy]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gewichte aus dem Training, angewendet auf Training und Test.

    Rueckgabe: (Gewichte je Strategie, Tagesrenditen im Training, Tagesrenditen im Test).
    """
    train, test = split_train_test(returns, train_end)
    weights = pd.DataFrame({name: fn(train) for name, fn in strategies.items()}, index=returns.columns)
    in_sample = pd.DataFrame({name: portfolio_returns(train, weights[name]) for name in weights})
    out_sample = pd.DataFrame({name: portfolio_returns(test, weights[name]) for name in weights})
    return weights, in_sample, out_sample


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

def walk_forward(
    returns: pd.DataFrame, strategy: Strategy, train_days: int, rebalance_days: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Rollierende Neuoptimierung.

    Zum Zeitpunkt t werden nur die letzten train_days Renditen VOR t verwendet.
    Die Gewichte gelten dann fuer die naechsten rebalance_days Handelstage.
    Rueckgabe: (Tagesrenditen ab dem ersten Anlagezeitpunkt, Gewichte je Neuoptimierung).
    """
    if len(returns) <= train_days:
        raise ValueError("Zu wenig Daten fuer das Trainingsfenster.")
    pieces, history = [], {}
    for start in range(train_days, len(returns), rebalance_days):
        train = returns.iloc[start - train_days:start]
        w = strategy(train)
        block = returns.iloc[start:start + rebalance_days]
        pieces.append(portfolio_returns(block, w))
        history[returns.index[start]] = w
    weights = pd.DataFrame(history, index=returns.columns).T
    return pd.concat(pieces), weights


def walk_forward_all(
    returns: pd.DataFrame, strategies: dict[str, Strategy], train_days: int, rebalance_days: int
) -> pd.DataFrame:
    """Walk-Forward fuer alle Strategien, Tagesrenditen nebeneinander."""
    series = {name: walk_forward(returns, fn, train_days, rebalance_days)[0]
              for name, fn in strategies.items()}
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Vergleich mit Zufallsportfolios
# ---------------------------------------------------------------------------

def random_portfolio_sharpes(test: pd.DataFrame, weights: np.ndarray, rf: float, periods: int = 252) -> np.ndarray:
    """Realisierte Sharpe Ratio je Zufallsportfolio im Testzeitraum (vektorisiert)."""
    daily = test.to_numpy() @ weights.T                    # (Tage, Portfolios)
    n_days = daily.shape[0]
    ann_ret = np.prod(1.0 + daily, axis=0) ** (periods / n_days) - 1.0
    ann_vol = daily.std(axis=0, ddof=1) * np.sqrt(periods)
    return (ann_ret - rf) / ann_vol


def percentile_rank(value: float, sample: np.ndarray) -> float:
    """Anteil der Zufallswerte, die kleiner sind als value (0 = schlechtester, 1 = bester)."""
    return float(np.mean(sample < value))