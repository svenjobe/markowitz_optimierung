"""Grafiken des Projekts (matplotlib)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import (
    AQUA, BLUE, GRID, INK, INK_SECONDARY, MAGENTA, ORANGE, SURFACE,
    save_figure, set_title, style_axis,
)

# Feste Farbe je Strategie, in allen Grafiken gleich
STRATEGY_COLORS = {
    "1/N": INK_SECONDARY,
    "Min-Varianz": BLUE,
    "Max-Sharpe": ORANGE,
    "Max-Sharpe (Sigma geschrumpft)": AQUA,
}


def _color(name: str) -> str:
    return STRATEGY_COLORS.get(name, MAGENTA)


def plot_frontier(
    cloud: pd.DataFrame, frontier: pd.DataFrame, assets: pd.DataFrame,
    points: dict[str, tuple[float, float]], title: str, path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(cloud["vol"] * 100, cloud["return"] * 100, s=6, color=GRID, label="Zufallsportfolios", zorder=1)
    ax.plot(frontier["vol"] * 100, frontier["return"] * 100, color=BLUE, linewidth=2.2,
            label="Effizienter Rand", zorder=3)
    ax.scatter(assets["vol"] * 100, assets["return"] * 100, s=28, color=INK_SECONDARY, zorder=4)
    for ticker, row in assets.iterrows():
        ax.annotate(ticker, (row["vol"] * 100, row["return"] * 100), textcoords="offset points",
                    xytext=(4, 4), fontsize=8, color=INK_SECONDARY)
    for name, (ret, vol) in points.items():
        ax.scatter([vol * 100], [ret * 100], s=70, color=_color(name), edgecolor=INK, zorder=5, label=name)
    set_title(ax, title)
    ax.set_xlabel("Volatilitaet (% p. a.)")
    ax.set_ylabel("Erwartete Rendite (% p. a.)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    style_axis(ax, "both")
    save_figure(fig, path)


def plot_weights(weights: pd.DataFrame, path: str) -> None:
    """Gewichte je ETF und Strategie nebeneinander."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(weights.columns)
    width = 0.8 / n
    x = np.arange(len(weights))
    for i, name in enumerate(weights.columns):
        ax.bar(x + (i - (n - 1) / 2) * width, weights[name] * 100, width=width * 0.92,
               color=_color(name), label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(weights.index)
    set_title(ax, "Gewichte je Strategie (aus den Trainingsdaten)")
    ax.set_ylabel("Gewicht (%)")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    save_figure(fig, path)


def plot_wealth(daily: pd.DataFrame, title: str, path: str) -> None:
    wealth = (1.0 + daily).cumprod() * 100.0
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for name in wealth.columns:
        ax.plot(wealth.index, wealth[name], color=_color(name), linewidth=1.8, label=name)
    set_title(ax, title)
    ax.set_ylabel("Wert (Start = 100)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axis(ax)
    save_figure(fig, path)


def plot_correlation(returns: pd.DataFrame, path: str) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7.2, 6))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.6 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    set_title(ax, "Korrelation der Tagesrenditen (Trainingszeitraum)")
    fig.set_facecolor(SURFACE)
    save_figure(fig, path)


def plot_rank_histogram(random_sharpes: np.ndarray, strategy_sharpes: dict[str, float], path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(random_sharpes, bins=50, color=GRID, edgecolor=SURFACE)
    ymax = ax.get_ylim()[1]
    for k, (name, value) in enumerate(strategy_sharpes.items()):
        ax.axvline(value, color=_color(name), linewidth=2.2, label=f"{name} ({value:.2f})")
    ax.set_ylim(0, ymax * 1.05)
    set_title(ax, "Out-of-Sample: Sharpe Ratio der Strategien gegen 5.000 Zufallsportfolios")
    ax.set_xlabel("Realisierte Sharpe Ratio im Testzeitraum")
    ax.set_ylabel("Anzahl Portfolios")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)
    save_figure(fig, path)