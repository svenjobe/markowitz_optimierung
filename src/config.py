"""
Konfiguration der Markowitz-Optimierung mit Out-of-Sample-Test.

Alle Einstellungen stehen an einer Stelle. Aenderungen an Titeln, Zeitraum
oder Grenzen werden nur hier vorgenommen.
"""

from __future__ import annotations

# Zehn ETFs aus verschiedenen Anlageklassen (Yahoo-Finance-Ticker)
TICKERS: dict[str, str] = {
    "SPY": "US-Aktien gross",
    "QQQ": "US-Technologie (Nasdaq 100)",
    "IWM": "US-Aktien klein",
    "EFA": "Industrielaender ohne USA",
    "EEM": "Schwellenlaender",
    "TLT": "US-Staatsanleihen lang",
    "LQD": "US-Unternehmensanleihen",
    "GLD": "Gold",
    "VNQ": "US-Immobilien",
    "DBC": "Rohstoffe",
}

# Betrachtungszeitraum mit festem Enddatum
START_DATE: str = "2016-08-15"
END_DATE: str = "2026-08-14"

# Aufteilung: bis TRAIN_END wird optimiert, danach nur getestet (Out-of-Sample)
TRAIN_END: str = "2023-12-29"

# Walk-Forward: Trainingsfenster und Neuoptimierung in Handelstagen
WF_TRAIN_DAYS: int = 756      # ca. 3 Jahre
WF_REBALANCE_DAYS: int = 63   # ca. ein Quartal

# Optimierung
RISK_FREE_RATE: float = 0.03  # Naeherung, in Prozent p. a. als Dezimalzahl
MAX_WEIGHT: float = 0.30      # Obergrenze je ETF, keine Leerverkaeufe
TRADING_DAYS: int = 252
N_FRONTIER_POINTS: int = 40
N_RANDOM_PORTFOLIOS: int = 5000
SHRINKAGE: float = 0.2        # Anteil der Diagonale bei der Schrumpfung von Sigma
SEED: int = 42

DATA_DIR: str = "data"
OUTPUT_DIR: str = "output"