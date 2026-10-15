
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Grundgroessen
# ---------------------------------------------------------------------------

def annual_stats(returns: pd.DataFrame, periods: int = 252) -> tuple[pd.Series, pd.DataFrame]:
    return returns.mean() * periods, returns.cov() * periods


def portfolio_performance(w, mu: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu.to_numpy())
    vol = float(np.sqrt(w @ cov.to_numpy() @ w))
    return ret, vol


def shrink_covariance(cov: pd.DataFrame, delta: float) -> pd.DataFrame:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta muss zwischen 0 und 1 liegen.")
    diag = pd.DataFrame(np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns)
    return (1.0 - delta) * cov + delta * diag


def condition_number(cov: pd.DataFrame) -> float:
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    return float(eigenvalues.max() / eigenvalues.min())


def min_variance_closed_form(cov: pd.DataFrame) -> np.ndarray:
    """Minimum-Varianz-Portfolio ohne weitere Nebenbedingung als sum(w) = 1.

    Lagrange: L = w' S w - lambda (1'w - 1). Ableiten und null setzen ergibt
    2 S w = lambda 1, also w = S^-1 1 / (1' S^-1 1). Es wird das Gleichungssystem
    S x = 1 geloest (statt die Inverse zu bilden) und anschliessend normiert.
    """
    ones = np.ones(len(cov))
    x = np.linalg.solve(cov.to_numpy(), ones)
    return x / x.sum()


def tangency_closed_form(mu: pd.Series, cov: pd.DataFrame, rf: float) -> np.ndarray:
    x = np.linalg.solve(cov.to_numpy(), mu.to_numpy() - rf)
    return x / x.sum()


# ---------------------------------------------------------------------------
# Numerische Loesungen (long-only, Obergrenze)
# ---------------------------------------------------------------------------

def _check_feasible(n: int, max_weight: float) -> None:
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Mit {n} Titeln und Obergrenze {max_weight:.0%} laesst sich keine Summe von 100 % erreichen."
        )


def _solve(objective, x0, bounds, constraints, jac=None) -> np.ndarray:
    res = minimize(objective, x0, jac=jac, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Optimierung nicht konvergiert: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def min_variance(cov: pd.DataFrame, max_weight: float = 1.0) -> np.ndarray:
    n = len(cov)
    _check_feasible(n, max_weight)
    c = cov.to_numpy()
    return _solve(
        lambda w: w @ c @ w,
        np.full(n, 1.0 / n),
        [(0.0, max_weight)] * n,
        [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        jac=lambda w: 2.0 * c @ w,
    )


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, max_weight: float = 1.0) -> np.ndarray:
    n = len(mu)
    _check_feasible(n, max_weight)
    m, c = mu.to_numpy(), cov.to_numpy()

    def neg_sharpe(w: np.ndarray) -> float:
        return -(w @ m - rf) / np.sqrt(w @ c @ w)

    return _solve(
        neg_sharpe,
        np.full(n, 1.0 / n),
        [(0.0, max_weight)] * n,
        [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )


def max_return(mu: pd.Series, max_weight: float = 1.0) -> np.ndarray:
    n = len(mu)
    _check_feasible(n, max_weight)
    w = np.zeros(n)
    remaining = 1.0
    for idx in np.argsort(-mu.to_numpy()):
        take = min(max_weight, remaining)
        w[idx] = take
        remaining -= take
        if remaining <= 1e-12:
            break
    return w


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, n_points: int = 40, max_weight: float = 1.0
) -> pd.DataFrame:
    n = len(mu)
    _check_feasible(n, max_weight)
    m, c = mu.to_numpy(), cov.to_numpy()

    w_min = min_variance(cov, max_weight)
    r_low = float(w_min @ m)
    r_high = float(max_return(mu, max_weight) @ m) - 1e-6
    targets = np.linspace(r_low, max(r_low, r_high), n_points)

    rows = []
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ m - t},
        ]
        try:
            w = _solve(lambda w: w @ c @ w, w_min, [(0.0, max_weight)] * n, constraints,
                       jac=lambda w: 2.0 * c @ w)
        except RuntimeError:
            continue  # einzelne nicht loesbare Zielrenditen ueberspringen
        ret, vol = portfolio_performance(w, mu, cov)
        rows.append({"return": ret, "vol": vol, "weights": w})
    return pd.DataFrame(rows)


def random_weights(n_assets: int, n: int, max_weight: float, seed: int = 42) -> np.ndarray:
    _check_feasible(n_assets, max_weight)
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    while len(kept) < n:
        draws = rng.dirichlet(np.ones(n_assets), size=n * 4)
        kept.extend(draws[draws.max(axis=1) <= max_weight])
    return np.array(kept[:n])