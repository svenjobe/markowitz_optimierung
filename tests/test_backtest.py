"""Tests des Out-of-Sample-Backtests. Offline, mit synthetischen Renditen."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    equal_weight, make_strategies, percentile_rank, performance_table, portfolio_returns,
    random_portfolio_sharpes, realized_performance, split_train_test, static_backtest,
    walk_forward,
)


def _returns(n_days=1000, n_assets=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = rng.normal(0.0004, 0.01, (n_days, n_assets)) + rng.normal(0, 0.004, (n_days, 1))
    return pd.DataFrame(data, index=idx, columns=list("ABCD")[:n_assets])


def test_split_puts_train_end_into_training_and_has_no_overlap():
    r = _returns(100)
    cut = r.index[59]
    train, test = split_train_test(r, str(cut.date()))
    assert train.index[-1] == cut
    assert test.index[0] > cut
    assert len(train) + len(test) == len(r)
    assert train.index.intersection(test.index).empty


def test_split_with_empty_part_raises():
    r = _returns(50)
    with pytest.raises(ValueError):
        split_train_test(r, "2000-01-01")
    with pytest.raises(ValueError):
        split_train_test(r, "2100-01-01")


def test_equal_weight_and_portfolio_returns():
    r = _returns(10, 4)
    w = equal_weight(r)
    assert w.tolist() == pytest.approx([0.25] * 4)
    assert portfolio_returns(r, w).iloc[0] == pytest.approx(r.iloc[0].mean())


def test_realized_performance_known_values():
    constant = pd.Series([0.01] * 252)
    perf = realized_performance(constant, rf=0.0)
    assert perf["Rendite p.a."] == pytest.approx(1.01 ** 252 - 1)
    assert perf["Max. Drawdown"] == pytest.approx(0.0)
    drawdown = realized_performance(pd.Series([0.10, -0.50, 0.05, 0.01]), rf=0.0)
    assert drawdown["Max. Drawdown"] == pytest.approx(-0.50)


def test_static_backtest_weights_valid_and_use_only_training_data():
    r = _returns(800)
    strategies = make_strategies(rf=0.02, max_weight=0.5, shrinkage=0.2)
    cut = str(r.index[499].date())
    weights, in_s, out_s = static_backtest(r, cut, strategies)
    assert list(weights.columns) == list(strategies)
    assert weights.sum().tolist() == pytest.approx([1.0] * len(strategies))
    assert len(in_s) == 500 and len(out_s) == 300

    # Test-Renditen veraendern darf die Gewichte nicht beeinflussen (kein Look-Ahead)
    changed = r.copy()
    changed.iloc[500:] = changed.iloc[500:] * 5 + 0.01
    weights2, _, _ = static_backtest(changed, cut, strategies)
    assert weights2.to_numpy() == pytest.approx(weights.to_numpy())


def test_walk_forward_uses_only_past_data_and_covers_all_remaining_days():
    r = _returns(400)
    seen = []

    def recorder(train: pd.DataFrame) -> np.ndarray:
        seen.append((train.index[0], train.index[-1], len(train)))
        return equal_weight(train)

    daily, weights = walk_forward(r, recorder, train_days=100, rebalance_days=50)
    assert len(daily) == 300                      # 400 - 100 Tage Training
    assert daily.index[0] == r.index[100]
    assert len(weights) == 6                      # 300 / 50 Neuoptimierungen
    for (_, last, length), start in zip(seen, weights.index):
        assert length == 100
        assert last < start                       # Training endet vor dem Anlagezeitpunkt


def test_walk_forward_with_too_little_data_raises():
    with pytest.raises(ValueError):
        walk_forward(_returns(50), equal_weight, train_days=100, rebalance_days=20)


def test_random_portfolio_sharpes_match_single_portfolio_calculation():
    r = _returns(300)
    rng = np.random.default_rng(3)
    w = rng.dirichlet(np.ones(4), size=5)
    fast = random_portfolio_sharpes(r, w, rf=0.02)
    slow = [realized_performance(portfolio_returns(r, row), 0.02)["Sharpe Ratio"] for row in w]
    assert fast == pytest.approx(slow)


def test_percentile_rank():
    sample = np.array([0.1, 0.2, 0.3, 0.4])
    assert percentile_rank(0.35, sample) == pytest.approx(0.75)
    assert percentile_rank(0.0, sample) == 0.0
    assert percentile_rank(1.0, sample) == 1.0


def test_performance_table_has_one_row_per_strategy():
    r = _returns(300)
    daily = pd.DataFrame({"x": portfolio_returns(r, equal_weight(r)),
                          "y": portfolio_returns(r, [1, 0, 0, 0])})
    table = performance_table(daily, rf=0.02)
    assert list(table.index) == ["x", "y"]
    assert {"Rendite p.a.", "Volatilitaet p.a.", "Sharpe Ratio", "Max. Drawdown"} == set(table.columns)