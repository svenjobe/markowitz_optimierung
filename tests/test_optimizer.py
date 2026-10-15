
import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    annual_stats, condition_number, efficient_frontier, max_return, max_sharpe,
    min_variance, min_variance_closed_form, portfolio_performance, random_weights,
    shrink_covariance, tangency_closed_form,
)

NAMES = ["A", "B", "C"]


def _uncorrelated():
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.16]), index=NAMES, columns=NAMES)
    mu = pd.Series([0.05, 0.08, 0.10], index=NAMES)
    return mu, cov


def _random_cov(n=6, seed=0):
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(rng.normal(0.0004, 0.01, (1000, n)) + rng.normal(0, 0.005, (1000, 1)),
                           columns=[f"X{i}" for i in range(n)])
    return annual_stats(returns)


def test_min_variance_closed_form_matches_hand_calculation():
    # w proportional zu 1/Varianz: 25 : 11,11 : 6,25
    _, cov = _uncorrelated()
    w = min_variance_closed_form(cov)
    expected = np.array([25.0, 1 / 0.09, 6.25])
    assert w == pytest.approx(expected / expected.sum())
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_closed_form_satisfies_first_order_condition():
    # Lagrange-Bedingung: Sigma w ist ein Vielfaches des Einsvektors
    _, cov = _random_cov()
    w = min_variance_closed_form(cov)
    grad = cov.to_numpy() @ w
    assert grad == pytest.approx(np.full(len(w), grad[0]), rel=1e-9)


def test_min_variance_closed_form_has_lowest_variance_among_alternatives():
    _, cov = _random_cov()
    w = min_variance_closed_form(cov)
    rng = np.random.default_rng(1)
    c = cov.to_numpy()
    for _ in range(200):
        other = rng.normal(size=len(w))
        other = other / other.sum()
        assert w @ c @ w <= other @ c @ other + 1e-12


def test_tangency_closed_form_matches_hand_calculation():
    # w proportional zu (mu - rf) / Varianz: 0,75 : 0,667 : 0,5
    mu, cov = _uncorrelated()
    w = tangency_closed_form(mu, cov, rf=0.02)
    expected = np.array([0.03 / 0.04, 0.06 / 0.09, 0.08 / 0.16])
    assert w == pytest.approx(expected / expected.sum())


def test_numeric_solver_matches_closed_form_when_no_bound_is_active():
    mu, cov = _uncorrelated()
    assert min_variance(cov, max_weight=1.0) == pytest.approx(min_variance_closed_form(cov), abs=1e-4)
    assert max_sharpe(mu, cov, 0.02, max_weight=1.0) == pytest.approx(
        tangency_closed_form(mu, cov, 0.02), abs=1e-3)


def test_long_only_weights_respect_bounds_and_sum():
    mu, cov = _random_cov()
    for w in (min_variance(cov, 0.3), max_sharpe(mu, cov, 0.02, 0.3)):
        assert w.sum() == pytest.approx(1.0)
        assert (w >= -1e-9).all()
        assert (w <= 0.3 + 1e-6).all()


def test_closed_form_can_have_negative_weights_but_long_only_cannot():
    cov = pd.DataFrame([[0.04, 0.057], [0.057, 0.09]], index=["A", "B"], columns=["A", "B"])
    w_closed = min_variance_closed_form(cov)
    w_long = min_variance(cov)
    assert w_closed.tolist() == pytest.approx([0.033 / 0.016, -0.017 / 0.016], rel=1e-6)
    assert w_closed.min() < 0
    assert w_long.min() >= -1e-9
    assert w_long.tolist() == pytest.approx([1.0, 0.0], abs=1e-4)


def test_infeasible_cap_raises():
    mu, cov = _uncorrelated()
    with pytest.raises(ValueError):
        min_variance(cov, max_weight=0.2)   # 3 Titel x 20 % = 60 % < 100 %


def test_portfolio_performance_known_values():
    mu, cov = _uncorrelated()
    ret, vol = portfolio_performance([0.5, 0.5, 0.0], mu, cov)
    assert ret == pytest.approx(0.065)
    assert vol == pytest.approx(np.sqrt(0.25 * 0.04 + 0.25 * 0.09))


def test_max_return_fills_best_assets_first():
    mu = pd.Series([0.02, 0.10, 0.06], index=NAMES)
    w = max_return(mu, max_weight=0.6)
    assert w.tolist() == pytest.approx([0.0, 0.6, 0.4])


def test_efficient_frontier_is_monotone_and_starts_at_min_variance():
    mu, cov = _random_cov()
    frontier = efficient_frontier(mu, cov, n_points=15, max_weight=0.4)
    assert len(frontier) >= 10
    assert (np.diff(frontier["vol"]) >= -1e-6).all()
    assert (np.diff(frontier["return"]) >= -1e-9).all()
    _, vol_min = portfolio_performance(min_variance(cov, 0.4), mu, cov)
    assert frontier["vol"].iloc[0] == pytest.approx(vol_min, rel=1e-4)
    assert all(abs(w.sum() - 1.0) < 1e-6 for w in frontier["weights"])


def test_shrink_covariance_limits_and_diagonal():
    _, cov = _random_cov()
    assert shrink_covariance(cov, 0.0).to_numpy() == pytest.approx(cov.to_numpy())
    full = shrink_covariance(cov, 1.0).to_numpy()
    assert full == pytest.approx(np.diag(np.diag(cov.to_numpy())))
    part = shrink_covariance(cov, 0.3).to_numpy()
    assert np.diag(part) == pytest.approx(np.diag(cov.to_numpy()))
    with pytest.raises(ValueError):
        shrink_covariance(cov, 1.5)


def test_shrinkage_improves_condition_number():
    _, cov = _random_cov()
    assert condition_number(shrink_covariance(cov, 0.3)) < condition_number(cov)


def test_condition_number_of_identity_is_one():
    assert condition_number(pd.DataFrame(np.eye(4))) == pytest.approx(1.0)


def test_random_weights_shape_sum_and_cap():
    w = random_weights(10, 500, max_weight=0.3, seed=1)
    assert w.shape == (500, 10)
    assert w.sum(axis=1) == pytest.approx(np.ones(500))
    assert w.max() <= 0.3 + 1e-12