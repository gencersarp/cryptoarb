from __future__ import annotations

import pandas as pd

from portfolio.analytics import compute_metrics


def test_sharpe_excess_uses_risk_free_rate():
    # flat positive drift with low vol so raw Sharpe is positive
    r = [0.0002] * 40 + [0.0] * 10
    eq = pd.Series((1 + pd.Series(r)).cumprod() * 100_000.0)
    m0 = compute_metrics(eq, 100_000.0, risk_free_rate_annual=0.0)
    mrf = compute_metrics(eq, 100_000.0, risk_free_rate_annual=0.10)

    assert m0["sharpe"] > 0
    assert m0["sharpe_excess"] == m0["sharpe"]
    assert mrf["ann_excess_return"] < mrf["ann_return"]
    assert mrf["sharpe_excess"] < mrf["sharpe"]
