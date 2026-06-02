"""Fair-premium pricing given a fitted RND.

For a contract spec (right, strike, expiry):
    fair_premium   = e^{-rT} · E_q[payoff(S_T)]
    edge_ratio     = market_premium / fair_premium
    expected_pnl   = E_q[payoff] - market_premium
    prob_itm       = Pr_q(in-the-money at expiry)
"""

from .fair import ContractSpec, FairValue, price_contract  # noqa: F401
