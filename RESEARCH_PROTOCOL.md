# ETF refinement protocol

This extension is a retrospective exploratory study. Every existing period has
already been inspected. No result may be called a fresh independent test.
The bundled data stop on 2025-12-31. A prospective independent evaluation needs
new data and a frozen protocol; it cannot be manufactured from these files.

Entry point: `python run_research.py --output experiments/refinement_v1`.
Use a new output directory for every run. Original experiments remain references.

## Fixed comparisons before this run

- Four chronological fit/selection/evaluation windows: original W1-W3 plus 2025.
- Identical eligible signal weeks and a common 60-session warm-up for all variants.
- Baselines: 5/20/60-session momentum, 5-session reversal, equal-weight average
  of cross-sectional 20/60-session momentum ranks. Select one using selection
  mean Rank IC only; preserve every baseline in output.
- Original4, extended7, extended9 (add raw 20/60-session momentum), individual
  removal of each of the three medium-term features, replace relative-SPY20
  with own20. Both regression and pairwise ranking use the same four-parameter
  grid and selection Rank IC. No evaluation-driven winner selection.
- Price returns and adjusted-open proxy returns are distinct parallel runs.
  Proxy Open = Open * AdjClose / Close; adjusted Close for features. This is a
  dividend-adjusted approximation, not a verified cash-dividend payment account.
  Existing factor/event audit runs first. Raw input data are never rewritten.
- Top3 (33.3% each), Top5 (20% each), buffered Top3 (retain holdings while rank
  <=5, fill vacancies by score) at 0/5/10/20 bps. This is rule-based industry
  allocation. Only Top5 satisfies the original 20% asset cap; no claim is made
  that Top3 inherits the old risk-constrained optimizer.
- Equal-weight and SPY references, plus positive-score portfolio and matched
  invested-weight equal portfolio for extended7 regression, at 10 bps.
- Within-window paired 4-week circular block bootstrap, 2000 repetitions,
  seed 42. Intervals are exploratory and not multiplicity adjusted. No claim
  of causation, reliable alpha, or a new independent validation.

## Metric and accounting conventions

Rank IC averages valid weekly Spearman correlations; constant predictions have
undefined IC. Top3 overlap uses ticker-ascending tie breaks. Daily open-marked
volatility/Sharpe use 252 sessions, risk-free rate zero. Terminal positions are
liquidated. Window accounts start independently at one and must not be spliced
into a continuous NAV. Weekly returns allocate rebalance fees at the exit open
to the preceding interval, preserving exact terminal NAV compounding.
Cash matching controls invested fraction, not market beta or sector risk.

## Outstanding boundaries

Independent-provider quote verification, actual dividend payment-date accounting,
new holdout data, instructor permission for proposal changes, and confirmation
of any separate foundation-model requirement are not established by this code.
Do not add LLM-generated features solely to claim foundation-model use.

Yahoo adjustment convention reference:
https://in.help.yahoo.com/kb/adjusted-close-sln28256.html
XGBoost ranking reference:
https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
