# From return prediction to sector ranking: do better forecasts produce better portfolios?

Latest evidence: DATA_ACCESS_FIX.md and experiments/new_period_2026 document a
42-week 2026 evaluation under the previously frozen model rules and a disclosed
Nasdaq/issuer data substitution. Seven-feature Top5 returned 13.76%, nine-feature
6.62%, equal-weight11 11.37%, SPY 11.62%. All tested paired return intervals
include zero. This period is now viewed and must not be reused as an unseen
holdout. Do not hide the weaker nine-feature outcome. SPY daily risk statistics
have a flagged quote-quality caveat; full-period endpoint return is unaffected.

## Main question and three hypotheses

Can weekly market features rank 11 US sector ETFs well enough to improve a
cost-aware, diversified allocation? This is a retrospective comparison, not a
claim of profitable market timing. Hypotheses below organise existing evidence;
they are not a preregistration of already-viewed periods.

1. Objective: using identical features and selection Rank IC, does direct ranking
   improve on return regression? Primary comparison: rank_extended7 vs reg_extended7.
2. Features: do raw 20/60-session momentum features add stable information?
   Primary comparison: reg_extended9 vs reg_extended7 on the same eligible weeks.
3. Decision: does forecast ranking translate into net investment value?
   Primary allocation: Top5 equal weights (20% each), one-way 10bps, gross before
   tax; reference equal-weight11, SPY and the selection-period-chosen simple rule.

All candidates and negative windows stay visible. Top3, buffered allocation,
other costs, individual feature ablations and price-only returns are sensitivity
analyses, not extra opportunities to declare a winner. No further hyperparameter
search is justified by these observed evaluation results.

## What the assessment can see

| Assessment area | Work and reasoning | Evidence |
|---|---|---|
| Workload: sourcing | Yahoo access problems; frozen raw responses; independent issuer records | data/raw_2021_2025, data/issuer_evidence |
| Workload: preparation | Calendar, duplicate/missing checks, warm-up, split units, dividend dates | prepare_data.py, price_basis.py, issuer_dividends.py |
| Workload: iteration | Regression/ranking, feature additions/removals, fixed grids | RESEARCH_PROTOCOL.md, selection_grid.csv |
| Reasoning: baseline | Momentum, reversal, blended rule; select only before evaluation | run_research.py, ranking_summary.csv |
| Reasoning: refinement | Test whether ranking and portfolio outcomes agree; cash matching and costs | REFINEMENT_RESULTS.md, policy_effects.csv |
| Reasoning: limitations | No stable dominance, confidence intervals, previously inspected periods | bootstrap.csv, block_sensitivity.csv, NEW_DATA_PROTOCOL.md |
| Presentation | Three focused plots and two main tables | reports/focused_evidence |
| Report | Question -> data -> experiment -> failures -> practical limits | PROJECT_STORY.md, DATA_REPAIR_RESULTS.md |

## Suggested presentation sequence

1. Why sector rotation? Can forecasting help more than simple diversification?
2. Show the timing diagram and what is available at signal close.
3. Explain three controlled comparisons, including simple baselines.
4. Ranking-skill chart: improvements change across periods.
5. Account and cost charts: concentration and turnover affect outcomes.
6. What failed, what became more reliable, and what remains unverified.

## Claims and constraints

Safe claim: Top5 improved this nine-feature model's retrospective returns over
its Top3 counterpart in each observed window. Unsafe claim: the model reliably
beats the market. Positive 2025 results do not remove negative earlier windows.
Bootstrap intervals are exploratory, dependent on block length and not adjusted
for all multiple comparisons. Windows are separate accounts, never one continuous NAV.

The assessment document supplied by the user describes workload, reasoning,
presentation and report quality. It does not itself state that a foundation model
is mandatory, or that a changed proposal is automatically approved. Full course
requirements and instructor permission remain to be checked by the group.
Current XGBoost models are not foundation models. Do not relabel them.

## Target-definition audit

Across 2,838 eligible ETF-week labels, the mean absolute difference between the
adjusted-open proxy and gross issuer dividend-entitlement buy-and-hold return is
0.0245 basis points; the maximum is 3.30 basis points. See the full label audit in
reports/focused_evidence. Both definitions include unpaid entitlements in value;
neither is identical to a reinvested trading-account return. The observed small
average difference does not guarantee unchanged trees, ranks or holdings after
retraining. It supports retaining the proxy with an explicit sensitivity caveat,
rather than asserting that it is an exact cash-account training target.

Checkpoint contributions must be actual member work with evidence and honest
AI-assistance disclosure. No member names, hours, understanding or contributions
are inferred from these automated changes.
