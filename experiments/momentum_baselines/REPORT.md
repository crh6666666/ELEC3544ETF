# 20/60-session momentum baseline comparison

Reproduce from the project root: `python3 run_momentum_comparison.py`.

This experiment adds momentum baselines, not new trained-model features. Scores
are Close(t)/Close(t-h)-1, h=5,20,60 trading sessions. All use identical signal
weeks, next-session-open entry, next-week-open exit, weekly Top 3 with weights
1/3, and final liquidation. Ticker ascending breaks ties. Costs are one-way
0/5/10/20 bps; main comparison below is 10 bps. Returns exclude cash dividends.
All periods have previously been inspected; this is retrospective exploration.
Windows are separate accounts and must not be presented as one continuous NAV.

| Window | Weeks | 5d net return | 20d net return | 60d net return | Equal weight |
|---|---:|---:|---:|---:|---:|
| W1: 2023 H2 | 24 | -5.34% | -2.39% | 5.49% | 5.68% |
| W2: 2024 H1 | 24 | -2.32% | 9.51% | 4.51% | 8.63% |
| W3: 2024 H2 | 25 | 6.69% | -3.50% | 0.41% | 5.25% |
| 2025 exploratory | 51 | -2.41% | -9.47% | -0.02% | 10.60% |

60d beats 5d in three of four windows on net returns, but not consistently on
Rank IC or drawdown and never beats equal weight in these four windows.
20d beats every A-D model in W2 at 10 bps, showing that the previous 5d-only
baseline overstated the models' relative advantage in that window. This is
not evidence that 20d or 60d is a reliable winner. No significance claim is made.
Do not choose the horizon independently for each evaluation window.

Verification: every momentum score recomputed using history truncated at its
signal; 5d scores reconciled to original features/saved baseline; all labels
recomputed from raw opens; all 60 original A-D and 5d portfolio scenarios
(3 windows x 5 methods x 4 costs) reproduced within 1e-10 tolerance.

Outputs: scores.csv, weekly_metrics.csv, ranking_summary.csv,
portfolio_summary.csv, experiment.json (configuration and input fingerprints).
