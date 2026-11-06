# Reserved new-period evaluation: freeze before scoring

Status: protocol reserved, no 2026 ETF performance inspected or scored in this
workspace. This does not establish that other group members have not seen it.

Candidate evaluation range: first complete weekly holding period in 2026 through
2026-10-30, only exits on or before the cutoff. No unfinished labels.
This is unseen by this pipeline, not a live prospective trading trial.

Frozen primary models: XGBoost regression, extended7 and extended9 as defined in
RESEARCH_PROTOCOL.md. Fit 2021-2024 (with purged boundary labels), select only
on 2025 using mean weekly Rank IC, depth 2/3 x trees 50/100, seed42, all other
parameters unchanged. All methods share valid 60-session warm-up samples.
No 2026 labels may enter fitting, selection, feature choice or allocation choice.

Primary allocation: weekly Top5 equal weights (20% per ETF), no leverage or
short selling, single-side 10bps, zero interest, gross before tax. ETF universe
unchanged. References: equal-weight11, SPY, selection-period-chosen baseline
from 5/20/60-day momentum, 5-day reversal and mean 20/60 momentum ranks.
Primary accounting: source split-adjusted opens plus issuer-confirmed dividends,
ex-date receivables and first-open-after-pay-date cash availability. Adjusted-open
proxy is a sensitivity check. Do not mix data snapshots across split units.

Report all primary candidates, daily NAV, drawdown, turnover, fees, yearly and
weekly paired differences, block-bootstrap 2/4/8-week intervals. Evaluate once.
No method is promoted solely for winning this window. Freeze modifications and
hash input snapshots before running; retain failed checks and excluded rows.

Data access currently unresolved: Yahoo chart probe returned 429, Stooq CSV probe
returned 404 on 2026-11-06. Never fill new periods synthetically or silently reuse
old quotes. Issuer distributions are available but do not supply daily OHLC.
2026 training features also need aligned SPY/ETF prices and split events.
