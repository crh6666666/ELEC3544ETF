# Data-source addendum, recorded before scoring the 2026 ETF period

Yahoo is unavailable (429). Nasdaq's public historical endpoint returned OHLCV
without adjusted closes. Download a separate 2021-2026 snapshot, check its full
2021-2025 overlap with frozen Yahoo data, trading sessions and split units first.
Do not silently replace or rescale a discrepancy.

If overlap passes, preserve frozen 2021-2025 bars and append 2026 Nasdaq bars in
the same split-unit coordinate. Construct dividend-adjusted feature prices with
issuer distributions and the previous-close multiplier, matching the existing
adjustment convention. A constant overall normalization does not affect return
or price/MA features. All issuer events and normalization rules are recorded.
Primary accounting remains the payable-date cash-dividend ledger.

Fit, selection, models, parameter grids, Top5 rule, costs and evaluation cutoff
stay as in NEW_DATA_PROTOCOL.md. No 2026 label is used to select any choice.
Any source convention change is disclosed: this is a new-period test under a
documented data substitution, not an unchanged live trading track record.
