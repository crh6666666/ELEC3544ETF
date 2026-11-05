"""Predeclared 2021--2025 study design; never selected from test returns."""
START = '2021-01-01'
END_EXCLUSIVE = '2026-01-01'
TRAIN_YEARS = (2021, 2022, 2023)
VALIDATION_YEARS = (2024,)
TEST_YEARS = (2025,)
SPLIT_DESIGN = {'method':'signal_calendar_year_and_labels_realized_within_partition_years',
                'train':list(TRAIN_YEARS),'validation':list(VALIDATION_YEARS),
                'test':list(TEST_YEARS),
                'status':'retrospective_exploratory; dates fixed before expanded model evaluation'}


def split_and_purge(frame):
    """Keep source index; require all prior-partition labels before next signal."""
    result = frame.copy()
    result['Split'] = 'excluded'
    for name, years in [('train',TRAIN_YEARS),('validation',VALIDATION_YEARS),('test',TEST_YEARS)]:
        result.loc[result.SignalDate.dt.year.isin(years),'Split'] = name
    purged = []
    for name, later in [('train','validation'),('validation','test')]:
        if not (result.Split == later).any():
            raise ValueError('No samples in '+later)
        cutoff = result.loc[result.Split == later,'SignalCloseUTC'].min()
        last_year = max(TRAIN_YEARS if name == 'train' else VALIDATION_YEARS)
        bad = (result.Split == name) & ((result.ExitOpenUTC >= cutoff) |
                                        (result.ExitDate.dt.year > last_year))
        purged.extend(result.loc[bad,'SignalDate'].astype(str).unique())
        result.loc[bad,'Split'] = 'purged_unrealized_label'
    return result, purged
