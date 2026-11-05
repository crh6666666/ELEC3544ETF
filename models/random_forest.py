"""Random forest training, validation selection and parameter export."""
import json

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .common import FEATURES, FitResult

DEPTHS = (3, 5)
MIN_SAMPLES_LEAF = (10, 20)


def fit_model(train, validation, features=FEATURES):
    models, grid = [], []
    for depth in DEPTHS:
        for leaf in MIN_SAMPLES_LEAF:
            model = RandomForestRegressor(n_estimators=300, max_depth=depth,
                min_samples_leaf=leaf, random_state=42, n_jobs=1)
            model.fit(train[features], train.TargetPriceReturn)
            score = mean_squared_error(validation.TargetPriceReturn,
                                       model.predict(validation[features]))
            models.append(model)
            grid.append(dict(max_depth=depth, min_samples_leaf=leaf,
                             validation_mse=float(score)))
    best = min(range(len(grid)), key=lambda i: grid[i]['validation_mse'])
    return FitResult(models[best], grid, grid[best])


def save_outputs(result, features, out, metadata):
    from price_basis import CONTRACT
    from study_config import SPLIT_DESIGN

    model = result.model
    pd.DataFrame(result.grid).to_csv(out/'random_forest_validation_grid.csv', index=False)
    params = dict(model='RandomForestRegressor', selected=result.selected,
        parameters=model.get_params(), features=list(features), price_basis=CONTRACT,
        fit_partition='train_only_no_refit', experiment_status=metadata['experiment_status'],
        FitAsOfMeaning='last training label realization; selection availability recorded separately; not wall-clock fit time',
        split_design=SPLIT_DESIGN,
        feature_importances=dict(zip(features, model.feature_importances_.tolist())))
    (out/'random_forest_parameters.json').write_text(json.dumps(params, indent=2))


if __name__ == '__main__':
    from .common import run_debug
    run_debug('random_forest', fit_model, save_outputs)
