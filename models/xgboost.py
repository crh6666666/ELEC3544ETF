"""XGBoost training, validation selection and native model export."""
import json

import pandas as pd
import xgboost
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from .common import FEATURES, FitResult

DEPTHS = (2, 3)
TREE_COUNTS = (50, 100)


def fit_model(train, validation, features=FEATURES):
    models, grid = [], []
    for depth in DEPTHS:
        for trees in TREE_COUNTS:
            model = XGBRegressor(objective='reg:squarederror',
                n_estimators=trees, max_depth=depth, learning_rate=0.03,
                min_child_weight=10, reg_lambda=10, subsample=1.0,
                colsample_bytree=1.0, tree_method='hist', random_state=42, n_jobs=1)
            model.fit(train[features], train.TargetPriceReturn)
            score = mean_squared_error(validation.TargetPriceReturn,
                                       model.predict(validation[features]))
            models.append(model)
            grid.append(dict(max_depth=depth, n_estimators=trees,
                             validation_mse=float(score)))
    best = min(range(len(grid)), key=lambda i: grid[i]['validation_mse'])
    return FitResult(models[best], grid, grid[best])


def save_outputs(result, features, out, metadata):
    from price_basis import CONTRACT

    model = result.model
    pd.DataFrame(result.grid).to_csv(out/'xgboost_validation_grid.csv', index=False)
    model.save_model(out/'xgboost_model.ubj')
    params = dict(model='XGBRegressor', version=xgboost.__version__, selected=result.selected,
        parameters=model.get_params(), features=list(features), price_basis=CONTRACT,
        fit_partition='train_only_no_refit', experiment_status=metadata['experiment_status'],
        feature_importances=dict(zip(features, model.feature_importances_.tolist())))
    (out/'xgboost_parameters.json').write_text(json.dumps(params, indent=2))


if __name__ == '__main__':
    from .common import run_debug
    run_debug('xgboost', fit_model, save_outputs)
