"""Ridge training, validation selection and parameter export."""
import json

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import FEATURES, FitResult

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def fit_model(train, validation, features=FEATURES):
    """Fit the scaler and candidates on train only; select by validation MSE."""
    models, grid = [], []
    for alpha in ALPHAS:
        model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
        model.fit(train[features], train.TargetPriceReturn)
        score = mean_squared_error(validation.TargetPriceReturn,
                                   model.predict(validation[features]))
        models.append(model)
        grid.append(dict(alpha=alpha, validation_mse=float(score)))
    best = min(range(len(grid)), key=lambda i: (grid[i]['validation_mse'], -grid[i]['alpha']))
    return FitResult(models[best], grid, grid[best])


def save_outputs(result, features, out, metadata):
    from price_basis import CONTRACT

    model = result.model
    pd.DataFrame(result.grid).to_csv(out/'validation_grid.csv', index=False)
    params = dict(features=list(features), alpha=result.selected['alpha'],
                  scaler_mean=model['scale'].mean_.tolist(),
                  scaler_scale=model['scale'].scale_.tolist(),
                  coef=model['ridge'].coef_.tolist(),
                  intercept=float(model['ridge'].intercept_),
                  fit_partition='train_only_no_refit', price_basis=CONTRACT,
                  price_basis_audit_id=metadata['price_basis_audit_id'])
    (out/'model_parameters.json').write_text(json.dumps(params, indent=2))


if __name__ == '__main__':
    from .common import run_debug
    run_debug('ridge', fit_model, save_outputs)
