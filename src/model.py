from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import optuna

def rmse_cv(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf)
    return scores.mean()

def get_base_models(random_state=42):
    rf = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=random_state, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                       colsample_bytree=0.8, random_state=random_state, verbosity=0)
    lgbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, random_state=random_state)
    return {"rf": rf, "xgb": xgb, "lgbm": lgbm}

def fit_and_compare(models: dict, X_train, y_train):
    scores = {}
    for name, model in models.items():
        print(f"Training {name}...")
        score = rmse_cv(model, X_train, y_train)
        scores[name] = score
        print(f"{name} CV RMSE: {score:.5f}")
    return scores

def ensemble_predictions(trained_models, X):
    preds = [m.predict(X) for m in trained_models]
    return np.mean(preds, axis=0)

def tune_xgb(X, y, timeout=300):
    def objective(trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
            "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
            "random_state": 42,
        }
        model = XGBRegressor(**param)
        score = rmse_cv(model, X, y, cv=4)
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout=timeout)
    return study.best_params

