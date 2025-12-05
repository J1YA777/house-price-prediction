import argparse
import numpy as np
from src.data import load_data, get_X_y
from src.features import add_basic_features
from src.preprocess import build_preprocessor
from src.model import get_base_models, fit_and_compare, tune_xgb
from src.utils import save_artifact, ensure_dir
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train(args):
    ensure_dir(args.out)
    train_df, test_df = load_data(args.data_dir)

    # Feature engineering
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    X, y = get_X_y(train_df)
    y_log = np.log1p(y)  # stabilize variance

    preprocessor, large_card_cols = build_preprocessor(X)
    preprocessor.fit(X)

    # Apply target encoding for large-cardinality categorical columns
    if large_card_cols:
        te = TargetEncoder(cols=large_card_cols, smoothing=0.3)
        te.fit(X[large_card_cols], y_log)
        X_large_enc = te.transform(X[large_card_cols])
        X_for_model = X.drop(columns=large_card_cols)
        X_for_model[large_card_cols] = X_large_enc
    else:
        X_for_model = X.copy()

    X_trans = preprocessor.transform(X_for_model)

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_trans, y_log, test_size=0.15, random_state=42)

    models = get_base_models()
    scores = fit_and_compare(models, X_trans, y_log)

    if args.tune:
        best_params = tune_xgb(X_trans, y_log, timeout=args.tune_timeout)
        print("Best XGB params:", best_params)
        models["xgb"].set_params(**best_params)

    trained = {}
    for name, model in models.items():
        print(f"Fitting {name} on full training data...")
        model.fit(X_trans, y_log)
        trained[name] = model

    save_artifact(preprocessor, f"{args.out}/preprocessor.joblib")
    if large_card_cols:
        save_artifact(te, f"{args.out}/target_encoder.joblib")
    for name, model in trained.items():
        save_artifact(model, f"{args.out}/{name}_model.joblib")

    # Evaluate ensemble on validation set
    val_preds = [m.predict(X_val) for m in trained.values()]
    ensemble_pred = np.mean(val_preds, axis=0)
    rmse_val = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    print(f"Ensemble RMSE (log-target) on val set: {rmse_val:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-timeout", type=int, default=300)
    args = parser.parse_args()
    train(args)
