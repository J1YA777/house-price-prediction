import argparse
import numpy as np
import pandas as pd
from src.data import load_data
from src.features import add_basic_features
from src.utils import load_artifact
import glob
import joblib

def predict(args):
    _, test = load_data(args.data_dir)
    test = add_basic_features(test)

    preprocessor = load_artifact(f"{args.model_dir}/preprocessor.joblib")

    # Load target encoder if exists
    try:
        te = load_artifact(f"{args.model_dir}/target_encoder.joblib")
    except FileNotFoundError:
        te = None

    X_test = test.copy()

    if te is not None:
        X_test[te.cols] = te.transform(X_test[te.cols])

    X_test_trans = preprocessor.transform(X_test)

    # Load all model files
    model_paths = glob.glob(f"{args.model_dir}/*_model.joblib")
    models = [joblib.load(p) for p in model_paths]

    preds = [m.predict(X_test_trans) for m in models]
    preds_avg = np.mean(preds, axis=0)
    preds_final = np.expm1(preds_avg)  # inverse log

    submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": preds_final
    })

    submission.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()
    predict(args)
