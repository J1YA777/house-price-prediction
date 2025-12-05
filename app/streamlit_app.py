import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_artifact
from src.features import add_basic_features

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file with test data", type="csv")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(test_df.head())

    # Feature engineering
    test_df = add_basic_features(test_df)

    # Load preprocessor
    preprocessor = load_artifact("models/preprocessor.joblib")
    try:
        te = load_artifact("models/target_encoder.joblib")
        test_df[te.cols] = te.transform(test_df[te.cols])
    except FileNotFoundError:
        te = None

    X_trans = preprocessor.transform(test_df)

    # Load models
    import glob
    import joblib
    model_paths = glob.glob("models/*_model.joblib")
    models = [joblib.load(p) for p in model_paths]

    if st.button("Predict House Prices"):
        preds = [m.predict(X_trans) for m in models]
        preds_avg = np.mean(preds, axis=0)
        preds_final = np.expm1(preds_avg)  # inverse log

        test_df["Predicted_SalePrice"] = preds_final
        st.success("Prediction Completed!")
        st.dataframe(test_df[["Id", "Predicted_SalePrice"]].head())

        # Download option
        csv = test_df[["Id", "Predicted_SalePrice"]].to_csv(index=False)
        st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

