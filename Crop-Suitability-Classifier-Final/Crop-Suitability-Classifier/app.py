from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_suitability_model.joblib"

st.set_page_config(
    page_title="CropSuit — Crop Suitability Predictor",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 CropSuit")
st.subheader("Crop Suitability Classification")
st.write(
    "Enter soil and environmental conditions to predict the crop category "
    "associated with the learned training patterns."
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. Run "
        "`python src/train_model.py` first."
    )
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
label_encoder = artifact["label_encoder"]
features = artifact["features"]
model_name = artifact["model_name"]
metrics = artifact["metrics"]

st.sidebar.header("Model Information")
st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Test Accuracy:** {metrics['accuracy']:.2%}")
st.sidebar.write(f"**Macro F1:** {metrics['f1']:.2%}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌾 Soil Parameters")
    nitrogen = st.number_input(
        "Nitrogen (N)", min_value=0.0, max_value=140.0,
        value=90.0, step=1.0
    )
    phosphorus = st.number_input(
        "Phosphorus (P)", min_value=0.0, max_value=145.0,
        value=42.0, step=1.0
    )
    potassium = st.number_input(
        "Potassium (K)", min_value=0.0, max_value=205.0,
        value=43.0, step=1.0
    )
    ph = st.number_input(
        "Soil pH", min_value=0.0, max_value=14.0,
        value=6.5, step=0.1
    )

with col2:
    st.markdown("### 🌦️ Environmental Parameters")
    temperature = st.number_input(
        "Temperature (°C)", min_value=-10.0, max_value=60.0,
        value=25.0, step=0.1
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0,
        value=70.0, step=0.5
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=1000.0,
        value=100.0, step=1.0
    )

st.markdown("---")

if st.button("🌱 Predict Suitable Crop", type="primary", use_container_width=True):
    values = pd.DataFrame([{
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }], columns=features)

    prediction_encoded = model.predict(values)[0]
    prediction = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    st.success(f"### Recommended Crop: **{prediction.title()}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(values)[0]
        classes = label_encoder.inverse_transform(
            np.arange(len(probabilities))
        )

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_rows = [
            {
                "Crop": str(classes[idx]).title(),
                "Model probability": f"{probabilities[idx]:.2%}",
            }
            for idx in top_indices
        ]

        st.metric(
            "Prediction Confidence",
            f"{probabilities[prediction_encoded]:.2%}"
        )

        st.markdown("### Top 3 Model Predictions")
        st.dataframe(
            pd.DataFrame(top_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Input Summary")
    display_df = values.T.rename(columns={0: "Value"})
    st.dataframe(display_df, use_container_width=True)

    st.info(
        "This is an educational ML prototype. The prediction reflects "
        "patterns learned from the training dataset and should not be "
        "treated as a guaranteed agronomic recommendation."
    )

st.markdown("---")
st.caption(
    "CropSuit | Educational capstone prototype | "
    "Foundational supervised machine learning"
)
