import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json

# Load trained GRU model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sepsis_gru_model.h5")

model = load_model()

# Load best threshold
with open("best_threshold.json", "r") as f:
    best_threshold = json.load(f)["threshold"]

st.title("🧪 Sepsis Detection Web App")
st.markdown("Upload hourly patient data (12 rows per patient, 41 features) to predict sepsis.")

uploaded_file = st.file_uploader("Upload patients_hourly.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    if "Patient_ID" not in df.columns:
        st.error("CSV must contain a 'Patient_ID' column.")
    else:
        results = []
        skipped = []
        for pid, group in df.groupby("Patient_ID"):
            if len(group) == 12:
                X = group.drop(columns=["Patient_ID"]).to_numpy().reshape(1, 12, -1)
                prob = model.predict(X, verbose=0)[0][0]
                label = "Sepsis Detected" if prob >= best_threshold else "No Sepsis"
                results.append({"Patient_ID": pid, "Probability": float(prob), "Status": label})
            else:
                skipped.append(pid)

        if results:
            res_df = pd.DataFrame(results)
            st.subheader("Prediction Results")
            st.dataframe(res_df)

            csv = res_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", data=csv, file_name="sepsis_predictions.csv", mime="text/csv")

        if skipped:
            st.warning(f"Skipped patients (not exactly 12 rows): {skipped}")
