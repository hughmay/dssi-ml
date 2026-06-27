import os
import joblib
import pandas as pd
import streamlit as st

# ── 1. FIXED: PLACED FIRST BEFORE ANY OTHER STREAMLIT ACTIONS ──────────────────
st.set_page_config(page_title="Iris Classifier", page_icon="🌿", layout="centered")

# ── Model & feature loading ───────────────────────────────────────────────────
MODEL_PATH = "models/iris_species_classification_model_v1.joblib"
FEATURES_PATH = "models/iris_species_classification_features_v1.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error(f"Missing files! Run your training pipeline first.\n"
                 f"Checked paths: '{MODEL_PATH}' and '{FEATURES_PATH}'")
        st.stop()

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


model, FEATURES = load_model()

# ── Species display config ────────────────────────────────────────────────────
SPECIES_CONFIG = {
    'Iris-setosa': {'emoji': '🌸', 'color': 'success', 'name': 'Setosa'},
    'Iris-versicolor': {'emoji': '🌼', 'color': 'info', 'name': 'Versicolor'},
    'Iris-virginica': {'emoji': '🌺', 'color': 'warning', 'name': 'Virginica'},
    0: {'emoji': '🌸', 'color': 'success', 'name': 'Setosa'},
    1: {'emoji': '🌼', 'color': 'info', 'name': 'Versicolor'},
    2: {'emoji': '🌺', 'color': 'warning', 'name': 'Virginica'},
}

# ── Input ranges derived from the Iris dataset ────────────────────────────────
FEATURE_CONFIG = {
    'SepalLengthCm': {'label': 'Sepal Length (cm)', 'min': 4.3, 'max': 7.9, 'default': 5.8},
    'SepalWidthCm': {'label': 'Sepal Width (cm)', 'min': 2.0, 'max': 4.4, 'default': 3.0},
    'PetalLengthCm': {'label': 'Petal Length (cm)', 'min': 1.0, 'max': 6.9, 'default': 3.7},
    'PetalWidthCm': {'label': 'Petal Width (cm)', 'min': 0.1, 'max': 2.5, 'default': 1.2},
}

# ── Session state ─────────────────────────────────────────────────────────────
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
def app_sidebar():
    st.sidebar.header('Iris Flower Measurements')

    inputs = {}
    for feat in FEATURES:
        cfg = FEATURE_CONFIG.get(feat, {'label': feat, 'min': 0.0, 'max': 10.0, 'default': 1.0})
        inputs[feat] = st.sidebar.number_input(
            label=cfg['label'],
            min_value=float(cfg['min']),
            max_value=float(cfg['max']),
            value=float(cfg['default']),
            step=0.1,
            format="%.1f",
        )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        predict_button = st.sidebar.button("Classify", key="predict", use_container_width=True)
    with col2:
        reset_button = st.sidebar.button("Reset", key="clear", use_container_width=True)

    if predict_button:
        st.session_state['input_features'] = inputs
    if reset_button:
        st.session_state['input_features'] = {}
        st.rerun()


# ── Main body ─────────────────────────────────────────────────────────────────
def app_body():
    st.title("🌿 Iris Species Classifier")
    st.write("Enter the flower measurements in the sidebar, then click **Classify**.")

    if not st.session_state['input_features']:
        st.info("Awaiting input — fill in the sidebar measurements and click **Classify**.")
        return

    # Build input DataFrame in the exact feature order the model expects
    input_df = pd.DataFrame([st.session_state['input_features']])[FEATURES]

    try:
        prediction = model.predict(input_df)[0]
        cfg = SPECIES_CONFIG.get(prediction, {'emoji': '🌿', 'color': 'success', 'name': str(prediction)})

        # Display primary prediction banner
        display_func = getattr(st, cfg['color'])
        display_func(f"**Predicted Species:** {cfg['emoji']} {cfg['name']}")

        # Confidence breakdown calculations
        if hasattr(model, "predict_proba"):
            st.subheader("Confidence Breakdown")
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            species_labels = [SPECIES_CONFIG.get(c, {'name': str(c)})['name'] for c in classes]

            proba_df = pd.DataFrame({
                'Species': species_labels,
                'Probability': [round(float(p) * 100, 1) for p in probabilities],
            }).sort_values('Probability', ascending=False)

            st.dataframe(
                proba_df.style.format({'Probability': '{:.1f}%'}),
                use_container_width=True,
                hide_index=True,
            )

        # Show raw values that generated the estimate
        st.subheader("Input Summary")
        display_df = pd.DataFrame([st.session_state['input_features']]).rename(
            columns={f: FEATURE_CONFIG.get(f, {'label': f})['label'] for f in FEATURES}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # FIXED: st.set_page_config removed from here entirely
    app_sidebar()
    app_body()


if __name__ == "__main__":
    main()
