import joblib
import pandas as pd
import streamlit as st

# ── Model & feature loading ───────────────────────────────────────────────────
MODEL_PATH    = "iris_species_classification_model_v1.joblib"
FEATURES_PATH = "iris_species_classification_features_v1.joblib"

@st.cache_resource
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

model, FEATURES = load_model()

# ── Species display config ────────────────────────────────────────────────────
SPECIES_CONFIG = {
    'Iris-setosa':     {'emoji': '🌸', 'color': 'success'},
    'Iris-versicolor': {'emoji': '🌼', 'color': 'info'},
    'Iris-virginica':  {'emoji': '🌺', 'color': 'warning'},
}

# ── Input ranges derived from the Iris dataset ────────────────────────────────
FEATURE_CONFIG = {
    'SepalLengthCm': {'label': 'Sepal Length (cm)', 'min': 4.3, 'max': 7.9, 'default': 5.8},
    'SepalWidthCm':  {'label': 'Sepal Width (cm)',  'min': 2.0, 'max': 4.4, 'default': 3.0},
    'PetalLengthCm': {'label': 'Petal Length (cm)', 'min': 1.0, 'max': 6.9, 'default': 3.7},
    'PetalWidthCm':  {'label': 'Petal Width (cm)',  'min': 0.1, 'max': 2.5, 'default': 1.2},
}

# ── Session state ─────────────────────────────────────────────────────────────
if 'input_features' not in st.session_state:
    st.session_state['input_features'] = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
def app_sidebar():
    st.sidebar.header('Iris Flower Measurements')

    inputs = {}
    for feat in FEATURES:
        cfg = FEATURE_CONFIG[feat]
        inputs[feat] = st.sidebar.number_input(
            label   = cfg['label'],
            min_value = float(cfg['min']),
            max_value = float(cfg['max']),
            value   = float(cfg['default']),
            step    = 0.1,
            format  = "%.1f",
        )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        predict_button = st.sidebar.button("Classify", key="predict")
    with col2:
        reset_button = st.sidebar.button("Reset", key="clear")

    if predict_button:
        st.session_state['input_features'] = inputs
    if reset_button:
        st.session_state['input_features'] = {}

# ── Main body ─────────────────────────────────────────────────────────────────
def app_body():
    st.markdown(
        '<p style="font-family:arial,sans-serif;color:Black;font-size:40px;">'
        '<b>🌿 Iris Species Classifier</b></p>',
        unsafe_allow_html=True,
    )
    st.write("Enter the flower measurements in the sidebar, then click **Classify**.")

    if not st.session_state['input_features']:
        st.info("Awaiting input — fill in the sidebar and click **Classify**.")
        return

    # Build input DataFrame in the exact feature order the model was trained on
    input_df = pd.DataFrame([st.session_state['input_features']])[FEATURES]

    try:
        prediction   = model.predict(input_df)[0]          # e.g. 'Iris-setosa'
        probabilities = model.predict_proba(input_df)[0]   # array of 3 floats
        classes       = model.classes_

        cfg = SPECIES_CONFIG.get(prediction, {'emoji': '🌿', 'color': 'success'})
        getattr(st, cfg['color'])(
            f"**Predicted Species:** {cfg['emoji']} {prediction.replace('Iris-', '')}"
        )

        # Confidence breakdown
        st.subheader("Confidence Breakdown")
        proba_df = pd.DataFrame({
            'Species':     [c.replace('Iris-', '') for c in classes],
            'Probability': [round(float(p) * 100, 1) for p in probabilities],
        }).sort_values('Probability', ascending=False)

        st.dataframe(
            proba_df.style.format({'Probability': '{:.1f}%'}),
            use_container_width=True,
            hide_index=True,
        )

        # Show the values that were classified
        st.subheader("Input Summary")
        display_df = pd.DataFrame([st.session_state['input_features']]).rename(
            columns={f: FEATURE_CONFIG[f]['label'] for f in FEATURES}
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app_sidebar()
    app_body()

if __name__ == "__main__":
    main()
