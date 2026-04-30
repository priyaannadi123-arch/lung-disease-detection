import streamlit as st
import numpy as np
import json
from PIL import Image
import os
import gdown

st.set_page_config(
    page_title="AI Lung Disease Detector",
    page_icon="🫁",
    layout="centered"
)

MODEL_PATH = "lung_disease_model_v2.h5"
GDRIVE_FILE_ID = "1BnLRraQgopC3gkxXtoJuKC0UpTb16MMd"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading AI model... please wait"):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=False
            )
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    with open("label_mapping.json", "r") as f:
        mapping = json.load(f)
    return model, mapping

model, LABEL_MAPPING = load_model()
RAW_LABELS = list(LABEL_MAPPING.keys())

HEALTH_TIPS = {
    "Adenocarcinoma": {
        "desc": "A type of non-small cell lung cancer found in mucus-secreting gland cells of the lung.",
        "tips": [
            "Consult an oncologist immediately for staging and treatment.",
            "Avoid smoking and second-hand smoke completely.",
            "Discuss targeted therapy and immunotherapy options with your doctor.",
            "Maintain an antioxidant-rich diet (fruits and vegetables).",
            "Regular follow-up CT scans are essential for monitoring progress."
        ]
    },
    "Large Cell Carcinoma": {
        "desc": "A fast-growing type of lung cancer that can appear in any part of the lung.",
        "tips": [
            "Seek immediate medical attention from a pulmonologist or oncologist.",
            "Stop smoking immediately — it significantly worsens outcomes.",
            "Ask your doctor about immunotherapy and targeted therapy options.",
            "Maintain regular exercise as tolerated to support lung capacity.",
            "Consider joining a lung cancer patient support group."
        ]
    },
    "Normal": {
        "desc": "No signs of lung disease detected in this CT scan.",
        "tips": [
            "Maintain a healthy lifestyle with regular physical exercise.",
            "Avoid smoking and prolonged exposure to air pollution.",
            "Get annual health checkups as a preventive measure.",
            "Eat a balanced diet rich in vitamins and minerals.",
            "Practice deep breathing exercises regularly for lung health."
        ]
    },
    "Squamous Cell Carcinoma": {
        "desc": "A type of lung cancer that typically starts in the bronchi in the centre of the lungs.",
        "tips": [
            "Immediate consultation with an oncologist is required.",
            "Quit smoking immediately — strongly linked to this cancer type.",
            "Ask about surgery, radiation therapy, or chemotherapy options.",
            "Monitor symptoms: persistent cough, blood in sputum, chest pain.",
            "Nutritional support and hydration are important during treatment."
        ]
    }
}

st.title("🫁 AI-Based Lung Disease Detection")
st.markdown("**Upload a chest CT scan image to detect lung disease using VGG19 Deep Learning**")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload CT Scan")
    uploaded_file = st.file_uploader(
        "Choose a CT scan image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest CT scan image for AI analysis"
    )

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded CT Scan", width=300)

    img = np.array(image.resize((224, 224))) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Analysing CT scan with AI..."):
        preds = model.predict(img)[0]
        idx = np.argmax(preds)
        raw_label = RAW_LABELS[idx]
        clean_label = LABEL_MAPPING[raw_label]
        confidence = preds[idx] * 100

    with col2:
        st.subheader("📊 Prediction Result")
        if clean_label == "Normal":
            st.success(f"✅ **{clean_label}**")
            st.info("No disease detected in this CT scan.")
        else:
            st.error(f"⚠️ **{clean_label} Detected**")
            st.warning("Please consult a doctor immediately.")

        st.metric("Confidence Score", f"{confidence:.2f}%")
        st.markdown("**Confidence per class:**")
        for i, raw in enumerate(RAW_LABELS):
            clean = LABEL_MAPPING[raw]
            st.progress(float(preds[i]),
                       text=f"{clean}: {preds[i]*100:.1f}%")

    st.markdown("---")
    st.subheader("🏥 Health Awareness & Recommendations")
    if clean_label in HEALTH_TIPS:
        t = HEALTH_TIPS[clean_label]
        if clean_label == "Normal":
            st.success(f"**Status:** {t['desc']}")
        else:
            st.error(f"**Detected Condition:** {t['desc']}")
        st.markdown("**Recommended Actions:**")
        for tip in t["tips"]:
            st.markdown(f"• {tip}")

    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Always consult a qualified doctor.")

else:
    with col2:
        st.info("👈 Upload a CT scan image to begin analysis.")
        st.markdown("""
        **This system detects:**
        - ✅ Normal (Healthy Lungs)
        - ⚠️ Adenocarcinoma
        - ⚠️ Large Cell Carcinoma
        - ⚠️ Squamous Cell Carcinoma
        """)

st.markdown("---")
st.caption("AI Lung Disease Detection | CSE B.Tech Final Year Project | VGG19 Transfer Learning")
