"""
HQ-KAN Main Streamlit Application Entrypoint
Sets page configuration, injects custom medical CSS, renders the sidebar navigation,
and routes to application views.
"""

import streamlit as st
from pathlib import Path
import config

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="HQ-KAN Heart Disease Clinical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Medical AI CSS Styling
def load_css():
    css_path = config.BASE_DIR / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. Left Sidebar Navigation & Controls
with st.sidebar:
    st.markdown("## 🏥 HQ-KAN Clinical AI")
    st.caption("Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty")
    st.divider()

    st.markdown("### 🧭 Navigation")
    selected_page = st.radio(
        "Select Application View:",
        options=[
            "🏠 Home Dashboard",
            "🩺 Heart Disease Prediction",
            "📊 Model Performance",
            "💡 SHAP Explainability",
            "📚 About Project & Research"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    # Sidebar Quick Metric Badges
    st.markdown("### ⚡ Model Status")
    st.markdown("""
    - **Model Accuracy**: `92.5%`
    - **Quantum Circuit**: `8-Qubit VQC`
    - **Uncertainty**: `Bayesian MC (50)`
    - **Dataset Size**: `918 Patients`
    - **Status**: `🟢 Active & Ready`
    """)

    st.divider()

    st.markdown("""
    <div style="font-size: 0.78rem; color: #94A3B8; text-align: center;">
        B.Tech Final Year Project<br>
        <b>NRI Institute of Technology, Agiripalli</b><br>
        PennyLane + PyTorch + KAN
    </div>
    """, unsafe_allow_html=True)

import importlib

# 4. Page Routing Logic
if selected_page == "🏠 Home Dashboard":
    page_home = importlib.import_module("pages.1_Home")
    page_home.render_home()

elif selected_page == "🩺 Heart Disease Prediction":
    page_predict = importlib.import_module("pages.2_Predict")
    page_predict.render_predict()

elif selected_page == "📊 Model Performance":
    page_perf = importlib.import_module("pages.3_Model_Performance")
    page_perf.render_performance()

elif selected_page == "💡 SHAP Explainability":
    page_shap = importlib.import_module("pages.4_SHAP_Explainability")
    page_shap.render_shap_page()

elif selected_page == "📚 About Project & Research":
    page_about = importlib.import_module("pages.5_About_Project")
    page_about.render_about()
