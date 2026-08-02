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

# Top Gradient Header Banner
st.markdown("""
<div style="background: linear-gradient(135deg, #1A5276 0%, #C0392B 100%); border-radius: 10px; padding: 20px; text-align: center; color: #FFFFFF; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <h2 style="color: #FFFFFF !important; margin: 0 0 6px 0; font-size: 1.8rem; font-weight: 700;">❤️ HQ-KAN Heart Disease Prediction System</h2>
    <h4 style="color: #E2E8F0 !important; margin: 0 0 10px 0; font-size: 1.1rem; font-weight: 400;">Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty</h4>
    <p style="color: #CBD5E1 !important; margin: 0; font-size: 0.85rem; font-style: italic;">NRI Institute of Technology, Agiripalli | B.Tech Final Year Project 2025-26</p>
</div>
""", unsafe_allow_html=True)

# 3. Left Sidebar Status & Controls
with st.sidebar:
    st.markdown("## 🏥 HQ-KAN Clinical AI")
    st.caption("Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty")
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

# 4. Default Home Dashboard View
import importlib
page_home = importlib.import_module("pages.1_Home")
page_home.render_home()
