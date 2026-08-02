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
