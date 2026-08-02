"""
Streamlit Page 1 — Home Dashboard
Presents project overview, hero banner, core feature cards, live metrics, and technology stack.
"""

import streamlit as st
import config

def render_home():
    # Hero Banner Card
    st.markdown("""
    <div class="hero-card">
        <h1>🏥 HQ-KAN: Heart Disease Clinical AI</h1>
        <p>Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty Quantification & SHAP Explainability for Early Heart Disease Detection.</p>
        <div>
            <span class="badge-tag">⚛️ Quantum ML</span>
            <span class="badge-tag">🧠 KAN Network</span>
            <span class="badge-tag">🎲 Bayesian Uncertainty</span>
            <span class="badge-tag">💡 SHAP XAI</span>
            <span class="badge-tag">⚡ Fast Clinical Inference</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Top Key Metrics Highlights
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="🎯 Model Accuracy", value="88.0%", delta="Empirical Model")
    with m2:
        st.metric(label="📊 UCI Dataset Size", value="918 Patients", delta="5 Global Hospitals")
    with m3:
        st.metric(label="⚡ Inference Time", value="< 0.15s", delta="Real-time Prediction")
    with m4:
        st.metric(label="⚛️ Quantum Qubits", value="8 Qubits", delta="4 Layers (VQC)")

    st.divider()

    # Core Novel Feature Cards
    st.subheader("💡 Core Medical AI Capabilities")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="med-card">
            <h4>⚛️ Quantum Machine Learning (VQC)</h4>
            <p style="color: #64748B; font-size: 0.92rem;">
            Employs an 8-qubit Variational Quantum Circuit with <i>AngleEmbedding</i> and <i>StronglyEntanglingLayers</i> 
            to model non-linear cardiac feature interactions across quantum state spaces.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="med-card">
            <h4>💡 SHAP Model Explainability</h4>
            <p style="color: #64748B; font-size: 0.92rem;">
            Calculates exact Shapley additive feature contributions for every patient. 
            Physicians see precisely <b>WHY</b> a specific prediction was rendered rather than relying on a black-box model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="med-card">
            <h4>🎲 Bayesian Uncertainty Quantification</h4>
            <p style="color: #64748B; font-size: 0.92rem;">
            Uses Monte Carlo Dropout (50 stochastic forward passes) to calculate standard deviation and confidence score ranges. 
            Flags border-line or low-confidence predictions for additional testing.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="med-card">
            <h4>🩺 Clinical Decision Support System</h4>
            <p style="color: #64748B; font-size: 0.92rem;">
            Fast, non-invasive patient screening dashboard. Doctors enter 11 basic diagnostic measurements 
            and receive risk classification, confidence intervals, and downloadable PDF clinical reports.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Technology Stack Icons & Information
    st.subheader("🛠️ Technology Stack & Frameworks")
    
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1:
        st.info("**PyTorch**\n\nDeep Learning")
    with t2:
        st.info("**PennyLane**\n\nQuantum Simulator")
    with t3:
        st.info("**pyKAN**\n\nKAN Splines")
    with t4:
        st.info("**Web App**\n\nClinical Dashboard")
    with t5:
        st.info("**SHAP**\n\nModel Explainability")
    with t6:
        st.info("**Scikit-Learn**\n\nPreprocessing & PCA")

if __name__ == "__main__":
    render_home()
