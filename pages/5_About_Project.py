"""
Streamlit Page 5 — About Project & Research Architecture
Documents clinical problem statement, UCI 918 dataset breakdown, 5-station HQ-KAN pipeline,
novelty claims table, academic credits, and guide information.
"""

import streamlit as st
import pandas as pd
import config

def render_about():
    st.title("📚 About HQ-KAN Project & Research Documentation")
    st.caption("Comprehensive documentation of architecture, dataset composition, novelty comparisons, and academic research credits.")

    # Hero Summary Box
    st.markdown("""
    <div class="hero-card">
        <h2>HQ-KAN: Hybrid Quantum Kolmogorov-Arnold Network</h2>
        <p>A B.Tech Final Year Engineering Project developing an end-to-end Quantum Machine Learning Decision Support System with Bayesian Uncertainty Quantification and SHAP Explainability for early heart disease detection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Problem Statement & Objectives
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="med-card">
            <h4>🎯 Problem Statement</h4>
            <p style="color: #64748B; font-size: 0.93rem;">
            Cardiovascular diseases remain the leading cause of mortality globally. Traditional diagnosis relies on clinical risk calculators 
            or classical ML models that treat patient measurements as flat independent variables, missing complex non-linear feature interactions.
            Furthermore, classical black-box models lack <b>uncertainty estimation</b> and <b>explainability</b>, preventing clinical adoption.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="med-card">
            <h4>🚀 Project Objectives</h4>
            <p style="color: #64748B; font-size: 0.93rem;">
            1. Formulate a single-channel <b>HQ-KAN</b> architecture combining KAN spline pre-layers with 8-qubit Variational Quantum Circuits.<br>
            2. Eliminate data leakage and recover PCA compression signal loss using a <b>classical feature-fusion skip branch</b>.<br>
            3. Provide <b>Bayesian MC Dropout</b> uncertainty quantification and <b>SHAP</b> explainability in a live Streamlit application.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Dataset Breakdown
    st.subheader("📊 UCI Combined Heart Disease Dataset (918 Patients)")
    st.write("Merged from 5 independent global hospital datasets to ensure high diversity and eliminate single-center bias:")

    ds_data = [
        {"Hospital / Source Dataset": "Cleveland Clinic Foundation", "Country": "USA", "Rows Contributed": "303 rows"},
        {"Hospital / Source Dataset": "Hungarian Institute of Cardiology", "Country": "Hungary", "Rows Contributed": "294 rows"},
        {"Hospital / Source Dataset": "Statlog Heart Dataset", "Country": "UK", "Rows Contributed": "270 rows"},
        {"Hospital / Source Dataset": "VA Long Beach Medical Center", "Country": "USA", "Rows Contributed": "200 rows"},
        {"Hospital / Source Dataset": "University Hospital Zurich", "Country": "Switzerland", "Rows Contributed": "123 rows"},
        {"Hospital / Source Dataset": "TOTAL (After removing 272 duplicate rows)", "Country": "GLOBAL COMBINED", "Rows Contributed": "918 FINAL ROWS"}
    ]
    st.dataframe(pd.DataFrame(ds_data), use_container_width=True)

    st.divider()

    # 5-Station HQ-KAN Pipeline Architecture
    st.subheader("⚙️ 5-Station HQ-KAN Pipeline Architecture")
    
    st.markdown("""
    1. **Station 1: Data Preparation & Leak-Free Scaling**:
       - Fixes biologically impossible zero values in Cholesterol and RestingBP using medians.
       - One-hot encodes categorical variables (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope).
       - Scales features to $[0, \pi]$ using `MinMaxScaler` for quantum `AngleEmbedding`.
       - PCA reduces features to 8 components matching the 8-qubit quantum circuit.

    2. **Station 2: KAN Pre-Layer (First Novelty)**:
       - Replaces fixed MLP activation functions (ReLU/Tanh) with learnable B-spline activation functions `KAN(width=[8, 16, 8])`.
       - Adapts non-linear cardiac risk accelerations before quantum embedding.

    3. **Station 3: Variational Quantum Circuit Core**:
       - `AngleEmbedding` maps 8 KAN outputs to Y-rotation angles on 8 qubits.
       - `StronglyEntanglingLayers` applies trainable RZ, RX rotations and CNOT entangling gates across all 8 qubits.
       - Measures PauliZ expectation values across all 8 qubits ($[-1, +1]$).

    4. **Station 4: Classical Post-Layer & Feature Fusion**:
       - Fuses the quantum PCA output ($kan\_out + q\_out$) with the top 6 clinical skip features (`skip_cols`).
       - Feeds combined vector into post-classifier `Linear(14->24) -> SiLU -> LayerNorm -> Linear(24->12) -> Linear(12->1) -> Sigmoid`.

    5. **Station 5: Bayesian Uncertainty & SHAP Explainability (Second & Third Novelties)**:
       - Runs 50 Monte Carlo Dropout stochastic passes during inference to compute mean probability and standard deviation.
       - Computes SHAP Shapley values for individual patient feature contribution charts.
    """)

    st.divider()

    # Novelty Comparison Table
    st.subheader("🏆 Research Literature Comparison & Novelties")

    comp_data = [
        {"Feature": "Dataset Size", "Verdone 2026": "303 rows", "Kumar 2025": "918 rows", "KACQ-DCNN 2024": "Varies", "YOUR PROJECT (HQ-KAN)": "918 rows (UCI Combined)"},
        {"Feature": "Pre-layer Architecture", "Verdone 2026": "Fixed MLP", "Kumar 2025": "No VQC", "KACQ-DCNN 2024": "Dual-channel KAN", "YOUR PROJECT (HQ-KAN)": "KAN Single-Channel (Simpler)"},
        {"Feature": "Dimension Reduction", "Verdone 2026": "Autoencoder (700 ep)", "Kumar 2025": "QGA Feature Select", "KACQ-DCNN 2024": "DenseKAN", "YOUR PROJECT (HQ-KAN)": "PCA + Skip Branch (Instant)"},
        {"Feature": "Qubits Measured", "Verdone 2026": "2 of 9 (22%)", "Kumar 2025": "QSVM Kernel", "KACQ-DCNN 2024": "4 of 4", "YOUR PROJECT (HQ-KAN)": "All 8 of 8 Qubits"},
        {"Feature": "Bayesian Uncertainty", "Verdone 2026": "None", "Kumar 2025": "None", "KACQ-DCNN 2024": "None", "YOUR PROJECT (HQ-KAN)": "Bayesian MC Dropout (50 passes)"},
        {"Feature": "Explainability", "Verdone 2026": "None", "Kumar 2025": "None", "KACQ-DCNN 2024": "SHAP + LIME", "YOUR PROJECT (HQ-KAN)": "SHAP KernelExplainer"},
        {"Feature": "Web Application", "Verdone 2026": "None", "Kumar 2025": "None", "KACQ-DCNN 2024": "None", "YOUR PROJECT (HQ-KAN)": "Streamlit Production Web App"},
        {"Feature": "Accuracy", "Verdone 2026": "90.98%", "Kumar 2025": "97.83%", "KACQ-DCNN 2024": "92.03%", "YOUR PROJECT (HQ-KAN)": "92.5% Target Accuracy"}
    ]

    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.divider()

    # Academic & Institution Information
    st.subheader("🎓 Academic Institution & Project Credits")

    st.info("""
    **B.Tech Final Year Engineering Project**
    - **Institution**: NRI Institute of Technology, Agiripalli
    - **Department**: Department of Computer Science & Engineering / Artificial Intelligence
    - **Project Title**: HQ-KAN: Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty Quantification for Early Heart Disease Prediction
    - **Frameworks Used**: PennyLane Quantum Computing Framework, PyTorch Deep Learning Library, pyKAN, Streamlit, Scikit-Learn.
    """)

if __name__ == "__main__":
    render_about()
