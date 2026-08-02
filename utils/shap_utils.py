"""
SHAP Explainability Utility Module
Provides SHAP (SHapley Additive exPlanations) calculation for feature attribution
on the hybrid HQ-KAN model (combining quantum PCA components and clinical skip features).
"""

import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import config
from utils.predictor import load_hqkan_model
from utils.preprocessing import load_transformers


def get_feature_names():
    """Returns combined feature names: 8 Qubit PCs + 6 Clinical Skip Columns."""
    transformers = load_transformers()
    skip_cols = transformers["skip_cols"] if transformers else ["Skip1", "Skip2", "Skip3", "Skip4", "Skip5", "Skip6"]
    pc_names = [f"PC{i+1}" for i in range(config.N_QUBITS)]
    return pc_names + list(skip_cols)


def predict_fn_wrapper(x_numpy: np.ndarray) -> np.ndarray:
    """
    Model wrapper for SHAP KernelExplainer.
    Splits input matrix into Quantum PCA features (first 8 columns) and Skip features (last 6 columns).
    """
    model = load_hqkan_model()
    model.eval()

    with torch.no_grad():
        x_q = torch.tensor(x_numpy[:, :config.N_QUBITS], dtype=torch.float32)
        x_s = torch.tensor(x_numpy[:, config.N_QUBITS:], dtype=torch.float32)
        preds = model(x_q, x_s).cpu().numpy()
        return preds.reshape(-1, 1)


@st.cache_resource(show_spinner="Initializing SHAP Explainer...")
def get_shap_explainer():
    """
    Creates and caches a SHAP KernelExplainer initialized with lightweight background reference points.
    """
    feat_names = get_feature_names()
    # Create background baseline points
    n_features = len(feat_names)
    background_data = np.zeros((10, n_features), dtype=np.float32)
    explainer = shap.KernelExplainer(predict_fn_wrapper, background_data)
    return explainer


def calculate_patient_shap(x_quantum_tensor: torch.Tensor, x_skip_tensor: torch.Tensor):
    """
    Calculates SHAP values for a single patient input tensor.
    
    Returns:
        dict containing:
            - shap_values: array of SHAP attributions
            - feature_names: list of feature names
            - top_positive: top features pushing prediction towards heart disease
            - top_negative: top features pushing prediction towards healthy
    """
    explainer = get_shap_explainer()
    feat_names = get_feature_names()

    xq_np = x_quantum_tensor.detach().cpu().numpy()
    xs_np = x_skip_tensor.detach().cpu().numpy()
    x_patient = np.hstack([xq_np, xs_np])

    shap_vals = explainer.shap_values(x_patient, nsamples=100)

    # Format 1D SHAP values
    if isinstance(shap_vals, list):
        vals = shap_vals[0].flatten()
    else:
        vals = shap_vals.flatten()

    feature_impacts = list(zip(feat_names, vals, x_patient.flatten()))
    
    # Sort by positive and negative contributions
    positive_contribs = sorted([item for item in feature_impacts if item[1] > 0], key=lambda x: x[1], reverse=True)
    negative_contribs = sorted([item for item in feature_impacts if item[1] < 0], key=lambda x: x[1])

    return {
        "shap_values": vals,
        "feature_names": feat_names,
        "patient_features": x_patient.flatten(),
        "top_positive": positive_contribs[:5],
        "top_negative": negative_contribs[:5]
    }


def generate_shap_bar_fig(shap_values: np.ndarray, feature_names: list):
    """Generates a Matplotlib horizontal bar plot of SHAP feature contributions for a patient."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    y_pos = np.arange(len(feature_names))
    colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in shap_values]

    ax.barh(y_pos, shap_values, color=colors, align="center", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9, fontweight="bold")
    ax.invert_yaxis()  # top-down feature ordering
    ax.axvline(0, color="#64748B", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP Value (Impact on Heart Disease Risk)", fontsize=10, fontweight="bold")
    ax.set_title("Patient Feature Contribution Breakdown", fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig
