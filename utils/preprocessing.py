"""
Data Preprocessing Pipeline Module
Handles patient input formatting, biological zero fixing, one-hot encoding alignment,
MinMax scaling for quantum AngleEmbedding, PCA compression, and skip-branch feature scaling.
"""

import joblib
import pandas as pd
import numpy as np
import torch
import streamlit as st
import config


@st.cache_resource(show_spinner="Loading Preprocessing Transformers...")
def load_transformers():
    """
    Loads all fitted scikit-learn transformers and column definitions from pickles.
    """
    try:
        angle_scaler = joblib.load(config.ANGLE_SCALER_PATH)
        pca = joblib.load(config.PCA_PATH)
        pca_scaler = joblib.load(config.PCA_SCALER_PATH)
        skip_scaler = joblib.load(config.SKIP_SCALER_PATH)
        skip_cols = joblib.load(config.SKIP_COLS_PATH)
        feature_cols = joblib.load(config.FEATURE_COLS_PATH)
        return {
            "angle_scaler": angle_scaler,
            "pca": pca,
            "pca_scaler": pca_scaler,
            "skip_scaler": skip_scaler,
            "skip_cols": skip_cols,
            "feature_cols": feature_cols
        }
    except Exception as e:
        st.error(f"Error loading preprocessing transformers: {e}")
        return None


def preprocess_patient_input(patient_dict: dict):
    """
    Transforms raw 11 patient features into quantum and skip branch tensors matching
    the exact training pipeline:
      Raw Dict -> DataFrame -> Zero Fixing -> One-Hot Alignment -> Angle Scale -> PCA -> PCA Scale -> Tensors
    """
    transformers = load_transformers()
    if transformers is None:
        raise RuntimeError("Transformers could not be loaded.")

    # 1. Convert input dictionary to pandas DataFrame
    df = pd.DataFrame([patient_dict])

    # 2. Fix biologically impossible zero values
    if "Cholesterol" in df.columns:
        df["Cholesterol"] = df["Cholesterol"].replace(0, 223.0)  # median cholesterol
    if "RestingBP" in df.columns:
        df["RestingBP"] = df["RestingBP"].replace(0, 130.0)      # median BP

    # 3. One-hot encode categorical features
    df_encoded = pd.get_dummies(df)

    # 4. Align with exact feature columns used during model training
    feature_cols = transformers["feature_cols"]
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0
    df_encoded = df_encoded[feature_cols]

    # 5. Quantum Branch Transformation
    # Scale to [0, pi] for AngleEmbedding
    x_sc = transformers["angle_scaler"].transform(df_encoded)
    # PCA compression to N_QUBITS (8 components)
    x_pca = transformers["pca"].transform(x_sc)
    # Scale PCA components to [-pi/2, pi/2]
    x_quantum = transformers["pca_scaler"].transform(x_pca)

    # 6. Skip Branch Transformation
    skip_cols = transformers["skip_cols"]
    x_skip_raw = df_encoded[skip_cols]
    x_skip = transformers["skip_scaler"].transform(x_skip_raw)

    # 7. Convert to PyTorch Tensors
    x_quantum_tensor = torch.tensor(x_quantum, dtype=torch.float32)
    x_skip_tensor = torch.tensor(x_skip, dtype=torch.float32)

    return x_quantum_tensor, x_skip_tensor, df_encoded
