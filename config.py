"""
HQ-KAN Web Application Configuration File
Contains global constants, paths, design tokens, feature definitions, and UI defaults.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Artifact & Resource Paths
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "best_hqkan.pt"
ANGLE_SCALER_PATH = MODELS_DIR / "angle_scaler.pkl"
PCA_PATH = MODELS_DIR / "pca.pkl"
PCA_SCALER_PATH = MODELS_DIR / "pca_scaler.pkl"
SKIP_SCALER_PATH = MODELS_DIR / "skip_scaler.pkl"
SKIP_COLS_PATH = MODELS_DIR / "skip_cols.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"
DATASET_PATH = DATA_DIR / "heart.csv"

# Model Hyperparameters
N_QUBITS = 8
N_LAYERS = 4
SKIP_K = 6
MC_SAMPLES = 50

# UI Theme Color Palette (Deep Blue & Medical Teal Theme)
THEME = {
    "primary_dark": "#0A2540",
    "primary": "#1A365D",
    "secondary": "#2B6CB0",
    "accent_teal": "#00A896",
    "accent_green": "#2ECC71",
    "accent_red": "#E74C3C",
    "accent_amber": "#F39C12",
    "bg_light": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text_dark": "#1E293B",
    "text_muted": "#64748B",
    "border_color": "#E2E8F0"
}

# Clinical Feature Definitions and Constraints
FEATURE_DEFS = {
    "Age": {
        "label": "Patient Age (Years)",
        "type": "slider",
        "min": 18,
        "max": 95,
        "default": 54,
        "help": "Age of the patient in years (Range: 18-95)."
    },
    "Sex": {
        "label": "Biological Sex",
        "type": "selectbox",
        "options": ["M", "F"],
        "labels": {"M": "Male", "F": "Female"},
        "default": "M",
        "help": "Patient biological sex."
    },
    "ChestPainType": {
        "label": "Chest Pain Type",
        "type": "selectbox",
        "options": ["ASY", "ATA", "NAP", "TA"],
        "labels": {
            "ASY": "Asymptomatic (ASY)",
            "ATA": "Atypical Angina (ATA)",
            "NAP": "Non-Anginal Pain (NAP)",
            "TA": "Typical Angina (TA)"
        },
        "default": "ASY",
        "help": "Type of chest pain reported by the patient."
    },
    "RestingBP": {
        "label": "Resting Blood Pressure (mm Hg)",
        "type": "slider",
        "min": 80,
        "max": 200,
        "default": 130,
        "help": "Resting blood pressure upon admission (Normal: < 120 mm Hg)."
    },
    "Cholesterol": {
        "label": "Serum Cholesterol (mg/dl)",
        "type": "slider",
        "min": 100,
        "max": 600,
        "default": 223,
        "help": "Serum cholesterol level (Desirable: < 200 mg/dl)."
    },
    "FastingBS": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "type": "selectbox",
        "options": [0, 1],
        "labels": {0: "No (<= 120 mg/dl)", 1: "Yes (> 120 mg/dl)"},
        "default": 0,
        "help": "Fasting blood sugar level greater than 120 mg/dl."
    },
    "RestingECG": {
        "label": "Resting ECG Results",
        "type": "selectbox",
        "options": ["Normal", "ST", "LVH"],
        "labels": {
            "Normal": "Normal",
            "ST": "ST-T Wave Abnormality",
            "LVH": "Left Ventricular Hypertrophy"
        },
        "default": "Normal",
        "help": "Resting electrocardiogram results."
    },
    "MaxHR": {
        "label": "Maximum Heart Rate Achieved (bpm)",
        "type": "slider",
        "min": 60,
        "max": 220,
        "default": 140,
        "help": "Maximum heart rate achieved during exercise test."
    },
    "ExerciseAngina": {
        "label": "Exercise Induced Angina",
        "type": "selectbox",
        "options": ["N", "Y"],
        "labels": {"N": "No", "Y": "Yes"},
        "default": "N",
        "help": "Exercise-induced chest pain."
    },
    "Oldpeak": {
        "label": "Oldpeak (ST Depression)",
        "type": "slider",
        "min": -2.5,
        "max": 6.2,
        "step": 0.1,
        "default": 1.0,
        "help": "ST depression induced by exercise relative to rest."
    },
    "ST_Slope": {
        "label": "ST Segment Slope",
        "type": "selectbox",
        "options": ["Flat", "Up", "Down"],
        "labels": {
            "Flat": "Flat Slope",
            "Up": "Upsloping",
            "Down": "Downsloping"
        },
        "default": "Flat",
        "help": "The slope of the peak exercise ST segment."
    }
}

# Preset Clinical Patient Profiles
PRESET_PATIENTS = {
    "Sample Healthy Patient": {
        "Age": 42,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 115,
        "Cholesterol": 195,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    },
    "Sample High-Risk Patient": {
        "Age": 63,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 286,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 108,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.6,
        "ST_Slope": "Flat"
    }
}
