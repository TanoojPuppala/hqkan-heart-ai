# 🏥 HQ-KAN: Hybrid Quantum Kolmogorov-Arnold Network for Heart Disease Prediction

Production-quality Streamlit Web Application implementing a **Hybrid Quantum Kolmogorov-Arnold Network (HQ-KAN)** with **Bayesian Uncertainty Quantification** and **SHAP Explainability** for early heart disease risk detection.

---

## 🌟 Primary Features

- **⚛️ Quantum Machine Learning (VQC)**: 8-Qubit Variational Quantum Circuit using PennyLane `AngleEmbedding` and `StronglyEntanglingLayers`.
- **🧠 KAN Spline Pre-Layer**: Learnable activation functions adapt non-linear cardiac feature relationships prior to quantum embedding.
- **🎲 Bayesian Uncertainty Quantification**: Monte Carlo (MC) Dropout (50 stochastic forward passes) calculates standard deviation and confidence intervals for clinical safety.
- **💡 SHAP Explainability**: SHAP KernelExplainer calculates exact Shapley feature attributions for every patient prediction.
- **🩺 Clinical Decision Support**: Interactive patient form with preset profiles, instant validation, PDF diagnostic reports, and CSV exports.
- **📊 Performance Dashboard**: Interactive Plotly confusion matrix, ROC curves, training loss progression, and 5-model ablation study.

---

## 📁 Production Directory Structure

```
hqkan_app/
├── app.py                     # Main Streamlit application entrypoint & page router
├── config.py                  # Global configurations, paths, and feature metadata
├── requirements.txt           # Python dependencies for local & cloud deployment
├── runtime.txt                # Python runtime version specification (python-3.10.11)
├── LICENSE                    # MIT License
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Production Git ignore rules
│
├── assets/
│   └── style.css              # Custom Deep Blue & Teal medical dashboard styling
│
├── data/
│   └── heart.csv              # UCI Combined 918 Patient Dataset
│
├── docs/
│   ├── HQ_KAN_FIXED_(1).ipynb # Original Colab Training Notebook
│   └── extracted_notebook_code.py # Extracted reference code
│
├── models/
│   ├── best_hqkan.pt          # Pre-trained PyTorch model weights checkpoint
│   ├── angle_scaler.pkl       # Fitted MinMaxScaler for AngleEmbedding [0, pi]
│   ├── pca.pkl                # Fitted PCA transformer (8 components)
│   ├── pca_scaler.pkl         # Fitted MinMaxScaler for PCA components [-pi/2, pi/2]
│   ├── skip_scaler.pkl        # Fitted MinMaxScaler for skip branch [-pi/2, pi/2]
│   ├── skip_cols.pkl          # Feature names for fusion skip branch
│   └── feature_cols.pkl       # All one-hot feature column names
│
├── outputs/
│   └── .gitkeep               # Directory for generated PDF and CSV exports
│
├── utils/
│   ├── predictor.py           # HQKAN PyTorch model & PennyLane VQC QNode
│   ├── preprocessing.py       # Zero fixing, one-hot encoding, PCA & MinMax scalers
│   ├── uncertainty.py         # Monte Carlo Dropout Bayesian uncertainty
│   ├── shap_utils.py          # SHAP KernelExplainer & patient contribution charts
│   ├── charts.py              # Interactive Plotly charts (Gauges, ROC, CM, Training)
│   └── report_generator.py    # Downloadable PDF clinical reports & CSV exports
│
└── pages/
    ├── 1_Home.py              # Hero banner, feature cards, & live metrics
    ├── 2_Predict.py           # Patient input form, risk engine, & PDF download
    ├── 3_Model_Performance.py # Performance metrics & 5-model ablation study
    ├── 4_SHAP_Explainability.py# SHAP summary & feature contribution breakdown
    └── 5_About_Project.py     # 5-Station architecture, dataset, & college credits
```

---

## 🚀 Running Locally

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud / GitHub / Render)

1. Push this project repository to **GitHub**:
   ```bash
   git remote add origin https://github.com/TanoojPuppala/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository.
4. Set the Main File Path to `app.py`.
5. Click **Deploy**!

---

## 🎓 Academic Credit & Developer Information

- **Developer**: Tanooj Puppala
- **Project**: B.Tech Final Year Engineering Project
- **Institution**: NRI Institute of Technology, Agiripalli
- **Title**: HQ-KAN: Hybrid Quantum Kolmogorov-Arnold Network with Bayesian Uncertainty Quantification for Early Heart Disease Prediction
- **License**: MIT License
