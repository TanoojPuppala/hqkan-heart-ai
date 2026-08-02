"""
Bayesian Uncertainty Quantification Module
Implements Monte Carlo (MC) Dropout to quantify prediction variance and clinical risk confidence.
"""

import torch
import numpy as np
import config


def predict_with_uncertainty(model, x_quantum: torch.Tensor, x_skip: torch.Tensor, n_samples: int = config.MC_SAMPLES):
    """
    Runs Monte Carlo Dropout inference across n_samples stochastic forward passes.
    
    Returns:
        dict containing:
            - mean_prob: Mean predicted disease probability [0.0, 1.0]
            - uncertainty_std: Standard deviation across MC samples
            - confidence_pct: Estimated confidence score percentage
            - risk_level: Categorical risk level ('Low', 'Moderate', 'High', 'Very High')
            - is_disease: Boolean indicating predicted disease status
            - sample_preds: Array of individual MC predictions for distribution plots
    """
    model.train()  # Activate dropout layers during inference for Monte Carlo estimation

    with torch.no_grad():
        preds_list = []
        for _ in range(n_samples):
            pred = model(x_quantum, x_skip)
            preds_list.append(pred.item())

    sample_preds = np.array(preds_list)
    mean_prob = float(np.mean(sample_preds))
    uncertainty_std = float(np.std(sample_preds))

    # Calculate confidence score percentage (100% - std percentage)
    confidence_pct = max(0.0, min(100.0, (1.0 - (uncertainty_std * 2.5)) * 100))

    # Categorize Risk Level based on probability and uncertainty margin
    if mean_prob < 0.35:
        risk_level = "Low Risk"
    elif mean_prob < 0.60:
        risk_level = "Moderate Risk"
    elif mean_prob < 0.82:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    is_disease = mean_prob > 0.50

    return {
        "mean_prob": mean_prob,
        "uncertainty_std": uncertainty_std,
        "confidence_pct": confidence_pct,
        "risk_level": risk_level,
        "is_disease": is_disease,
        "sample_preds": sample_preds
    }
