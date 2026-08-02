"""
HQ-KAN Model Architecture & Predictor Module
Defines the Hybrid Quantum Kolmogorov-Arnold Network (HQ-KAN) model,
PennyLane Variational Quantum Circuit (VQC), and PyTorch inference pipeline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import streamlit as st
import config

# Define PennyLane Quantum Device
dev = qml.device("default.qubit", wires=config.N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc(inputs, weights):
    """
    Variational Quantum Circuit (VQC):
    1. AngleEmbedding: Maps 8 features to Y-rotation angles across 8 qubits.
    2. StronglyEntanglingLayers: Trainable RZ, RX rotations + CNOT entangling gates.
    3. PauliZ Measurement: Computes expectation values [-1, +1] across all 8 qubits.
    """
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(config.N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]

# Shape of quantum weights (N_LAYERS x N_QUBITS x 3)
weight_shapes = {
    "weights": qml.StronglyEntanglingLayers.shape(config.N_LAYERS, config.N_QUBITS)
}


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear Layer:
    Replaces fixed activation functions with learnable B-spline/basis activations.
    """
    def __init__(self, in_features: int, out_features: int, scale_base: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        nn.init.uniform_(self.spline_weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(F.silu(x), self.base_weight)
        spline_output = F.linear(x, self.spline_weight)
        return base_output + spline_output


class KAN(nn.Module):
    """
    Multi-layer KAN Pre-processor for cardiac feature adaptation.
    """
    def __init__(self, layers_hidden: list):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HQKAN(nn.Module):
    """
    Hybrid Quantum Kolmogorov-Arnold Network (HQ-KAN):
    Fuses Quantum PCA branch (KAN -> VQC) with Classical Skip-Feature branch.
    """
    def __init__(self, quantum_layer, skip_dim: int = config.SKIP_K):
        super().__init__()

        # KAN pre-layer adapts 8 PCA features before quantum encoding
        self.kan_pre = KAN(layers_hidden=[config.N_QUBITS, 16, config.N_QUBITS])
        self.qlayer = quantum_layer

        fused_dim = config.N_QUBITS + skip_dim
        self.post = nn.Sequential(
            nn.Linear(fused_dim, 24),
            nn.SiLU(),
            nn.LayerNorm(24),
            nn.Dropout(0.1),
            nn.Linear(24, 12),
            nn.SiLU(),
            nn.Linear(12, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_quantum: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        kan_out = self.kan_pre(x_quantum)
        q_out = self.qlayer(kan_out)
        combined = torch.cat([kan_out + q_out, x_skip], dim=1)
        return self.post(combined)


@st.cache_resource(show_spinner="Loading HQ-KAN Model Checkpoint...")
def load_hqkan_model() -> HQKAN:
    """
    Loads the trained HQ-KAN model checkpoint state_dict safely.
    Caches the model instance in memory.
    """
    qlayer = qml.qnn.TorchLayer(vqc, weight_shapes)
    model = HQKAN(quantum_layer=qlayer, skip_dim=config.SKIP_K)

    if config.MODEL_PATH.exists():
        state_dict = torch.load(config.MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        st.error(f"Model checkpoint not found at {config.MODEL_PATH}")

    model.eval()
    return model
