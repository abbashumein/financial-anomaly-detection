# app/services/vae_scorer.py
"""
Loads the trained VAE once and scores live sequences with it.

Risk thresholds below are not arbitrary — they're the actual percentiles
of reconstruction error computed across the 285,275 real sequences the
model was trained on (see data/anomaly_results.csv). p95 = 0.10522487 is
the exact threshold the training notebook used to flag anomalies.
"""
import threading
import numpy as np
import torch

from app.models.vae import VAE

# Calibrated from data/anomaly_results.csv (n=285,275), computed once:
# p50=0.0438, p90=0.0867, p95=0.1052, p99=0.1548, max=0.6790
P50 = 0.043787673
P90 = 0.0867166
P95 = 0.10522487

_model = None
_lock = threading.Lock()


def _get_model(model_path: str = "models/vae_model.pt") -> VAE:
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # double-checked locking
                m = VAE(seq_len=20, latent_dim=10)
                state_dict = torch.load(model_path, map_location="cpu")
                m.load_state_dict(state_dict)
                m.eval()
                _model = m
    return _model


def risk_level_for(error: float) -> str:
    if error > P95:
        return "HIGH"
    if error > P90:
        return "MEDIUM"
    return "LOW"


def score_sequence(padded_sequence: np.ndarray, model_path: str = "models/vae_model.pt") -> dict:
    """
    Runs one real forward pass through the VAE and returns the
    reconstruction error plus how it compares to the training distribution.
    """
    model = _get_model(model_path)
    x = torch.tensor(padded_sequence, dtype=torch.float32).unsqueeze(0)  # (1, 20)
    with torch.no_grad():
        recon, mu, logvar = model(x)
        error = torch.mean((recon - x) ** 2, dim=1).item()

    return {
        "reconstruction_error": error,
        "risk_level": risk_level_for(error),
        "percentile_context": {
            "p50_typical": P50,
            "p90": P90,
            "p95_anomaly_threshold": P95,
        },
        "is_anomaly": error > P95,
    }
