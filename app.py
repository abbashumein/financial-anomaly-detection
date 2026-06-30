import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class VAE(nn.Module):
    def __init__(self, seq_len=20, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(seq_len, 32), nn.ReLU(), nn.Linear(32, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, seq_len), nn.Sigmoid())
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

@st.cache_resource
def load_model():
    model = VAE()
    model.load_state_dict(torch.load("models/vae_model.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_results():
    try:
        return pd.read_csv("data/anomaly_results.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["company", "tag", "anomaly_score", "risk_level", "is_anomaly", "explanation"])

st.title("Financial Anomaly Detection System")
st.markdown("AI-powered fraud detection on real SEC EDGAR filings")

model = load_model()
results = load_results()

st.subheader("Top Flagged Companies")
top = results[results["is_anomaly"] == True].nlargest(20, "anomaly_score")
st.dataframe(top[["company", "tag", "anomaly_score", "risk_level"]])

st.subheader("Investigate a Company")
company = st.selectbox("Select flagged company", top["company"].unique())

if st.button("Analyze"):
    row = top[top["company"] == company].iloc[0]
    st.metric("Anomaly Score", f"{row.anomaly_score:.4f}")
    st.metric("Risk Level", row.risk_level)

    with st.spinner("Running LangGraph agent..."):
        from app.services.rag_agent import analyze_company
        result = analyze_company(row.company, row.tag, row.anomaly_score)
        explanation = result.get("final_report", result.get("explanation", "No explanation generated"))

    st.subheader("AI Explanation")
    st.write(explanation)

    if row.risk_level == "HIGH":
        st.error("HIGH RISK — Recommend immediate audit review")
    elif row.risk_level == "MEDIUM":
        st.warning("MEDIUM RISK — Monitor closely")
    else:
        st.success("LOW RISK — Normal variation")
