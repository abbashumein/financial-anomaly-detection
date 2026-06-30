import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from groq import Groq

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
    model.load_state_dict(torch.load("vae_model.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_results():
    return pd.read_csv("anomaly_results.csv")

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
    
    with st.spinner("Generating AI explanation..."):
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""Financial fraud analyst.
Company: {row.company}
Metric: {row.tag}
Anomaly score: {row.anomaly_score:.4f}
Risk: {row.risk_level}
In 2 sentences explain what is suspicious and what auditor should check."""}]
        )
        explanation = response.choices[0].message.content
    
    st.subheader("AI Explanation")
    st.write(explanation)
    
    if row.risk_level == "HIGH":
        st.error("HIGH RISK — Recommend immediate audit review")
    elif row.risk_level == "MEDIUM":
        st.warning("MEDIUM RISK — Monitor closely")
    else:
        st.success("LOW RISK — Normal variation")
