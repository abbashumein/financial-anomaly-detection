import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    edgar_user_agent: str = os.getenv("EDGAR_USER_AGENT", "")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = "financial-anomaly-detection"
    model_path: str = "models/vae_model.pt"
    faiss_index_path: str = "models/faiss_index"
    results_csv_path: str = "data/anomaly_results.csv"
    # If unset, the API runs without auth (fine for local dev). Set this
    # in production/demo deployments to require the X-API-Key header.
    api_key: str = os.getenv("API_KEY", "")
    # Groq deprecated llama-3.3-70b-versatile on 2026-06-17; this makes
    # future model swaps a one-line .env change instead of a code hunt.
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
