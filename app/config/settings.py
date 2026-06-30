import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = "financial-anomaly-detection"
    model_path: str = "models/vae_model.pt"
    faiss_index_path: str = "models/faiss_index"
    results_csv_path: str = "data/anomaly_results.csv"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
