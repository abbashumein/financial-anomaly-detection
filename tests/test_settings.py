"""A blank GROQ_MODEL= in .env must fall back to the default model."""
from app.config.settings import Settings


def test_blank_groq_model_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "")
    s = Settings(_env_file=None)
    assert s.groq_model == "openai/gpt-oss-120b"


def test_explicit_groq_model_is_respected(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    s = Settings(_env_file=None)
    assert s.groq_model == "some/other-model"
