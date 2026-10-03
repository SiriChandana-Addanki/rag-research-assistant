"""Environment-backed configuration for the optional generation layer."""
from dataclasses import dataclass
import os


def _positive_int(name, default):
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _weight(name, default):
    value = float(os.getenv(name, default))
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class RAGConfig:
    gemini_api_key: str | None
    gemini_model: str
    top_k: int
    candidate_k: int
    semantic_weight: float
    context_limit: int
    timeout_seconds: float
    max_retries: int

    @classmethod
    def from_env(cls):
        timeout = float(os.getenv("RAG_TIMEOUT", "15"))
        if timeout <= 0:
            raise ValueError("RAG_TIMEOUT must be positive")
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            top_k=_positive_int("RAG_TOP_K", "5"),
            candidate_k=_positive_int("RAG_CANDIDATE_K", "10"),
            semantic_weight=_weight("RAG_SEMANTIC_WEIGHT", ".75"),
            context_limit=_positive_int("RAG_CONTEXT_LIMIT", "12000"),
            timeout_seconds=timeout,
            max_retries=max(0, int(os.getenv("RAG_MAX_RETRIES", "1"))),
        )
