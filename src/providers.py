"""Provider implementations. Imports of optional SDKs remain lazy."""
from __future__ import annotations
from dataclasses import dataclass
import json, os
from typing import Any


class ProviderError(RuntimeError): pass
class TransientProviderError(ProviderError): pass
class RateLimitError(TransientProviderError): pass


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    token_usage: dict[str, int] | None = None
    model: str | None = None


def _usage(response: Any):
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None: return None
    aliases = {"prompt_tokens": "prompt_token_count", "completion_tokens": "candidates_token_count", "total_tokens": "total_token_count"}
    values = {key: getattr(metadata, attribute, None) for key, attribute in aliases.items()}
    return {key: int(value) for key, value in values.items() if value is not None} or None

def estimate_cost(usage):
    """Return a USD estimate only when both usage and explicit local prices exist."""
    input_price, output_price = os.getenv("RAG_INPUT_COST_PER_MILLION"), os.getenv("RAG_OUTPUT_COST_PER_MILLION")
    if not usage or input_price in (None, "") or output_price in (None, ""): return "unavailable"
    return (usage.get("prompt_tokens", 0) * float(input_price) + usage.get("completion_tokens", 0) * float(output_price)) / 1_000_000

def _provider_diagnostics(error, status_code):
    """Keep only machine-readable SDK fields; never serialize the raw body/message."""
    body=getattr(error,"body",None)
    if isinstance(body,str):
        try: body=json.loads(body)
        except json.JSONDecodeError: body={}
    detail=body.get("error",body) if isinstance(body,dict) else {}
    headers=getattr(getattr(error,"response",None),"headers",{}) or {}
    retry_after=headers.get("retry-after") if hasattr(headers,"get") else None
    return {"http_status":status_code,"provider_status":detail.get("status") if isinstance(detail.get("status"),str) else None,"detail_types":[item.get("@type") for item in detail.get("details",[]) if isinstance(item,dict) and isinstance(item.get("@type"),str)],"retry_after":retry_after if isinstance(retry_after,(str,int,float)) else None}


class GeminiProvider:
    """Official Google Gen AI SDK adapter; API keys never enter prompts or logs."""
    def __init__(self, api_key=None, model_name=None, client=None, client_factory=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key: raise ValueError("GEMINI_API_KEY is required")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = client
        self._client_factory = client_factory
        if client is None and client_factory is None:
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError("google-genai is required for GeminiProvider") from error
            # `http_options` belongs on Client, not Models.generate_content.
            self._client_factory = lambda timeout: genai.Client(
                api_key=self.api_key,
                http_options={"timeout": int(timeout * 1000)},
            )

    def generate(self, prompt, timeout_seconds):
        try:
            client = self.client or self._client_factory(timeout_seconds)
            response = client.models.generate_content(
                model=self.model_name, contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except Exception as error:
            name = type(error).__name__.lower()
            status_code=getattr(error,"status_code",getattr(error,"code",None))
            status_code=status_code if isinstance(status_code,int) else None
            diagnostics=_provider_diagnostics(error,status_code)
            # A bare HTTP 429 / RESOURCE_EXHAUSTED response does not tell us
            # whether a short-lived rate limit or a quota exhaustion occurred.
            if status_code != 429 and ("rate" in name or "resourceexhausted" in name):
                wrapped=RateLimitError("Gemini rate limit")
            elif status_code == 503 or any(word in name for word in ("timeout", "deadline", "serviceunavailable", "connection", "server")):
                wrapped=TransientProviderError("transient Gemini provider failure")
            else:
                wrapped=ProviderError("Gemini provider request failed")
            wrapped.status_code=status_code
            wrapped.provider_diagnostics=diagnostics
            raise wrapped from error
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip(): raise ProviderError("Gemini returned an empty response")
        return ProviderResponse(text=text.strip(), token_usage=_usage(response), model=self.model_name)
