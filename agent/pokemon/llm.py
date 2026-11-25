"""The model call: OpenAI Chat Completions with strict JSON-schema output.

Like the starter's examples/llm/providers.OpenAIProvider (same interface,
same ProviderError), plus ``reasoning_effort`` for reasoning models
(GPT-5 family, o-series): lower effort answers faster, which matters for
15-second draft picks.

Environment (all optional except the key):
  OPENAI_API_KEY                  required
  POKEMON_MODEL                   battle + Team Preview model (default gpt-5.4-mini)
  POKEMON_DRAFT_MODEL             draft-pick model (default: POKEMON_MODEL)
  POKEMON_REASONING_EFFORT        none | low | medium | high (default: the model's own)
  POKEMON_DRAFT_REASONING_EFFORT  same, for draft picks (default: POKEMON_REASONING_EFFORT)
  OPENAI_BASE_URL                 default https://api.openai.com/v1
"""

from __future__ import annotations

import json
import os

import httpx

from examples.llm.providers import ProviderError

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class ChatProvider:
    def __init__(self, api_key: str, model: str, *, base_url: str = DEFAULT_BASE_URL,
                 reasoning_effort: str | None = None, timeout: float = 60.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"},
                                  timeout=timeout, transport=transport)

    def __repr__(self) -> str:  # never shows the key
        return f"ChatProvider(model={self.model!r}, reasoning_effort={self.reasoning_effort!r})"

    def complete_structured(self, messages: list[dict], schema_name: str, schema: dict, *,
                            timeout: float | None = None) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        extra = {"timeout": timeout} if timeout is not None else {}
        try:
            response = self._http.post("/chat/completions", json=body, **extra)
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI request failed ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise ProviderError(f"OpenAI request failed (HTTP {response.status_code})")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("OpenAI response had no message") from None
        if message.get("refusal"):
            raise ProviderError("the model refused to answer")
        try:
            data = json.loads(message.get("content") or "")
        except ValueError:
            raise ProviderError("the model's reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("the model's reply was not a JSON object")
        return data


def providers_from_env() -> tuple[ChatProvider | None, ChatProvider | None]:
    """(battle/preview provider, draft provider); (None, None) without a key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None
    base_url = os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    model = os.environ.get("POKEMON_MODEL") or DEFAULT_MODEL
    effort = os.environ.get("POKEMON_REASONING_EFFORT") or None
    draft_model = os.environ.get("POKEMON_DRAFT_MODEL") or model
    draft_effort = os.environ.get("POKEMON_DRAFT_REASONING_EFFORT") or effort
    main = ChatProvider(api_key, model, base_url=base_url, reasoning_effort=effort)
    if (draft_model, draft_effort) == (model, effort):
        return main, main
    return main, ChatProvider(api_key, draft_model, base_url=base_url, reasoning_effort=draft_effort)
