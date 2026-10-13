"""Provider-agnostic chat model factory plus a robust JSON-to-Pydantic step.

Any LangChain chat model works: Anthropic Claude, OpenAI, Azure OpenAI, AWS Bedrock,
or a local model through Ollama. We ask for JSON and validate it with Pydantic
rather than relying on provider-specific structured-output features, so small
local models work too."""

from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5-mini",
    "ollama": "llama3.2",
}


def get_chat_model(provider: str, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    return init_chat_model(model or DEFAULT_MODELS.get(provider), model_provider=provider, temperature=temperature)


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    m = _FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def invoke_structured[T: BaseModel](llm: BaseChatModel, system: str, user: str, schema: type[T], retries: int = 1) -> T:
    """Ask for JSON matching `schema`; on a parse/validation error, retry once with the error fed back."""
    fields = json.dumps(schema.model_json_schema()["properties"], indent=1)
    messages = [
        SystemMessage(f"{system}\n\nRespond with a single JSON object with these fields:\n{fields}\nNo other text."),
        HumanMessage(user),
    ]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        reply = llm.invoke(messages)
        try:
            return schema.model_validate(extract_json(reply.content))
        except (ValueError, ValidationError) as err:
            last_err = err
            messages += [reply, HumanMessage(f"That was not valid: {err}. Reply with corrected JSON only.")]
    raise RuntimeError(f"Model failed to return valid {schema.__name__}: {last_err}")
