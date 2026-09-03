"""Provider-agnostic structured-output LLM client.

Switching provider is an env var (`LLM_PROVIDER=openai|anthropic`), not a refactor.
Every call returns a validated Pydantic object or raises — malformed LLM output
never propagates into the pipeline.
"""
from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredLLM:
    def __init__(self, provider: str = "openai") -> None:
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI

            self._client = OpenAI()
        elif provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def complete(self, model: str, system: str, user: str, schema: Type[T], max_retries: int = 2) -> T:
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "openai":
                    return self._openai(model, system, user, schema)
                return self._anthropic(model, system, user, schema)
            except Exception as exc:  # noqa: BLE001 — retry any provider/validation error
                last_err = exc
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)
        raise RuntimeError(f"LLM structured call failed after retries: {last_err}")

    def chat(self, model: str, system: str, messages: list[dict]) -> str:
        """Plain conversational turn. messages: [{'role': 'user'|'assistant', 'content': str}]."""
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, *messages],
            )
            return resp.choices[0].message.content or ""
        resp = self._client.messages.create(
            model=model, max_tokens=1500, system=system, messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    # ------------------------------------------------------------- providers
    def _openai(self, model: str, system: str, user: str, schema: Type[T]) -> T:
        completion = self._client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned no parsed output (refusal or empty)")
        return parsed

    def _anthropic(self, model: str, system: str, user: str, schema: Type[T]) -> T:
        tool = {
            "name": "emit",
            "description": f"Emit a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        msg = self._client.messages.create(
            model=model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
        )
        for block in msg.content:
            if block.type == "tool_use":
                return schema.model_validate(block.input)
        raise ValueError("Anthropic returned no tool_use block")


class MockLLM(StructuredLLM):
    """Deterministic stand-in for tests: returns canned objects, no network."""

    def __init__(self, responses: dict[str, BaseModel] | None = None) -> None:  # noqa: D401
        self.provider = "mock"
        self.responses = responses or {}
        self.calls: list[dict] = []

    def chat(self, model: str, system: str, messages: list[dict]) -> str:
        self.calls.append({"model": model, "system": system, "chat": True})
        return self.responses.get("chat", "Mock reply. [PERFIL_COMPLETO]")  # type: ignore[return-value]

    def complete(self, model: str, system: str, user: str, schema: Type[T], max_retries: int = 2) -> T:
        self.calls.append({"model": model, "system": system, "user": user, "schema": schema.__name__})
        key = schema.__name__
        if key in self.responses:
            return self.responses[key]  # type: ignore[return-value]
        # Build a minimal valid object from the schema for smoke tests.
        example = _minimal_instance(schema, user)
        return example


def _minimal_instance(schema: Type[T], user: str) -> T:
    from .schemas import AnalystReport, ExecutiveDecision

    symbol = "TEST"
    try:
        payload = json.loads(user)
        symbol = payload.get("symbol", symbol)
    except Exception:
        for token in user.replace("\n", " ").split():
            if token.isupper() and 1 < len(token) <= 5 and token.isalpha():
                symbol = token
                break
    if schema is AnalystReport:
        return AnalystReport(  # type: ignore[return-value]
            symbol=symbol, stance="NEUTRAL", summary="Mock analyst report.", data_quality="PARTIAL"
        )
    if schema is ExecutiveDecision:
        return ExecutiveDecision(  # type: ignore[return-value]
            symbol=symbol, action="HOLD", conviction=3,
            thesis="Mock executive thesis.", invalidation="Mock invalidation.", horizon_days=10,
        )
    raise ValueError(f"MockLLM has no default for {schema.__name__}")
