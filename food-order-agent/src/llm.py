"""Configurable chat interface for Gemini and OpenAI-compatible providers."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.config import ROOT


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse: ...


class OpenAICompatibleChat:
    """Use native Gemini chat or an OpenAI-compatible cloud provider."""

    DEFAULT_BASE_URLS = {
        "openai": None,
        "gemini": None,
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }

    def __init__(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        # Gemini's free API tier is the default for easy portfolio setup.
        provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        self.provider = provider
        self.model = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if provider not in self.DEFAULT_BASE_URLS:
            raise ValueError(f"Provedor não suportado: {provider}")
        if not api_key:
            raise ValueError("Configure LLM_API_KEY no arquivo .env.")
        self.input_cost = float(os.getenv("LLM_INPUT_COST_PER_1M", "0"))
        self.output_cost = float(os.getenv("LLM_OUTPUT_COST_PER_1M", "0"))
        self._gemini_chat = None
        self._gemini_sent_messages = 0
        self._gemini_call_names: dict[str, str] = {}
        if provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=api_key)
        else:
            from openai import OpenAI
            base_url = os.getenv("LLM_BASE_URL") or self.DEFAULT_BASE_URLS.get(provider)
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        if self.provider == "gemini":
            return self._complete_gemini(messages, tools)
        result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=adapt_tools(tools, self.provider),
            tool_choice="auto",
        )
        message = result.choices[0].message
        usage = result.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("O modelo retornou argumentos de ferramenta inválidos.") from exc
            if not isinstance(args, dict):
                raise ValueError("Argumentos de ferramenta devem ser um objeto JSON.")
            # Some OpenAI-compatible adapters omit tool-call IDs. The agent
            # needs a stable non-empty ID to pair the subsequent tool result.
            call_id = str(call.id or uuid.uuid4().hex)
            calls.append(ToolCall(call_id, call.function.name, args))
        cost = (prompt_tokens * self.input_cost + completion_tokens * self.output_cost) / 1_000_000
        return LLMResponse(message.content, calls, prompt_tokens, completion_tokens, cost)

    def _complete_gemini(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Use Gemini's native chat so function-call signatures survive tool round trips."""
        from google.genai import types

        system_text = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if self._gemini_chat is None:
            declarations = []
            for item in adapt_tools(tools, "gemini"):
                fn = item["function"]
                declarations.append({"name": fn["name"], "description": fn.get("description", ""), "parameters": fn["parameters"]})
            config = types.GenerateContentConfig(
                system_instruction=system_text,
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )
            self._gemini_chat = self.client.chats.create(model=self.model, config=config)

        delta = messages[self._gemini_sent_messages:]
        self._gemini_sent_messages = len(messages)
        response = None
        for message in delta:
            role = message.get("role")
            if role == "user":
                response = self._gemini_chat.send_message(message.get("content", ""))
            elif role == "tool":
                call_id = message.get("tool_call_id", "")
                name = self._gemini_call_names.get(call_id)
                if name:
                    try:
                        result_data = json.loads(message.get("content", "{}"))
                    except (TypeError, json.JSONDecodeError):
                        result_data = {"result": message.get("content", "")}
                    response = self._gemini_chat.send_message(
                        types.Part.from_function_response(name=name, response=result_data)
                    )
        if response is None:
            return LLMResponse("Como posso ajudar com seu pedido?")

        calls: list[ToolCall] = []
        content_parts: list[str] = []
        for part in response.candidates[0].content.parts if response.candidates else []:
            if part.function_call:
                function = part.function_call
                call_id = str(getattr(function, "id", "") or uuid.uuid4().hex)
                args = dict(function.args or {})
                calls.append(ToolCall(call_id, function.name, args))
                self._gemini_call_names[call_id] = function.name
            elif part.text:
                content_parts.append(part.text)
        usage = response.usage_metadata
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cost = (prompt_tokens * self.input_cost + completion_tokens * self.output_cost) / 1_000_000
        return LLMResponse("\n".join(content_parts) or None, calls, prompt_tokens, completion_tokens, cost)


def adapt_tools(tools: list[dict[str, Any]], provider: str) -> list[dict[str, Any]]:
    """Normalize function schemas for provider-specific OpenAI compatibility.

    Gemini function declarations support a narrower JSON Schema subset than
    OpenAI's tool schema. Optional fields are omitted when not applicable, so
    nullable unions can safely become their non-null type.
    """
    adapted = deepcopy(tools)
    if provider != "gemini":
        return adapted

    for tool in adapted:
        function = tool.get("function", {})
        parameters = function.get("parameters", {})
        function["parameters"] = _normalize_gemini_schema(parameters)
    return adapted


def _normalize_gemini_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_normalize_gemini_schema(value) for value in schema]
    if not isinstance(schema, dict):
        return schema

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        # These OpenAPI/JSON Schema fields are not needed by Gemini's tools.
        if key in {"additionalProperties", "default"}:
            continue
        if key == "type" and isinstance(value, list):
            non_null_types = [item for item in value if item != "null"]
            if len(non_null_types) == 1:
                normalized[key] = non_null_types[0]
            elif non_null_types:
                normalized[key] = non_null_types
            continue
        normalized[key] = _normalize_gemini_schema(value)
    return normalized
