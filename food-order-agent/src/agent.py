"""Manual tool-calling loop with order-confirmation and budget guardrails."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from src.llm import ChatModel
from src.tools import OrderTools, TOOL_SCHEMAS

logger = logging.getLogger(__name__)
LOW_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))


@dataclass
class TurnLog:
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class Conversation:
    tools: OrderTools
    messages: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    logs: list[TurnLog] = field(default_factory=list)
    awaiting_confirmation: bool = False


class OrderAgent:
    def __init__(
        self,
        model: ChatModel,
        *,
        max_steps: int = 5,
        max_conversation_tokens: int | None = None,
        max_conversation_cost_usd: float | None = None,
        min_score: float = LOW_SCORE,
    ) -> None:
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_conversation_tokens or int(os.getenv("MAX_CONVERSATION_TOKENS", "12000"))
        self.max_cost = max_conversation_cost_usd or float(os.getenv("MAX_CONVERSATION_COST_USD", "0.25"))
        self.min_score = min_score

    @staticmethod
    def new_conversation(tools: OrderTools) -> Conversation:
        return Conversation(tools=tools)

    def turn(self, conversation: Conversation, user_text: str) -> tuple[str, TurnLog]:
        log = TurnLog()
        started = time.perf_counter()
        if conversation.awaiting_confirmation and _explicit_confirmation(user_text):
            conversation.tools.confirmation_pending = True
            conversation.awaiting_confirmation = False
        elif _explicit_rejection(user_text):
            conversation.tools.confirmation_pending = False
            conversation.awaiting_confirmation = False

        conversation.messages.append({"role": "user", "content": user_text})
        system = {
            "role": "system",
            "content": (
                "Você é um assistente de pedidos de comida. Responda sempre em português do Brasil. "
                "Use buscar_itens para consultar o cardápio; nunca invente pratos, preços ou restaurantes. "
                "Recomende somente itens retornados pelas ferramentas e respeite os filtros pedidos. "
                "Quando a busca vier vazia ou com aviso, diga claramente que não encontrou. "
                "Não afirme segurança sobre alergênicos ou contaminação cruzada; a descrição do menu não comprova isso. "
                "Antes de confirmar pedido, peça confirmação explícita e aguarde uma mensagem afirmativa do usuário."
            ),
        }
        for _ in range(self.max_steps):
            if self._over_budget(conversation):
                answer = "A conversa atingiu o limite de uso configurado. Posso continuar em uma nova conversa."
                conversation.messages.append({"role": "assistant", "content": answer})
                break
            try:
                response = self.model.complete([system, *conversation.messages], TOOL_SCHEMAS)
            except Exception as exc:  # API failures must not break the chat session.
                detail = _api_error_detail(exc)
                logger.error("LLM request failed (%s): %s", type(exc).__name__, detail)
                answer = "Tive um problema ao consultar o assistente agora. Tente novamente em instantes."
                log.tool_calls.append({"erro_api": type(exc).__name__})
                conversation.messages.append({"role": "assistant", "content": answer})
                break

            log.prompt_tokens += response.prompt_tokens
            log.completion_tokens += response.completion_tokens
            log.cost_usd += response.cost_usd
            conversation.total_tokens += response.prompt_tokens + response.completion_tokens
            conversation.total_cost_usd += response.cost_usd

            if not response.tool_calls:
                answer = response.content or "Como posso ajudar com seu pedido?"
                if _asks_for_confirmation(answer):
                    conversation.awaiting_confirmation = True
                conversation.messages.append({"role": "assistant", "content": answer})
                break

            # Recover a missing required query from the current user turn. Some
            # compatible APIs occasionally return an empty argument object.
            latest_user_text = next(
                (
                    message["content"]
                    for message in reversed(conversation.messages)
                    if message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                ),
                "",
            )
            for call in response.tool_calls:
                if call.name == "buscar_itens" and not str(call.arguments.get("consulta", "")).strip():
                    call.arguments["consulta"] = latest_user_text

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [
                    {"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                    for call in response.tool_calls
                ],
            }
            # Omit null/empty content in tool-call messages; Gemini's OpenAI
            # compatibility endpoint can reject an empty content field later.
            if response.content:
                assistant_message["content"] = response.content
            conversation.messages.append(assistant_message)
            for call in response.tool_calls:
                try:
                    result = conversation.tools.dispatch(call.name, call.arguments)
                    if call.name == "buscar_itens":
                        result = self._apply_score_guardrail(result)
                        conversation.tools.last_search_ids.intersection_update(
                            item["item_id"] for item in result.get("itens", [])
                        )
                    log.tool_calls.append({"nome": call.name, "argumentos": call.arguments, "resultado": result})
                except Exception as exc:
                    logger.exception("Tool %s failed", call.name)
                    result = {"erro": str(exc)}
                    log.tool_calls.append({"nome": call.name, "argumentos": call.arguments, "erro": str(exc)})
                conversation.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": json.dumps(result, ensure_ascii=False)})
        else:
            answer = "Não consegui concluir essa etapa com segurança. Pode reformular o pedido?"
            conversation.messages.append({"role": "assistant", "content": answer})

        log.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        conversation.logs.append(log)
        return answer, log

    def _over_budget(self, conversation: Conversation) -> bool:
        return conversation.total_tokens >= self.max_tokens or conversation.total_cost_usd >= self.max_cost

    def _apply_score_guardrail(self, result: dict[str, Any]) -> dict[str, Any]:
        items = result.get("itens", [])
        result["itens"] = [item for item in items if item["score"] >= self.min_score]
        if items and not result["itens"]:
            result["aviso"] = "below_min_score"
        if result.get("aviso"):
            # Do not allow stale weak results to be added to the cart.
            return result
        allowed = {item["item_id"] for item in result["itens"]}
        self._active_search_ids = allowed
        return result


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _api_error_detail(exc: Exception) -> str:
    """Extract a safe, concise provider error without logging request headers."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:500]
    response = getattr(exc, "response", None)
    try:
        text = response.text if response is not None else ""
    except Exception:
        text = ""
    return (text or str(exc) or "Sem detalhes fornecidos pelo provedor.")[:500]


def _explicit_confirmation(text: str) -> bool:
    folded = _fold(text).strip()
    return bool(
        re.fullmatch(
            r"(sim([, ]+confirmo)?|confirmo|pode confirmar|confirmar pedido|fechar pedido|pode fechar|isso mesmo)[.! ]*",
            folded,
        )
    )


def _explicit_rejection(text: str) -> bool:
    folded = _fold(text).strip()
    return bool(re.fullmatch(r"(nao|cancela|deixa pra la|nao confirmar)[.! ]*", folded))


def _asks_for_confirmation(text: str) -> bool:
    folded = _fold(text)
    asks_about_order = any(
        word in folded for word in ("pedido", "carrinho", "compra", "finalizar", "fechar")
    )
    asks_for_yes = "responda sim" in folded or "diga sim" in folded
    return asks_about_order and ("?" in text and "confirm" in folded or asks_for_yes)
