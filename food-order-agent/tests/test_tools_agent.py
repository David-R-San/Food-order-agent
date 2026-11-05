"""Cart isolation and tool-loop guardrail tests."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent import Conversation, OrderAgent
from src.llm import LLMResponse, ToolCall
from src.tools import Cart, OrderTools


class FakeRetriever:
    def search(self, query, k=5, max_price=None, cuisine=None):
        if query == "nada":
            return SimpleNamespace(hits=[], notice="no_matches", notice_detail=None)
        metadata = {
            "name": "Jerk Chicken",
            "restaurant": "Rasta Belly",
            "cuisine": "Caribbean",
            "price_raw": "$12",
            "description": "Grilled chicken",
        }
        hit = SimpleNamespace(item_id="item-1", metadata=metadata, name="Jerk Chicken", restaurant="Rasta Belly", cuisine="Caribbean", price_raw="$12", score=0.91)
        return SimpleNamespace(hits=[hit], notice=None, notice_detail=None)


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages, tools):
        return next(self.responses)


def test_cart_requires_a_recent_search_and_is_session_scoped():
    first = OrderTools(FakeRetriever(), Cart())
    second = OrderTools(FakeRetriever(), Cart())
    found = first.buscar_itens("frango")['itens'][0]
    first.adicionar_ao_carrinho(found["item_id"], 2)
    assert first.ver_carrinho()["itens"][0]["quantidade"] == 2
    assert second.ver_carrinho()["itens"] == []
    try:
        second.adicionar_ao_carrinho(found["item_id"])
    except ValueError as exc:
        assert "Busque" in str(exc)
    else:
        raise AssertionError("item without a search should not be addable")


def test_order_confirmation_is_blocked_until_agent_requested_confirmation():
    tools = OrderTools(FakeRetriever())
    item = tools.buscar_itens("frango")["itens"][0]
    tools.adicionar_ao_carrinho(item["item_id"])
    agent = OrderAgent(FakeModel([
        LLMResponse(None, [ToolCall("call-1", "confirmar_pedido", {})]),
        LLMResponse("Confirma o pedido do carrinho?"),
    ]))
    conversation = Conversation(tools)
    answer, log = agent.turn(conversation, "sim")
    assert answer == "Confirma o pedido do carrinho?"
    assert log.tool_calls[0]["resultado"]["confirmado"] is False
    assert tools.ver_carrinho()["itens"]


def test_explicit_confirmation_only_works_after_prompt():
    tools = OrderTools(FakeRetriever())
    item = tools.buscar_itens("frango")["itens"][0]
    tools.adicionar_ao_carrinho(item["item_id"])
    model = FakeModel([
        LLMResponse("Confirma o pedido do carrinho?"),
        LLMResponse(None, [ToolCall("call-2", "confirmar_pedido", {})]),
        LLMResponse("Pedido confirmado!"),
    ])
    agent = OrderAgent(model)
    conversation = Conversation(tools)
    agent.turn(conversation, "quero fechar")
    assert conversation.awaiting_confirmation
    answer, _ = agent.turn(conversation, "Sim, confirmo")
    assert answer == "Pedido confirmado!"
    assert tools.ver_carrinho()["itens"] == []
    assert not conversation.awaiting_confirmation


def test_api_error_does_not_break_turn():
    class BrokenModel:
        def complete(self, messages, tools):
            raise RuntimeError("offline")

    agent = OrderAgent(BrokenModel())
    answer, log = agent.turn(Conversation(OrderTools(FakeRetriever())), "oi")
    assert "problema" in answer
    assert log.latency_ms >= 0


def test_missing_search_query_uses_current_user_message():
    tools = OrderTools(FakeRetriever())
    agent = OrderAgent(FakeModel([
        LLMResponse(None, [ToolCall("call-missing-query", "buscar_itens", {})]),
        LLMResponse("Encontrei uma opção."),
    ]))
    conversation = Conversation(tools)

    answer, log = agent.turn(conversation, "Quero opções de tofu")

    assert answer == "Encontrei uma opção."
    assert log.tool_calls[0]["nome"] == "buscar_itens"
    assert log.tool_calls[0]["argumentos"]["consulta"] == "Quero opções de tofu"
    assistant_tool_message = next(
        message for message in conversation.messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert "content" not in assistant_tool_message
