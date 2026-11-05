"""Session-scoped tools exposed to the conversational agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retriever import MenuRetriever


@dataclass
class Cart:
    """An in-memory cart belonging to one Streamlit/browser session."""

    items: dict[str, dict[str, Any]] = field(default_factory=dict)


class OrderTools:
    def __init__(self, retriever: MenuRetriever, cart: Cart | None = None) -> None:
        self.retriever = retriever
        self.cart = cart or Cart()
        self._known_items: dict[str, dict[str, Any]] = {}
        self.last_search_ids: set[str] = set()
        self.confirmation_pending = False

    def buscar_itens(
        self,
        consulta: str,
        preco_max: float | None = None,
        culinaria: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        if not consulta.strip():
            raise ValueError("A consulta não pode ficar vazia.")
        result = self.retriever.search(
            consulta, k=max(1, min(int(k), 10)), max_price=preco_max, cuisine=culinaria
        )
        self.last_search_ids = {hit.item_id for hit in result.hits}
        for hit in result.hits:
            self._known_items[hit.item_id] = {
                "item_id": hit.item_id,
                "nome": hit.name,
                "restaurante": hit.restaurant,
                "culinaria": hit.cuisine,
                "preco": hit.price_raw,
                "descricao": str(hit.metadata.get("description", "")),
                "score": round(hit.score, 4),
            }
        return {
            "itens": [self._known_items[hit.item_id] for hit in result.hits],
            "aviso": result.notice,
            "detalhe": result.notice_detail,
        }

    def adicionar_ao_carrinho(self, item_id: str, quantidade: int = 1) -> dict[str, Any]:
        if quantidade < 1 or quantidade > 20:
            raise ValueError("A quantidade deve estar entre 1 e 20.")
        item = self._known_items.get(item_id)
        if item is None or item_id not in self.last_search_ids:
            raise ValueError("Busque o item no cardápio antes de adicioná-lo.")
        current = self.cart.items.get(item_id, {**item, "quantidade": 0})
        current["quantidade"] += quantidade
        self.cart.items[item_id] = current
        return {"adicionado": current}

    def ver_carrinho(self) -> dict[str, Any]:
        return {"itens": list(self.cart.items.values())}

    def remover_do_carrinho(self, item_id: str) -> dict[str, Any]:
        removed = self.cart.items.pop(item_id, None)
        if removed is None:
            raise ValueError("Esse item não está no carrinho.")
        return {"removido": removed, "itens": list(self.cart.items.values())}

    def confirmar_pedido(self) -> dict[str, Any]:
        if not self.confirmation_pending:
            return {"confirmado": False, "mensagem": "Aguardando confirmação explícita do usuário."}
        if not self.cart.items:
            return {"confirmado": False, "mensagem": "O carrinho está vazio."}
        self.confirmation_pending = False
        order = list(self.cart.items.values())
        self.cart.items.clear()
        return {"confirmado": True, "itens": order}

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call only the allowlisted tool names and return JSON-safe data."""
        handlers = {
            "buscar_itens": self.buscar_itens,
            "adicionar_ao_carrinho": self.adicionar_ao_carrinho,
            "ver_carrinho": self.ver_carrinho,
            "remover_do_carrinho": self.remover_do_carrinho,
            "confirmar_pedido": self.confirmar_pedido,
        }
        if name not in handlers:
            raise ValueError(f"Ferramenta desconhecida: {name}")
        return handlers[name](**arguments)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "buscar_itens", "description": "Busca itens existentes no cardápio. Use filtros quando forem pedidos.", "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}, "preco_max": {"type": ["number", "null"]}, "culinaria": {"type": ["string", "null"]}, "k": {"type": "integer", "default": 5}}, "required": ["consulta"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "adicionar_ao_carrinho", "description": "Adiciona um item recém encontrado ao carrinho.", "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}, "quantidade": {"type": "integer", "default": 1}}, "required": ["item_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "ver_carrinho", "description": "Mostra os itens do carrinho.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "remover_do_carrinho", "description": "Remove um item do carrinho.", "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "confirmar_pedido", "description": "Finaliza o pedido somente depois de autorização explícita do usuário.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
]
