"""Streamlit demo for the Portuguese food-order assistant."""

from __future__ import annotations

import logging

import streamlit as st

from src.agent import Conversation, OrderAgent
from src.config import CHROMA_DIR
from src.ingest import build_index
from src.llm import OpenAICompatibleChat
from src.retriever import MenuRetriever
from src.tools import Cart, OrderTools

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Assistente de pedidos", page_icon="🍽️", layout="wide")
st.title("🍽️ Assistente de pedidos")
st.caption("Cardápios de demonstração · Faça seu pedido em português")


@st.cache_resource(show_spinner="Carregando o modelo de busca...")
def load_retriever() -> MenuRetriever | None:
    if not CHROMA_DIR.exists():
        return None
    try:
        retriever = MenuRetriever()
        return retriever if retriever.count() else None
    except Exception:
        logging.exception("Could not open the menu index")
        return None


retriever = load_retriever()
if retriever is None:
    st.info("O índice de cardápios ainda não existe. Gere-o para iniciar a conversa.")
    if st.button("Criar índice de cardápios", type="primary"):
        try:
            with st.spinner("Lendo 5.213 itens e criando embeddings multilíngues..."):
                build_index()
            load_retriever.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível criar o índice: {exc}")
    st.code("python -m src.ingest", language="bash")
    st.stop()

if "order_tools" not in st.session_state:
    st.session_state.order_tools = OrderTools(retriever, Cart())
    st.session_state.conversation = Conversation(st.session_state.order_tools)
    st.session_state.chat_history = []
    st.session_state.agent = None

tools: OrderTools = st.session_state.order_tools

with st.sidebar:
    st.header("🛒 Seu carrinho")
    cart_items = tools.ver_carrinho()["itens"]
    if not cart_items:
        st.caption("Seu carrinho está vazio.")
    else:
        for item in cart_items:
            st.markdown(f"**{item['quantidade']}× {item['nome']}**")
            st.caption(f"{item['restaurante']} · {item['preco'] or 'preço não informado'}")
    st.divider()
    st.caption(f"Itens distintos: {len(cart_items)}")
    with st.expander("Logs das ferramentas", expanded=False):
        logs = [
            {
                "latencia_ms": log.latency_ms,
                "tokens": log.prompt_tokens + log.completion_tokens,
                "custo_usd": log.cost_usd,
                "ferramentas": log.tool_calls,
            }
            for log in st.session_state.conversation.logs
        ]
        st.json(logs or {"mensagem": "Ainda não há chamadas nesta sessão."})

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ex.: Quero algo vegetariano até 20 dólares")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            if st.session_state.agent is None:
                st.session_state.agent = OrderAgent(OpenAICompatibleChat())
            with st.spinner("Consultando o cardápio..."):
                answer, _turn_log = st.session_state.agent.turn(
                    st.session_state.conversation, prompt
                )
            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as exc:
            logging.exception("Chat turn failed")
            answer = "Não consegui responder agora. Confira as configurações do LLM e tente novamente."
            st.error(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
    st.rerun()
