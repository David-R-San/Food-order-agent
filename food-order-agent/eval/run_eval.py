"""Run retrieval and tool-path evaluation, optionally against a live LLM.

Default mode uses a deterministic fixture model so the evaluation runs without
API credentials or token charges. Pass --live to measure actual model latency,
token use, and provider-reported cost. Fixture tool-call rate validates tool
execution/wiring, not the model's ability to choose a tool.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import OrderAgent  # noqa: E402
from src.config import CHROMA_DIR  # noqa: E402
from src.ingest import build_index  # noqa: E402
from src.llm import LLMResponse, OpenAICompatibleChat, ToolCall  # noqa: E402
from src.retriever import MenuRetriever, SentenceTransformerEmbedder  # noqa: E402
from src.tools import OrderTools  # noqa: E402


class FixtureModel:
    def __init__(self, arguments: dict[str, Any]) -> None:
        self.arguments = arguments
        self.called = False

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        if not self.called:
            self.called = True
            return LLMResponse(None, [ToolCall("eval-call", "buscar_itens", self.arguments)])
        return LLMResponse("Busca de avaliação concluída.")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_index() -> MenuRetriever:
    embedder = SentenceTransformerEmbedder()
    try:
        retriever = MenuRetriever(embedder=embedder)
        if retriever.count() == 0:
            # Recover cleanly from an interrupted first-time embedding run.
            return build_index(embedder=embedder)
        return retriever
    except Exception:
        return build_index(embedder=embedder)


def run(live: bool = False) -> dict[str, Any]:
    cases = load_cases(ROOT / "eval" / "cases.jsonl")
    retriever = ensure_index()
    hits = 0
    eligible = 0
    correct_tools = 0
    latencies: list[float] = []
    costs: list[float] = []
    no_result_cases = [case for case in cases if case["id"].startswith("none-")]
    no_result_empty = 0
    failures: list[dict[str, str]] = []

    for case in cases:
        filters = case.get("filters", {})
        query = case["query"]
        expected_name = case.get("expected_item")
        search_result = retriever.search(
            query,
            k=5,
            max_price=filters.get("max_price"),
            cuisine=filters.get("cuisine"),
            min_score=0.20,
        )
        if expected_name:
            eligible += 1
            if any(expected_name.casefold() in item.name.casefold() for item in search_result.hits):
                hits += 1
            else:
                failures.append({"id": case["id"], "reason": "expected_item_not_in_top5"})
        if case in no_result_cases and not search_result.hits:
            no_result_empty += 1

        arguments = {
            "consulta": query,
            "preco_max": filters.get("max_price"),
            "culinaria": filters.get("cuisine"),
            "k": 5,
        }
        model = OpenAICompatibleChat() if live else FixtureModel(arguments)
        agent = OrderAgent(model, min_score=0.20)
        conversation = agent.new_conversation(OrderTools(retriever))
        _answer, turn_log = agent.turn(conversation, query)
        observed = [call for call in turn_log.tool_calls if "nome" in call]
        expected_tool = case["expected_tool"]
        matching_call = any(
            call.get("nome") == expected_tool
            and call.get("argumentos", {}).get("consulta") == query
            and call.get("argumentos", {}).get("preco_max") == filters.get("max_price")
            and call.get("argumentos", {}).get("culinaria") == filters.get("cuisine")
            for call in observed
        )
        if matching_call and not any("erro" in call for call in turn_log.tool_calls):
            correct_tools += 1
        elif live:
            failures.append({"id": case["id"], "reason": "expected_tool_not_called"})
        latencies.append(turn_log.latency_ms)
        costs.append(turn_log.cost_usd)

    return {
        "mode": "live_llm" if live else "fixture_offline",
        "total_cases": len(cases),
        "hit_at_5": hits / eligible if eligible else 0.0,
        "hit_at_5_cases": eligible,
        "tool_call_rate": correct_tools / len(cases) if cases else 0.0,
        "tool_call_correct": correct_tools,
        "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "mean_cost_usd": statistics.mean(costs) if costs else 0.0,
        "no_result_empty_rate": no_result_empty / len(no_result_cases) if no_result_cases else 0.0,
        "no_result_cases": len(no_result_cases),
        "failures": failures,
    }


def write_report(result: dict[str, Any], path: Path) -> None:
    mode_text = (
        "LLM real configurado no ambiente; latência e custo vêm da execução do provedor."
        if result["mode"] == "live_llm"
        else "Modelo fixture determinístico: valida execução das ferramentas, não a decisão do LLM. Custo de inferência igual a zero neste modo."
    )
    lines = [
        "# Resultados da avaliação",
        "",
        f"Modo: **{result['mode']}**. {mode_text}",
        "",
        f"Casos: **{result['total_cases']}**",
        f"Hit@5 (casos com item esperado, n={result['hit_at_5_cases']}): **{result['hit_at_5']:.1%}**",
        f"Taxa de chamada correta de ferramenta: **{result['tool_call_rate']:.1%}** ({result['tool_call_correct']}/{result['total_cases']})",
        f"Latência média por conversa: **{result['mean_latency_ms']:.1f} ms**",
        f"Custo médio por conversa: **US$ {result['mean_cost_usd']:.6f}**",
        f"Casos sem resultado que retornaram lista vazia: **{result['no_result_empty_rate']:.1%}** ({result['no_result_cases']} casos)",
        "",
        "## Interpretação",
        "",
        "Hit@5 só considera os casos com um item esperado definido. Os demais cobrem filtros, ambiguidades e ausência de resultados. O custo depende do provedor e dos valores `LLM_INPUT_COST_PER_1M` e `LLM_OUTPUT_COST_PER_1M` configurados.",
        "",
        "Para medir a escolha de ferramenta pelo modelo e obter custo/latência reais, configure `.env` e execute `python eval/run_eval.py --live`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Usa o LLM configurado; pode consumir créditos.")
    args = parser.parse_args()
    try:
        result = run(live=args.live)
    except Exception as exc:
        raise SystemExit(f"Falha na avaliação: {exc}. Gere o índice com `python -m src.ingest`.") from exc
    write_report(result, ROOT / "eval" / "results.md")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Relatório salvo em {ROOT / 'eval' / 'results.md'}")


if __name__ == "__main__":
    main()
