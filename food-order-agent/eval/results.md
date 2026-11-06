# Resultados da avaliação

Modo: **fixture_offline**. Modelo fixture determinístico: valida execução das ferramentas, não a decisão do LLM. Custo de inferência igual a zero neste modo.

Casos: **30**
Hit@5 (casos com item esperado, n=12): **91.7%**
Taxa de chamada correta de ferramenta: **100.0%** (30/30)
Latência média por conversa: **90.5 ms**
Custo médio por conversa: **US$ 0.000000**
Casos sem resultado que retornaram lista vazia: **25.0%** (4 casos)

## Interpretação

Hit@5 só considera os casos com um item esperado definido. Os demais cobrem filtros, ambiguidades e ausência de resultados. O custo depende do provedor e dos valores `LLM_INPUT_COST_PER_1M` e `LLM_OUTPUT_COST_PER_1M` configurados.

Para medir a escolha de ferramenta pelo modelo e obter custo/latência reais, configure `.env` e execute `python eval/run_eval.py --live`.
