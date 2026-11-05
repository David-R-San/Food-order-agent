"""Provider configuration defaults to Gemini's free API model."""

from src.llm import OpenAICompatibleChat, adapt_tools
from src.tools import TOOL_SCHEMAS
from types import SimpleNamespace


def test_default_provider_uses_gemini_free_flash_lite(monkeypatch) -> None:
    # Keep a developer's local .env from changing this configuration test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    for key in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    import sys
    from types import ModuleType

    fake_google = ModuleType("google")
    fake_genai = SimpleNamespace(Client=lambda **_kwargs: object())
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    chat = OpenAICompatibleChat()

    assert chat.model == "gemini-3.1-flash-lite"
    assert chat.provider == "gemini"
    assert chat.client is not None


def test_gemini_tool_schemas_drop_unsupported_json_schema_fields() -> None:
    adapted = adapt_tools(TOOL_SCHEMAS, "gemini")
    search_parameters = adapted[0]["function"]["parameters"]

    assert search_parameters["properties"]["preco_max"]["type"] == "number"
    assert search_parameters["properties"]["culinaria"]["type"] == "string"
    assert "default" not in search_parameters["properties"]["k"]
    assert "additionalProperties" not in search_parameters
    # Provider adaptation must not mutate the schemas sent to other providers.
    assert TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["preco_max"]["type"] == ["number", "null"]


def test_openai_compatible_empty_tool_call_id_gets_a_generated_id() -> None:
    chat = OpenAICompatibleChat.__new__(OpenAICompatibleChat)
    chat.provider = "groq"
    chat.model = "gemini-3.1-flash-lite"
    chat.input_cost = 0.0
    chat.output_cost = 0.0
    tool_call = SimpleNamespace(
        id="",
        function=SimpleNamespace(name="buscar_itens", arguments='{"consulta":"tofu"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    result = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )
    chat.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: result)
        )
    )

    response = chat.complete([], TOOL_SCHEMAS)

    assert response.tool_calls[0].call_id
    assert response.tool_calls[0].name == "buscar_itens"
