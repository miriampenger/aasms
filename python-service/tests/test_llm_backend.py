import importlib

import pytest

from src.core import config as config_module
from src.core import llm_provider as lp
from src.services.classification_service import ClassificationService
from src.services.inclusion_evaluation_service import InclusionEvaluationService


@pytest.mark.asyncio
async def test_evaluate_inclusion_recommendation_changes_with_exclusion(fake_provider_factory):
    source_content = {"title": "Test", "abstract": "Something", "content": "body"}
    provider = fake_provider_factory(decision=True)
    service = InclusionEvaluationService(llm_provider=provider, use_voting=False)

    # Test 1: Inclusion only (Expect True)
    result = await service.evaluate_inclusion(
        source_content=source_content,
        inclusion_criteria=["C1", "C2"],
        exclusion_criteria=[],
        research_questions=["RQ1"],
    )
    assert result["recommendation"] is True

    # Test 2: With Exclusion (Expect False because E1 will be 'decided' as True -> Excluded)
    provider2 = fake_provider_factory(decision=True)
    service2 = InclusionEvaluationService(llm_provider=provider2, use_voting=False)
    result_negative = await service2.evaluate_inclusion(
        source_content=source_content,
        inclusion_criteria=["C1"],
        exclusion_criteria=["E1"],
        research_questions=["RQ1"],
    )
    assert result_negative["recommendation"] is False


@pytest.mark.asyncio
async def test_classification_formatting_handles_list_schema(fake_provider_factory):
    source_content = {"title": "Test", "abstract": "Something", "content": "body"}
    schema = [{"name": "venue", "categories": ["journal", "conference"]}]
    provider = fake_provider_factory(decision=True)
    classification_service = ClassificationService(provider)

    classifications = await classification_service.classify_source(
        source_content=source_content,
        classification_schema=schema,
        research_questions=["RQ1"],
    )
    formatted = ClassificationService.format_classifications(classifications, schema)

    assert formatted[0]["facet_name"] == "venue"
    assert formatted[0]["category"] in ["journal", "conference", "unknown", "Not Applicable"]


def test_resolve_provider_prefers_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "keyA")
    monkeypatch.setenv("OPENAI_API_KEY", "keyO")
    monkeypatch.setenv("LLM_PROVIDER", "auto")

    importlib.reload(config_module)
    importlib.reload(lp)

    provider_name, _ = lp.resolve_provider_name()
    assert provider_name == "claude"

    # Clean up
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_resolve_provider_small_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "keyA")
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("CLAUDE_SMALL_MODEL", "claude-small")

    importlib.reload(config_module)
    importlib.reload(lp)

    provider_name, model_name = lp.resolve_provider_name(prefer_small_model=True)
    assert provider_name == "claude"
    assert model_name == "claude-small"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("CLAUDE_SMALL_MODEL", raising=False)


class _FakeOpenAIClient:
    """Stands in for AsyncOpenAI: records with_options() and responses.create() calls."""

    def __init__(self):
        self.options = {}
        self.request = {}
        client = self

        class Responses:
            async def create(self, **kwargs):
                client.request.update(kwargs)

                class FakeResponse:
                    id = "resp_123"
                    output_text = '{"ok": true}'

                return FakeResponse()

        self.responses = Responses()

    def with_options(self, **options):
        self.options.update(options)
        return self


async def _call_openai(monkeypatch, effort: str):
    import src.services.llm_client as llm_client

    fake = _FakeOpenAIClient()
    monkeypatch.setattr(llm_client, "_get_openai_client", lambda: fake)
    monkeypatch.setattr(llm_client.settings, "openai_reasoning_effort", effort)

    result = await llm_client._generate_json_openai(
        model_name="gpt-5.4-mini-2026-03-17",
        prompt="screen this paper",
        temperature=0.1,
        max_tokens=200,
        response_schema=None,
    )
    assert result == {"ok": True, "_response_id": "resp_123"}
    return fake


@pytest.mark.asyncio
async def test_openai_reasoning_effort_is_sent_when_configured(monkeypatch):
    fake = await _call_openai(monkeypatch, effort="low")
    assert fake.request["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_openai_reasoning_effort_is_left_to_the_model_by_default(monkeypatch):
    """Defaults differ by model (gpt-5.2 uses "none", gpt-5-pro only accepts "high")."""
    fake = await _call_openai(monkeypatch, effort="")
    assert "reasoning" not in fake.request


@pytest.mark.asyncio
async def test_generate_json_calls_leave_retries_to_tenacity(monkeypatch):
    """SDK retries inside tenacity's attempts would multiply requests (up to 36 for one
    Anthropic call), so the generate_json calls run with max_retries=0."""
    import src.services.llm_client as llm_client

    openai_fake = await _call_openai(monkeypatch, effort="")
    assert openai_fake.options == {"max_retries": 0}

    anthropic_options = {}

    class FakeAnthropicClient:
        def with_options(self, **options):
            anthropic_options.update(options)
            return self

        class messages:  # noqa: N801 - mirrors the SDK attribute name
            @staticmethod
            async def create(**kwargs):
                class Block:
                    text = '{"ok": true}'

                class Message:
                    content = [Block()]

                return Message()

    monkeypatch.setattr(llm_client, "_get_anthropic_client", lambda: FakeAnthropicClient())
    result = await llm_client._generate_json_claude("claude-small", "prompt", 0.1, 100)

    assert result == {"ok": True}
    assert anthropic_options == {"max_retries": 0}


def test_shared_sdk_clients_keep_their_own_retries(monkeypatch):
    """pdf_service calls the shared clients without tenacity, so they keep SDK retries."""
    import src.services.llm_client as llm_client

    monkeypatch.setattr(llm_client.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(llm_client.settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_client, "_anthropic_client", None)
    monkeypatch.setattr(llm_client, "_openai_client", None)

    assert llm_client._get_anthropic_client().max_retries == 5
    assert llm_client._get_openai_client().max_retries > 0
