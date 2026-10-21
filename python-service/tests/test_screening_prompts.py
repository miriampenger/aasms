"""Tests for the screening prompts."""

from src.core.prompts import build_per_criterion_prompt


def test_per_criterion_prompt_asks_for_reasoning_before_decision():
    """Claude receives no response schema (llm_client sends it the prompt only), so the
    key order of the JSON example in the prompt is what makes it reason before it decides.
    """
    prompt = build_per_criterion_prompt(
        criterion="Reports an empirical evaluation",
        source_content={"title": "T", "abstract": "A"},
        research_questions=["RQ1"],
    )

    assert prompt.index('"reasoning"') < prompt.index('"decision"')
