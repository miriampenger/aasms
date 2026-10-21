"""Tests for the screening prompts."""

from src.core.prompts import build_all_criteria_prompt, build_per_criterion_prompt


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


def test_all_criteria_prompt_numbers_criteria_and_embeds_the_document_once():
    prompt = build_all_criteria_prompt(
        inclusion_criteria=["Proposes a tool", "Reports an evaluation"],
        exclusion_criteria=["Not peer reviewed"],
        source_content={"title": "Unique Title 42", "abstract": "A"},
        research_questions=["RQ1"],
    )

    assert "I1. Proposes a tool" in prompt
    assert "I2. Reports an evaluation" in prompt
    assert "E1. Not peer reviewed" in prompt
    assert "exactly 3 evaluations" in prompt
    assert prompt.count("Unique Title 42") == 1
    assert prompt.index('"reasoning"') < prompt.index('"decision"')


def test_all_criteria_prompt_leaves_out_an_empty_criteria_group():
    prompt = build_all_criteria_prompt(
        inclusion_criteria=["Proposes a tool"],
        exclusion_criteria=[],
        source_content={"title": "T"},
        research_questions=[],
    )

    assert "Exclusion criteria" not in prompt
    assert "exactly 1 evaluations" in prompt
