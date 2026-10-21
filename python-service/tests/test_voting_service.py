"""Tests for the multi-LLM voting service."""

import pytest

from src.services.multi_llm_voting_service import (
    AggregatedVote,
    MultiLLMVotingService,
    VoteResult,
    VotingSummary,
)

# --- Constructor ---


def test_requires_minimum_two_providers(fake_provider_factory):
    with pytest.raises(ValueError, match="at least 2 providers"):
        MultiLLMVotingService([fake_provider_factory()])


def test_accepts_two_providers(fake_provider_factory):
    service = MultiLLMVotingService([fake_provider_factory(), fake_provider_factory()])
    assert len(service.providers) == 2


def test_provider_names_extracted(fake_provider_factory):
    service = MultiLLMVotingService([fake_provider_factory(), fake_provider_factory()])
    # FakeProvider -> "fake"
    assert service.provider_names == ["fake", "fake"]


# --- Voting Logic ---


@pytest.mark.asyncio
async def test_unanimous_positive(fake_provider_factory):
    providers = [fake_provider_factory(decision=True) for _ in range(3)]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test prompt",
        criterion="test criterion",
        response_schema={"decision": "boolean"},
    )

    assert result.final_decision is True
    assert result.vote_count == 3
    assert result.total_voters == 3
    assert result.agreement_ratio == 1.0


@pytest.mark.asyncio
async def test_unanimous_negative(fake_provider_factory):
    providers = [fake_provider_factory(decision=False) for _ in range(3)]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test prompt",
        criterion="test criterion",
        response_schema={"decision": "boolean"},
    )

    assert result.final_decision is False
    assert result.vote_count == 3
    assert result.agreement_ratio == 1.0


@pytest.mark.asyncio
async def test_majority_two_to_one(fake_provider_factory):
    providers = [
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=False),
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is True
    assert result.vote_count == 2
    assert result.total_voters == 3
    assert result.agreement_ratio == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_majority_one_to_two(fake_provider_factory):
    providers = [
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=False),
        fake_provider_factory(decision=False),
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is False
    assert result.vote_count == 2


@pytest.mark.asyncio
async def test_tie_goes_negative_with_two_providers(fake_provider_factory):
    """With 2 providers and a 1-1 tie, positive_votes == majority_threshold (not >), so False."""
    providers = [
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=False),
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    # 1 positive out of 2 -> 1 > 1.0 is False -> decision is False
    assert result.final_decision is False


# --- Confidence Aggregation ---


@pytest.mark.asyncio
async def test_confidence_from_agreeing_voters_only(fake_provider_factory):
    providers = [
        fake_provider_factory(decision=True, confidence=0.9),
        fake_provider_factory(decision=True, confidence=0.7),
        fake_provider_factory(decision=False, confidence=0.95),
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is True
    # Confidence from agreeing voters (0.9, 0.7), not the dissenter (0.95)
    assert result.aggregated_confidence == pytest.approx(0.8)


# --- Reasoning Aggregation ---


@pytest.mark.asyncio
async def test_reasoning_from_agreeing_voters(fake_provider_factory):
    providers = [
        fake_provider_factory(decision=True, reasoning="good paper"),
        fake_provider_factory(decision=True, reasoning="relevant topic"),
        fake_provider_factory(decision=False, reasoning="not relevant"),
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert "good paper" in result.aggregated_reasoning
    assert "relevant topic" in result.aggregated_reasoning
    assert "not relevant" not in result.aggregated_reasoning


# --- Error Handling ---


@pytest.mark.asyncio
async def test_one_provider_fails_still_votes(fake_provider_factory):
    """If one of three providers fails, the remaining two still produce a valid vote."""
    failing = fake_provider_factory()
    failing.set_error(RuntimeError("API error"))

    providers = [
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=True),
        failing,
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is True
    assert result.vote_count == 2
    assert result.total_voters == 3


@pytest.mark.asyncio
async def test_empty_reply_is_a_failed_vote_not_a_negative_one(fake_provider_factory):
    """An unparseable reply arrives as {} and must not become a silent decision=False vote.

    llm_client returns {} instead of raising when a response is empty or does not match
    the schema. Counting that as a valid "criterion not satisfied" vote would push every
    affected paper toward exclusion with no error visible anywhere.
    """
    malformed = fake_provider_factory(decision=True)
    malformed.set_responses([{}])

    providers = [fake_provider_factory(decision=True), malformed]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    phantom = next(v for v in result.individual_votes if v.provider == "fake" and v.error)
    assert "decision" in phantom.error
    assert sum(1 for v in result.individual_votes if v.error is None) == 1


@pytest.mark.asyncio
async def test_empty_reply_does_not_fake_a_conflicted_vote(fake_provider_factory):
    """Two agreeing providers plus one malformed reply is unanimous, not a 2:1 split.

    The stratified verification step audits conflicted 2:1 votes, so a phantom negative
    would send manual review after disagreements that never happened.
    """
    malformed = fake_provider_factory(decision=True)
    malformed.set_responses([{}])

    providers = [
        fake_provider_factory(decision=True),
        fake_provider_factory(decision=True),
        malformed,
    ]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is True
    assert result.vote_count == 2
    assert result.total_voters == 3
    assert result.agreement_ratio == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["false", "true", None, 1])
async def test_non_boolean_decision_is_a_failed_vote(fake_provider_factory, decision):
    """bool("false") is True and bool(None) is False: neither may count as a vote."""
    odd = fake_provider_factory()
    odd.set_responses([{"decision": decision, "confidence": 0.9, "reasoning": "x"}])
    service = MultiLLMVotingService([fake_provider_factory(), fake_provider_factory(), odd])

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    failed = [v for v in result.individual_votes if v.error]
    assert len(failed) == 1
    assert "boolean 'decision'" in failed[0].error
    assert result.vote_count == 2


@pytest.mark.asyncio
async def test_insufficient_valid_votes(fake_provider_factory):
    """If too many providers fail, returns failed result."""
    failing1 = fake_provider_factory()
    failing1.set_error(RuntimeError("fail1"))
    failing2 = fake_provider_factory()
    failing2.set_error(RuntimeError("fail2"))

    providers = [fake_provider_factory(decision=True), failing1, failing2]
    service = MultiLLMVotingService(providers)

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is False
    assert result.vote_count == 0
    assert result.agreement_ratio == 0.0
    assert "Insufficient" in result.aggregated_reasoning


@pytest.mark.asyncio
async def test_all_providers_fail(fake_provider_factory):
    failing1 = fake_provider_factory()
    failing1.set_error(RuntimeError("fail1"))
    failing2 = fake_provider_factory()
    failing2.set_error(RuntimeError("fail2"))

    service = MultiLLMVotingService([failing1, failing2])

    result = await service.vote_on_criterion(
        prompt="test", criterion="C1", response_schema={"decision": "boolean"}
    )

    assert result.final_decision is False
    assert result.agreement_ratio == 0.0


# --- Batched Voting on All Criteria ---

SOURCE = {"title": "A study", "abstract": "Something relevant"}


def _evaluation(criterion_id: str, decision: bool, reasoning: str = "because") -> dict:
    return {"id": criterion_id, "reasoning": reasoning, "decision": decision, "confidence": 0.8}


def _batch_reply(**decisions: bool) -> dict:
    return {"evaluations": [_evaluation(cid, d) for cid, d in decisions.items()]}


async def _vote_all(service, inclusion=("C1", "C2"), exclusion=("X1",)):
    return await service.vote_on_all_criteria(
        inclusion_criteria=list(inclusion),
        exclusion_criteria=list(exclusion),
        source_content=SOURCE,
        research_questions=["RQ1"],
    )


@pytest.mark.asyncio
async def test_batch_makes_one_call_per_provider_and_takes_the_majority_per_criterion(
    fake_provider_factory,
):
    providers = [fake_provider_factory() for _ in range(3)]
    providers[0].set_responses([_batch_reply(I1=True, I2=True, E1=False)])
    providers[1].set_responses([_batch_reply(I1=True, I2=False, E1=False)])
    providers[2].set_responses([_batch_reply(I1=False, I2=False, E1=True)])
    service = MultiLLMVotingService(providers)

    inclusion, exclusion = await _vote_all(service)

    assert [p.call_count for p in providers] == [1, 1, 1]
    assert [v.criterion for v in inclusion] == ["C1", "C2"]
    assert [v.final_decision for v in inclusion] == [True, False]
    assert [v.criterion for v in exclusion] == ["X1"]
    assert exclusion[0].final_decision is False
    assert all(v.vote_count == 2 and v.total_voters == 3 for v in inclusion + exclusion)


@pytest.mark.asyncio
async def test_batch_reads_exclusion_votes_by_id_not_by_position(fake_provider_factory):
    """Inclusion and exclusion criteria share one reply; E1 must never pick up I1's vote."""
    reply = {"evaluations": [_evaluation("E1", True), _evaluation("I1", False)]}
    providers = [fake_provider_factory() for _ in range(2)]
    for p in providers:
        p.set_responses([reply])
    service = MultiLLMVotingService(providers)

    inclusion, exclusion = await _vote_all(service, inclusion=("C1",), exclusion=("X1",))

    assert inclusion[0].final_decision is False
    assert exclusion[0].final_decision is True


@pytest.mark.asyncio
async def test_batch_entry_without_id_is_not_matched_to_any_criterion(fake_provider_factory):
    """A reply entry that names no criterion must not be read as the answer to every one."""
    unlabeled = fake_provider_factory()
    unlabeled.set_responses(
        [{"evaluations": [{"reasoning": "?", "decision": True, "confidence": 0.9}]}]
    )
    providers = [fake_provider_factory(decision=False), fake_provider_factory(decision=False)]
    service = MultiLLMVotingService([*providers, unlabeled])

    inclusion, exclusion = await _vote_all(service)

    for vote in inclusion + exclusion:
        failed = [v for v in vote.individual_votes if v.error]
        assert len(failed) == 1
        assert "no evaluation" in failed[0].error
        assert vote.final_decision is False
        assert vote.agreement_ratio == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broken_i1, expected_error",
    [
        ({"id": "I1", "reasoning": "cut off"}, "decision"),
        ({"id": "I1", "decision": "true", "confidence": 0.9}, "decision"),
        ({"id": "I1", "decision": None, "confidence": 0.9}, "decision"),
        ({"id": "I1", "decision": True, "confidence": "high"}, "confidence"),
    ],
)
async def test_batch_malformed_evaluation_is_a_failed_vote(
    fake_provider_factory, broken_i1, expected_error
):
    """Same rules as the per-criterion path: a non-boolean decision or a non-numeric
    confidence is a failed vote, not a silent decision=False (or True)."""
    malformed = fake_provider_factory()
    malformed.set_responses([{"evaluations": [broken_i1, _evaluation("I2", True)]}])
    providers = [fake_provider_factory(decision=True), fake_provider_factory(decision=True)]
    service = MultiLLMVotingService([*providers, malformed])

    inclusion, _ = await _vote_all(service, inclusion=("C1", "C2"), exclusion=())

    first = inclusion[0]
    assert any(v.error and expected_error in v.error for v in first.individual_votes)
    assert first.final_decision is True
    assert first.vote_count == 2
    assert first.agreement_ratio == 1.0
    assert inclusion[1].vote_count == 3


@pytest.mark.asyncio
async def test_batch_provider_error_fails_only_that_providers_votes(fake_provider_factory):
    failing = fake_provider_factory()
    failing.set_error(RuntimeError("rate limited"))
    service = MultiLLMVotingService(
        [fake_provider_factory(decision=True), fake_provider_factory(decision=True), failing]
    )

    inclusion, exclusion = await _vote_all(service)

    for vote in inclusion + exclusion:
        assert vote.final_decision is True
        assert vote.vote_count == 2
        assert [v.error for v in vote.individual_votes if v.error] == ["rate limited"]


@pytest.mark.asyncio
async def test_batch_reply_without_evaluations_list_fails_that_provider(fake_provider_factory):
    empty = fake_provider_factory()
    empty.set_responses([{}])
    service = MultiLLMVotingService([fake_provider_factory(), empty])

    inclusion, _ = await _vote_all(service, inclusion=("C1",), exclusion=())

    assert "Insufficient" in inclusion[0].aggregated_reasoning
    assert any("evaluations" in (v.error or "") for v in inclusion[0].individual_votes)


@pytest.mark.asyncio
async def test_batch_reply_without_ids_is_not_read_by_position(fake_provider_factory):
    """Even with one entry per criterion, unlabelled evaluations are not assigned by
    position: the model may have listed them in another order (here under another key)."""
    unlabeled = fake_provider_factory()
    unlabeled.set_responses(
        [
            {
                "evaluations": [
                    {"criterion": "E1", "reasoning": "a", "decision": True, "confidence": 0.9},
                    {"criterion": "I1", "reasoning": "b", "decision": False, "confidence": 0.9},
                ]
            }
        ]
    )
    service = MultiLLMVotingService(
        [fake_provider_factory(decision=True), fake_provider_factory(decision=True), unlabeled]
    )

    inclusion, exclusion = await _vote_all(service, inclusion=("C1",), exclusion=("X1",))

    for vote in inclusion + exclusion:
        assert vote.vote_count == 2
        assert sum(1 for v in vote.individual_votes if v.error) == 1


@pytest.mark.asyncio
async def test_batch_duplicate_ids_are_ambiguous_and_fail(fake_provider_factory):
    """[I2, I2, E1] for criteria I1, I2, E1: neither I2 entry can be trusted for I2,
    and I1 has no answer at all."""
    confused = fake_provider_factory()
    confused.set_responses(
        [
            {
                "evaluations": [
                    _evaluation("I2", False),
                    _evaluation("I2", True),
                    _evaluation("E1", False),
                ]
            }
        ]
    )
    service = MultiLLMVotingService(
        [fake_provider_factory(decision=True), fake_provider_factory(decision=True), confused]
    )

    inclusion, exclusion = await _vote_all(service)

    assert [v.vote_count for v in inclusion] == [2, 2]
    assert all(sum(1 for v in vote.individual_votes if v.error) == 1 for vote in inclusion)
    assert exclusion[0].total_voters == 3
    assert not any(v.error for v in exclusion[0].individual_votes)


@pytest.mark.asyncio
async def test_batch_without_criteria_makes_no_calls(fake_provider_factory):
    providers = [fake_provider_factory(), fake_provider_factory()]
    service = MultiLLMVotingService(providers)

    assert await _vote_all(service, inclusion=(), exclusion=()) == ([], [])
    assert [p.call_count for p in providers] == [0, 0]


# --- Voting Summary ---


def test_compute_summary_empty(fake_provider_factory):
    service = MultiLLMVotingService([fake_provider_factory(), fake_provider_factory()])
    summary = service.compute_voting_summary([])

    assert summary.total_criteria_evaluated == 0
    assert summary.unanimous_decisions == 0
    assert summary.split_decisions == 0
    assert summary.overall_agreement_ratio == 0.0


def test_compute_summary_unanimous_and_split(fake_provider_factory):
    service = MultiLLMVotingService([fake_provider_factory(), fake_provider_factory()])

    votes = [
        AggregatedVote(
            criterion="C1",
            final_decision=True,
            vote_count=3,
            total_voters=3,
            agreement_ratio=1.0,
        ),
        AggregatedVote(
            criterion="C2",
            final_decision=False,
            vote_count=2,
            total_voters=3,
            agreement_ratio=2 / 3,
        ),
    ]

    summary = service.compute_voting_summary(votes)

    assert summary.total_criteria_evaluated == 2
    assert summary.unanimous_decisions == 1
    assert summary.split_decisions == 1
    assert summary.overall_agreement_ratio == pytest.approx((1.0 + 2 / 3) / 2)


# --- Serialization ---


def test_vote_result_to_dict():
    vote = VoteResult(provider="claude", decision=True, confidence=0.9, reasoning="looks good")
    d = vote.to_dict()
    assert d["provider"] == "claude"
    assert d["decision"] is True
    assert d["error"] is None


def test_aggregated_vote_to_dict():
    vote = AggregatedVote(
        criterion="C1",
        final_decision=True,
        vote_count=2,
        total_voters=3,
        agreement_ratio=2 / 3,
        individual_votes=[
            VoteResult(provider="a", decision=True, confidence=0.9, reasoning="yes"),
        ],
    )
    d = vote.to_dict()
    assert len(d["individual_votes"]) == 1
    assert d["criterion"] == "C1"


def test_voting_summary_to_dict():
    summary = VotingSummary(
        total_providers=3,
        providers_used=["claude", "openai", "gemini"],
        overall_agreement_ratio=0.85,
        total_criteria_evaluated=5,
        unanimous_decisions=3,
        split_decisions=2,
    )
    d = summary.to_dict()
    assert d["total_providers"] == 3
    assert d["providers_used"] == ["claude", "openai", "gemini"]
