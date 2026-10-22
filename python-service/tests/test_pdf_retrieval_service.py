"""Tests for the open-access PDF retrieval service."""

import httpx
import pytest

from src.services.pdf_retrieval_service import (
    ELSEVIER_DOI_PREFIXES,
    ELSEVIER_HOSTS,
    IEEE_DOI_PREFIXES,
    IEEE_HOSTS,
    PdfCandidate,
    PdfQuery,
    PdfRetrievalService,
    belongs_to_publisher,
    describe_failure,
    extract_arxiv_id,
    ieee_access_is_free,
    is_forbidden_url,
    order_candidates,
    titles_match,
)

PDF_BYTES = b"%PDF-1.7\nfake pdf body\n%%EOF"
LANDING_HTML = (
    b'<html><head><meta name="citation_pdf_url" '
    b'content="https://example.org/paper.pdf"></head><body>Landing</body></html>'
)


def make_service(handler) -> tuple[PdfRetrievalService, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PdfRetrievalService(client), client


def json_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


# ------------------------------------------------------------------ helpers


def test_extract_arxiv_id_from_doi():
    assert extract_arxiv_id("10.48550/arXiv.2301.04567") == "2301.04567"


def test_extract_arxiv_id_from_url():
    assert extract_arxiv_id(None, "https://arxiv.org/abs/2301.04567v2") == "2301.04567v2"


def test_extract_arxiv_id_old_style():
    assert extract_arxiv_id("https://arxiv.org/abs/cs/0112017") == "cs/0112017"


def test_extract_arxiv_id_returns_none_for_plain_doi():
    assert extract_arxiv_id("10.1145/3597503.3623301") is None


def test_titles_match_ignores_punctuation_and_case():
    assert titles_match(
        "Cooperative Awareness in CAVs: A Survey",
        "cooperative awareness in cavs - a survey",
    )


def test_titles_match_rejects_different_papers():
    assert not titles_match("A Survey of Platooning", "Reinforcement Learning for Traffic Lights")


def test_clean_doi_strips_url_prefix():
    assert PdfQuery(doi="https://doi.org/10.1145/ABC").clean_doi == "10.1145/abc"


# ------------------------------------------------------------------ resolvers


@pytest.mark.asyncio
async def test_arxiv_id_shortcut_downloads_without_api_calls():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "arxiv.org/pdf" in str(request.url):
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.48550/arXiv.2301.04567", title="Paper"))

    assert result.found is True
    assert result.provider == "arxiv"
    assert result.content == PDF_BYTES
    # Only the arXiv PDF was requested — no Unpaywall/OpenAlex round trips.
    assert calls == ["https://arxiv.org/pdf/2301.04567"]


@pytest.mark.asyncio
async def test_unpaywall_best_location_is_used(monkeypatch):
    monkeypatch.setattr(
        "src.services.pdf_retrieval_service.settings.oa_contact_email", "test@example.org"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.unpaywall.org" in url:
            assert "email=test%40example.org" in url
            return json_response(
                {
                    "best_oa_location": {
                        "url_for_pdf": "https://repo.example.org/oa.pdf",
                        "license": "cc-by",
                        "version": "publishedVersion",
                    }
                }
            )
        if url == "https://repo.example.org/oa.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/3597503", title="Paper"))

    assert result.found is True
    assert result.provider == "unpaywall"
    assert result.license == "cc-by"
    assert result.source_url == "https://repo.example.org/oa.pdf"


@pytest.mark.asyncio
async def test_unpaywall_skipped_without_contact_email(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        await service.retrieve(PdfQuery(doi="10.1145/3597503"))

    assert "api.unpaywall.org" not in seen


@pytest.mark.asyncio
async def test_openalex_falls_back_to_title_search(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.openalex.org/works?" in url:
            return json_response(
                {
                    "results": [
                        {"title": "A completely different paper", "best_oa_location": None},
                        {
                            "title": "Cooperative Perception for CAVs",
                            "best_oa_location": {
                                "pdf_url": "https://oa.example.org/cav.pdf",
                                "license": "cc-by-sa",
                            },
                        },
                    ]
                }
            )
        if url == "https://oa.example.org/cav.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(title="Cooperative Perception for CAVs"))

    assert result.found is True
    assert result.provider == "openalex"


@pytest.mark.asyncio
async def test_title_search_ignores_mismatched_results(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.openalex.org/works?" in url:
            return json_response(
                {
                    "results": [
                        {
                            "title": "An unrelated study on compilers",
                            "best_oa_location": {"pdf_url": "https://oa.example.org/wrong.pdf"},
                        }
                    ]
                }
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(title="Cooperative Perception for CAVs"))

    assert result.found is False
    assert "https://oa.example.org/wrong.pdf" not in [c.url for c in result.candidates]


@pytest.mark.asyncio
async def test_semantic_scholar_open_access_pdf(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.semanticscholar.org" in url:
            return json_response(
                {"title": "Paper", "openAccessPdf": {"url": "https://s2.example.org/p.pdf"}}
            )
        if url == "https://s2.example.org/p.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/3597503", title="Paper"))

    assert result.found is True
    assert result.provider == "semantic-scholar"


# ------------------------------------------------------------------ download


@pytest.mark.asyncio
async def test_download_follows_citation_pdf_url_on_html_landing_page():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.org/landing":
            return httpx.Response(200, content=LANDING_HTML)
        if str(request.url) == "https://example.org/paper.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        downloaded = await service.download_pdf("https://example.org/landing")

    assert downloaded is not None
    assert downloaded.content == PDF_BYTES
    # Provenance points at the file, not the landing page we came in through
    assert downloaded.url == "https://example.org/paper.pdf"


@pytest.mark.asyncio
async def test_download_rejects_html_without_pdf_link():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>Paywall</body></html>")

    service, client = make_service(handler)
    async with client:
        assert await service.download_pdf("https://example.org/paywalled") is None


@pytest.mark.asyncio
async def test_download_enforces_size_limit(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.pdf_retrieval_max_mb", 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_BYTES + b"x" * (2 * 1024 * 1024))

    service, client = make_service(handler)
    async with client:
        assert await service.download_pdf("https://example.org/huge.pdf") is None


@pytest.mark.asyncio
async def test_download_returns_none_on_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    service, client = make_service(handler)
    async with client:
        assert await service.download_pdf("https://example.org/p.pdf") is None


@pytest.mark.asyncio
async def test_failing_provider_does_not_abort_the_chain(monkeypatch):
    monkeypatch.setattr(
        "src.services.pdf_retrieval_service.settings.oa_contact_email", "test@example.org"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.unpaywall.org" in url:
            raise httpx.ConnectError("unpaywall down", request=request)
        if "api.openalex.org" in url:
            return json_response({"open_access": {"oa_url": "https://oa.example.org/fallback.pdf"}})
        if url == "https://oa.example.org/fallback.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/3597503", title="Paper"))

    assert result.found is True
    assert result.provider == "openalex"
    assert "unpaywall" in result.providers_tried


@pytest.mark.asyncio
async def test_no_metadata_yields_not_found(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery())

    assert result.found is False
    assert result.reason.startswith("No downloadable copy found")


# ------------------------------------------------- candidate ranking & reporting


def test_repository_copies_are_tried_before_publisher_copies():
    publisher = PdfCandidate(url="https://www.mdpi.com/x/pdf", provider="openalex")
    repo_landing = PdfCandidate(
        url="https://research.chalmers.se/publication/1",
        provider="openalex",
        kind="landing",
        is_repository=True,
    )
    repo_pdf = PdfCandidate(
        url="https://research.chalmers.se/file/1.pdf", provider="openalex", is_repository=True
    )

    assert order_candidates([publisher, repo_landing, repo_pdf]) == [
        repo_pdf,
        repo_landing,
        publisher,
    ]


def test_describe_failure_reports_a_blocked_open_access_copy():
    reason = describe_failure([PdfCandidate(url="https://www.mdpi.com/x/pdf", provider="openalex")])

    assert "www.mdpi.com" in reason
    assert "refused the download" in reason


def test_describe_failure_ignores_the_speculative_landing_probe():
    """A Scopus record page is not an open-access link and must not be reported as one."""
    probe = PdfCandidate(
        url="https://www.scopus.com/pages/publications/1",
        provider="landing-page",
        kind="landing",
        is_open_access=False,
    )

    assert describe_failure([probe]).startswith("No downloadable copy found")


@pytest.mark.asyncio
async def test_openalex_repository_landing_page_is_used_when_the_publisher_blocks(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.openalex.org" in url:
            return json_response(
                {
                    "locations": [
                        {
                            "is_oa": True,
                            "pdf_url": "https://www.mdpi.com/1/pdf",
                            "landing_page_url": "https://doi.org/10.3390/x",
                            "source": {"type": "journal"},
                        },
                        {
                            "is_oa": True,
                            "pdf_url": None,
                            "landing_page_url": "https://research.chalmers.se/publication/1",
                            "source": {"type": "repository"},
                        },
                    ]
                }
            )
        if url == "https://research.chalmers.se/publication/1":
            return httpx.Response(200, content=LANDING_HTML)
        if url == "https://example.org/paper.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        # The publisher answers automated requests with 403, as MDPI/Elsevier do.
        return httpx.Response(403, content=b"<html>blocked</html>")

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.3390/x"))

    assert result.found is True
    assert result.source_url == "https://example.org/paper.pdf"


@pytest.mark.asyncio
async def test_unpaywall_repository_landing_page_is_a_candidate(monkeypatch):
    monkeypatch.setattr(
        "src.services.pdf_retrieval_service.settings.oa_contact_email", "test@example.org"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.unpaywall.org" in url:
            return json_response(
                {
                    "best_oa_location": {
                        "url_for_pdf": None,
                        "url_for_landing_page": "https://repo.example.org/item/1",
                        "host_type": "repository",
                    }
                }
            )
        if url == "https://repo.example.org/item/1":
            return httpx.Response(200, content=LANDING_HTML)
        if url == "https://example.org/paper.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/1"))

    assert result.found is True
    assert result.provider == "unpaywall"


@pytest.mark.asyncio
async def test_index_landing_pages_are_never_probed(monkeypatch):
    """Scopus record pages redirect through an Elsevier login and never serve a PDF."""
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(
            PdfQuery(doi="10.1109/x", url="https://www.scopus.com/pages/publications/85065493613")
        )

    assert not any("scopus.com" in url for url in requested)
    assert result.reason.startswith("No downloadable copy found")


@pytest.mark.asyncio
async def test_resolve_candidates_deduplicates_repeated_locations(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.openalex.org" in str(request.url):
            location = {
                "is_oa": True,
                "pdf_url": "https://www.mdpi.com/1/pdf",
                "source": {"type": "journal"},
            }
            return json_response(
                {
                    "best_oa_location": location,
                    "primary_location": location,
                    "locations": [location, location],
                }
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        candidates, _ = await service.resolve_candidates(PdfQuery(doi="10.3390/x"))

    assert [c.url for c in candidates] == ["https://www.mdpi.com/1/pdf"]


@pytest.mark.asyncio
async def test_semantic_scholar_retries_once_after_a_429(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    monkeypatch.setattr("src.services.pdf_retrieval_service.SEMANTIC_SCHOLAR_BACKOFF_SECONDS", 0)
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.semanticscholar.org" in url:
            attempts.append(url)
            if len(attempts) == 1:
                return httpx.Response(429)
            return json_response(
                {"title": "Paper", "openAccessPdf": {"url": "https://s2.example.org/p.pdf"}}
            )
        if url == "https://s2.example.org/p.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/1", title="Paper"))

    assert len(attempts) == 2
    assert result.found is True
    assert result.provider == "semantic-scholar"


@pytest.mark.asyncio
async def test_index_hosts_listed_by_openalex_are_dropped(monkeypatch):
    """OpenAlex marks some Scopus record URLs as open-access locations; they are not."""
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if "api.openalex.org" in url:
            return json_response(
                {
                    "locations": [
                        {
                            "is_oa": True,
                            "landing_page_url": "http://www.scopus.com/inward/record.url?scp=1",
                            "source": {"type": "repository"},
                        }
                    ]
                }
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/x"))

    assert not any("scopus.com" in url for url in requested)
    assert result.open_access_urls() == []
    assert result.reason.startswith("No downloadable copy found")


@pytest.mark.asyncio
async def test_blocked_open_access_links_are_reported_for_manual_download(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.openalex.org" in str(request.url):
            return json_response(
                {
                    "locations": [
                        {
                            "is_oa": True,
                            "pdf_url": "https://www.mdpi.com/1/pdf",
                            "source": {"type": "journal"},
                        }
                    ]
                }
            )
        return httpx.Response(403, content=b"<html>bot wall</html>")

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.3390/x"))

    assert result.found is False
    assert result.open_access_urls() == ["https://www.mdpi.com/1/pdf"]


@pytest.mark.asyncio
async def test_semantic_scholar_api_key_is_sent_when_configured(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    monkeypatch.setattr(
        "src.services.pdf_retrieval_service.settings.semantic_scholar_api_key", "secret-key"
    )
    keys: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.semanticscholar.org" in url:
            keys.append(request.headers.get("x-api-key"))
            return json_response({"title": "Paper", "openAccessPdf": None})
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        await service.retrieve(PdfQuery(doi="10.1145/1", title="Paper"))

    assert keys and all(key == "secret-key" for key in keys)


@pytest.mark.asyncio
async def test_no_api_key_header_without_configuration(monkeypatch):
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.oa_contact_email", "")
    monkeypatch.setattr("src.services.pdf_retrieval_service.settings.semantic_scholar_api_key", "")
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.semanticscholar.org" in str(request.url):
            seen.append("x-api-key" in request.headers)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        await service.retrieve(PdfQuery(doi="10.1145/1", title="Paper"))

    assert seen and not any(seen)


# ---------------------------------------------------------------- publisher APIs


def configure_publishers(
    monkeypatch,
    *,
    elsevier: str = "",
    inst_token: str = "",
    ieee: str = "",
    contact_email: str = "",
) -> None:
    """Pin every credential the service reads, so a real .env cannot leak in."""
    for name, value in (
        ("oa_contact_email", contact_email),
        ("semantic_scholar_api_key", ""),
        ("elsevier_api_key", elsevier),
        ("elsevier_inst_token", inst_token),
        ("ieee_api_key", ieee),
    ):
        monkeypatch.setattr(f"src.services.pdf_retrieval_service.settings.{name}", value)


@pytest.mark.asyncio
async def test_elsevier_article_retrieval_uses_the_api_key(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key", inst_token="inst-token")
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.elsevier.com":
            seen["key"] = request.headers.get("x-els-apikey")
            seen["token"] = request.headers.get("x-els-insttoken")
            seen["accept"] = request.headers.get("accept")
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1016/j.trc.2023.104123", title="Paper"))

    assert result.found is True
    assert result.provider == "elsevier"
    assert seen == {
        "key": "els-key",
        "token": "inst-token",
        "accept": "application/pdf",
    }
    # Provenance keeps a link a reader can open, not the credentialed endpoint.
    assert result.source_url == "https://doi.org/10.1016/j.trc.2023.104123"


@pytest.mark.asyncio
async def test_elsevier_is_skipped_for_a_foreign_doi(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key")
    hosts: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/TITS.2023.1", title="Paper"))

    assert "api.elsevier.com" not in hosts
    # A resolver that did not apply must not be reported as checked.
    assert "elsevier" not in result.providers_tried


@pytest.mark.asyncio
async def test_elsevier_is_skipped_without_a_key(monkeypatch):
    configure_publishers(monkeypatch)
    hosts: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1016/j.trc.2023.1", title="Paper"))

    assert "api.elsevier.com" not in hosts
    assert "elsevier" not in result.providers_tried


@pytest.mark.asyncio
async def test_paywalled_elsevier_article_is_not_reported_as_open_access(monkeypatch):
    """A refused download must not turn the API endpoint into an 'open copy' link."""
    configure_publishers(monkeypatch, elsevier="els-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1016/j.trc.2023.1", title="Paper"))

    assert result.found is False
    assert result.open_access_urls() == []
    assert "els-key" not in (result.reason or "")


@pytest.mark.asyncio
async def test_scopus_fills_in_a_missing_doi(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key", contact_email="test@example.org")
    unpaywall_dois: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.url.host == "api.elsevier.com" and "/search/scopus" in url:
            assert request.headers.get("x-els-apikey") == "els-key"
            return json_response(
                {
                    "search-results": {
                        "entry": [
                            {"dc:title": "Some other paper", "prism:doi": "10.1000/wrong"},
                            {
                                "dc:title": "Cooperative Awareness in CAVs",
                                "prism:doi": "10.1109/TITS.2023.7",
                            },
                        ]
                    }
                }
            )
        if "api.unpaywall.org" in url:
            unpaywall_dois.append(request.url.path.removeprefix("/v2/"))
            return json_response(
                {"best_oa_location": {"url_for_pdf": "https://repo.example.org/oa.pdf"}}
            )
        if url == "https://repo.example.org/oa.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(title="Cooperative Awareness in CAVs"))

    assert result.found is True
    assert result.providers_tried[0] == "scopus"
    assert unpaywall_dois == ["10.1109/tits.2023.7"]


@pytest.mark.asyncio
async def test_scopus_lookup_is_skipped_when_a_doi_is_known(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        await service.retrieve(PdfQuery(doi="10.1145/1", title="Paper"))

    assert "/content/search/scopus" not in paths


@pytest.mark.asyncio
async def test_scopus_ignores_a_different_paper(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if "/search/scopus" in str(request.url):
            return json_response(
                {
                    "search-results": {
                        "entry": [
                            {"dc:title": "Traffic light control with RL", "prism:doi": "10.1000/x"}
                        ]
                    }
                }
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(title="Cooperative Awareness in CAVs"))

    assert result.found is False
    assert "scopus" in result.providers_tried


@pytest.mark.asyncio
async def test_ieee_open_access_article_is_downloaded(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            assert request.url.params["apikey"] == "ieee-key"
            assert request.url.params["doi"] == "10.1109/tits.2023.7"
            return json_response(
                {
                    "articles": [
                        {
                            "title": "Paper",
                            "access_type": "OPEN_ACCESS",
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7",
                            "html_url": "https://ieeexplore.ieee.org/document/7",
                        }
                    ]
                }
            )
        if request.url.host == "ieeexplore.ieee.org":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/TITS.2023.7", title="Paper"))

    assert result.found is True
    assert result.provider == "ieee"
    assert result.license == "IEEE Open Access"
    assert result.source_url == "https://ieeexplore.ieee.org/document/7"


@pytest.mark.asyncio
async def test_ieee_open_access_link_is_offered_when_xplore_blocks_the_download(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            return json_response(
                {
                    "articles": [
                        {
                            "title": "Paper",
                            "access_type": "Open Access",
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7",
                            "html_url": "https://ieeexplore.ieee.org/document/7",
                        }
                    ]
                }
            )
        return httpx.Response(418, content=b"<html>bot wall</html>")

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/TITS.2023.7", title="Paper"))

    assert result.found is False
    assert result.open_access_urls() == ["https://ieeexplore.ieee.org/document/7"]
    assert "ieeexplore.ieee.org" in result.reason


@pytest.mark.asyncio
async def test_ieee_locked_article_is_not_reported_as_open_access(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            return json_response(
                {
                    "articles": [
                        {
                            "title": "Paper",
                            "access_type": "Locked",
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7",
                            "html_url": "https://ieeexplore.ieee.org/document/7",
                        }
                    ]
                }
            )
        return httpx.Response(403)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/TITS.2023.7", title="Paper"))

    assert result.found is False
    assert result.open_access_urls() == []
    assert result.reason.startswith("No downloadable copy found")


@pytest.mark.asyncio
async def test_ieee_downloads_from_the_getpdf_endpoint_not_the_stamp_page(monkeypatch):
    """The API's pdf_url is an HTML page with the PDF in an iframe."""
    configure_publishers(monkeypatch, ieee="ieee-key")
    downloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            return json_response(
                {
                    "articles": [
                        {
                            "title": "Paper",
                            "article_number": "9305204",
                            "access_type": "OPEN_ACCESS",
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9305204",
                            "html_url": "https://ieeexplore.ieee.org/document/9305204",
                        }
                    ]
                }
            )
        if request.url.host == "ieeexplore.ieee.org":
            downloads.append(request.url.path)
            if request.url.path == "/stampPDF/getPDF.jsp":
                assert request.url.params["arnumber"] == "9305204"
                return httpx.Response(200, content=PDF_BYTES)
            return httpx.Response(200, content=b"<html><iframe src='x.pdf'></iframe></html>")
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/ACCESS.2020.3046999"))

    assert result.found is True
    assert result.provider == "ieee"
    assert downloads == ["/stampPDF/getPDF.jsp"]
    assert result.source_url == "https://ieeexplore.ieee.org/document/9305204"


@pytest.mark.asyncio
async def test_ieee_creative_commons_article_counts_as_open_access(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            return json_response(
                {
                    "articles": [
                        {
                            "title": "Paper",
                            "article_number": "9305204",
                            "access_type": (
                                "CCBY - IEEE is not the copyright holder of this material. "
                                "Please follow the instructions via "
                                "https://creativecommons.org/licenses/by/4.0/ to obtain "
                                "full-text articles and stipulations in the API documentation."
                            ),
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9305204",
                            "html_url": "https://ieeexplore.ieee.org/document/9305204",
                        }
                    ]
                }
            )
        if request.url.host == "ieeexplore.ieee.org":
            # Xplore refuses the bot: the record must still count as open access
            return httpx.Response(403)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1109/ACCESS.2020.3046999"))

    assert result.found is False
    ieee = [c for c in result.candidates if c.provider == "ieee"]
    assert ieee and ieee[0].is_open_access is True
    assert ieee[0].license == "CC BY"
    assert result.open_access_urls() == ["https://ieeexplore.ieee.org/document/9305204"]


def test_ieee_access_is_free_reads_only_the_first_token():
    assert ieee_access_is_free("OPEN_ACCESS") == (True, "IEEE Open Access")
    assert ieee_access_is_free("Ephemera") == (True, "IEEE Open Access")
    assert ieee_access_is_free("CCBY - IEEE is not the copyright holder of this material.") == (
        True,
        "CC BY",
    )
    assert ieee_access_is_free("CCBYNCND - ...") == (True, "CC BYNCND")
    assert ieee_access_is_free("LOCKED") == (False, None)
    assert ieee_access_is_free(None) == (False, None)


@pytest.mark.asyncio
async def test_ieee_searches_by_title_when_the_record_has_no_doi(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")
    titles: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ieeexploreapi.ieee.org":
            titles.append(request.url.params["article_title"])
            return json_response(
                {
                    "articles": [
                        {
                            "title": "A totally different paper",
                            "access_type": "OPEN_ACCESS",
                            "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9",
                        }
                    ]
                }
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(title="Cooperative Awareness in CAVs"))

    assert titles == ["Cooperative Awareness in CAVs"]
    # The title did not match, so the mismatched article is never downloaded.
    assert result.found is False


@pytest.mark.asyncio
async def test_ieee_is_skipped_for_a_foreign_doi(monkeypatch):
    configure_publishers(monkeypatch, ieee="ieee-key")
    hosts: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi="10.1145/3597503", title="Paper"))

    assert "ieeexploreapi.ieee.org" not in hosts
    assert "ieee" not in result.providers_tried


# ACM's terms of use forbid automated access to the ACM DL; a violation costs the
# institution its downloading rights. The service swallows transport errors, so
# the tests record every request and check the list afterwards.

ACM_DOI = "10.1145/3597503.3623301"


def test_is_forbidden_url_covers_the_acm_dl_and_acm_doi_links():
    assert is_forbidden_url("https://dl.acm.org/doi/pdf/10.1145/1")
    assert is_forbidden_url("https://acm.org/")
    assert is_forbidden_url("https://doi.org/10.1145/3597503")
    assert is_forbidden_url("https://dx.doi.org/10.1145%2F3597503")
    assert not is_forbidden_url("https://doi.org/10.1109/TITS.2023.1")
    assert not is_forbidden_url("https://notacm.org/paper.pdf")


@pytest.mark.asyncio
async def test_acm_dl_is_never_requested(monkeypatch):
    """Index links, the source URL, a DOI link, a redirect and a citation_pdf_url."""
    configure_publishers(monkeypatch, contact_email="me@example.org")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "api.unpaywall.org":
            return json_response(
                {
                    "best_oa_location": {"url_for_pdf": f"https://dl.acm.org/doi/pdf/{ACM_DOI}"},
                    "oa_locations": [
                        {"url_for_pdf": f"https://doi.org/{ACM_DOI}"},
                        {"url_for_pdf": "https://mirror.example.org/redirect"},
                        {"url_for_landing_page": "https://repo.example.org/landing"},
                    ],
                }
            )
        if request.url.path == "/redirect":
            return httpx.Response(
                302, headers={"Location": f"https://dl.acm.org/doi/pdf/{ACM_DOI}"}
            )
        if request.url.path == "/landing":
            return httpx.Response(
                200,
                content=b'<meta name="citation_pdf_url" content="https://dl.acm.org/doi/pdf/x">',
            )
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(
            PdfQuery(doi=ACM_DOI, title="Paper", url=f"https://dl.acm.org/doi/abs/{ACM_DOI}")
        )

    assert result.found is False
    assert not [url for url in seen if is_forbidden_url(url)]
    # The source's own ACM DL page is handed to the user instead.
    assert f"https://dl.acm.org/doi/abs/{ACM_DOI}" in result.open_access_urls()
    assert "https://mirror.example.org/redirect" in seen
    assert "https://repo.example.org/landing" in seen


@pytest.mark.asyncio
async def test_acm_paper_imported_from_scopus_gets_an_acm_dl_link(monkeypatch):
    """Scopus stores its own record page as the URL, which is never offered."""
    configure_publishers(monkeypatch)
    service, client = make_service(lambda request: httpx.Response(404))
    async with client:
        result = await service.retrieve(
            PdfQuery(doi=ACM_DOI, url="https://www.scopus.com/inward/record.uri?eid=2-s2.0-1")
        )

    assert result.open_access_urls() == [f"https://dl.acm.org/doi/{ACM_DOI}"]
    assert "ACM forbids automated downloads" in (result.reason or "")


@pytest.mark.asyncio
async def test_acm_paper_is_downloaded_from_a_copy_outside_the_acm_dl(monkeypatch):
    configure_publishers(monkeypatch, contact_email="me@example.org")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.unpaywall.org":
            return json_response(
                {"best_oa_location": {"url_for_pdf": "https://repo.example.org/acm-paper.pdf"}}
            )
        if str(request.url) == "https://repo.example.org/acm-paper.pdf":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        result = await service.retrieve(PdfQuery(doi=ACM_DOI, title="Paper"))

    assert result.found is True
    assert result.provider == "unpaywall"


@pytest.mark.asyncio
async def test_resolve_candidates_reports_public_links_not_api_endpoints(monkeypatch):
    configure_publishers(monkeypatch, elsevier="els-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    service, client = make_service(handler)
    async with client:
        candidates, providers = await service.resolve_candidates(
            PdfQuery(doi="10.1016/j.trc.2023.1", title="Paper")
        )

    elsevier = [c for c in candidates if c.provider == "elsevier"]
    assert elsevier and elsevier[0].link == "https://doi.org/10.1016/j.trc.2023.1"
    assert "elsevier" in providers


def test_belongs_to_publisher_matches_on_the_source_url():
    query = PdfQuery(url="https://www.sciencedirect.com/science/article/pii/S1")

    assert belongs_to_publisher(query, ELSEVIER_DOI_PREFIXES, ELSEVIER_HOSTS)
    assert not belongs_to_publisher(query, IEEE_DOI_PREFIXES, IEEE_HOSTS)


def test_describe_failure_names_the_providers_that_were_asked():
    reason = describe_failure([], ["unpaywall", "ieee", "landing-page"])

    assert reason == "No downloadable copy found (checked Unpaywall, IEEE Xplore)"
