"""Open-access PDF retrieval for bibliographic records.

Bulk imports (Scopus, IEEE, ACM) only give us metadata. This service tries to
turn that metadata into an actual PDF by asking the open scholarly APIs in
order and downloading the first candidate that really serves a PDF:

1. arXiv, when an arXiv id is already visible in the DOI or the source URL
2. ScienceDirect, for Elsevier DOIs (requires ``elsevier_api_key``)
3. Unpaywall (requires ``oa_contact_email``)
4. OpenAlex
5. Semantic Scholar
6. arXiv title search
7. IEEE Xplore, for IEEE DOIs (requires ``ieee_api_key``)
8. the publisher landing page (``citation_pdf_url`` meta tag)

The publisher steps are skipped unless their credential is configured, and each
only runs for records that belong to that publisher, so the Scopus/ScienceDirect
key is not spent on an IEEE paper. Records without a DOI first get one looked up
through the Scopus Search API when an Elsevier key is present, which helps every
later provider.

Nothing here bypasses a paywall or a publisher login: the publisher APIs are
called with the credentials you supply, and Elsevier serves full text only
for its open-access articles unless an institutional token is configured.
The ACM DL is never contacted at all (see ``is_forbidden_url``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from urllib.parse import quote, unquote, urlparse

import httpx
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup

from src.core.config import settings

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"

# Polite rate limits — these are shared APIs and most of them ask for a
# specific ceiling in their docs. arXiv asks for one request every three seconds.
_LIMITER_RATES: dict[str, tuple[float, float]] = {
    "unpaywall": (5, 1),
    "openalex": (5, 1),
    "semantic-scholar": (1, 1),
    "arxiv": (1, 3),
    "elsevier": (2, 1),
    "ieee": (2, 1),
    "download": (4, 1),
}
_limiters: dict[tuple[str, int], AsyncLimiter] = {}


def _limiter(name: str) -> AsyncLimiter:
    """Return the limiter for ``name``, one instance per event loop.

    aiolimiter warns (and misbehaves) when a limiter is shared across loops,
    which happens in tests and anywhere a fresh loop is spun up.
    """
    key = (name, id(asyncio.get_running_loop()))
    limiter = _limiters.get(key)
    if limiter is None:
        limiter = AsyncLimiter(*_LIMITER_RATES[name])
        _limiters[key] = limiter
    return limiter


_ARXIV_NEW_ID = re.compile(r"(?:arxiv[.:/ ]|abs/|pdf/)(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_ARXIV_OLD_ID = re.compile(r"(?:abs/|pdf/)([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.IGNORECASE)
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Below this similarity a title-based hit is treated as a different paper.
TITLE_MATCH_THRESHOLD = 0.9

# How long to wait out a Semantic Scholar 429 before the single retry.
SEMANTIC_SCHOLAR_BACKOFF_SECONDS = 2.0

# Bibliographic indexes and social sites. Their record pages never carry a PDF,
# and probing them just burns a request on a login redirect.
INDEX_HOSTS = frozenset(
    {
        "scopus.com",
        "webofscience.com",
        "engineeringvillage.com",
        "dimensions.ai",
        "lens.org",
        "semanticscholar.org",
        "researchgate.net",
        "academia.edu",
    }
)


# DOI prefixes and hosts that identify a record as belonging to a publisher we
# have an API for. Gating on these keeps a limited key (IEEE grants 200 calls a
# day) from being spent on papers the publisher does not carry.
ELSEVIER_DOI_PREFIXES = ("10.1016", "10.1006", "10.1053", "10.1067", "10.1078", "10.1157")
ELSEVIER_HOSTS = ("sciencedirect.com", "elsevier.com")
IEEE_DOI_PREFIXES = ("10.1109", "10.23919")
IEEE_HOSTS = ("ieee.org",)

# The ACM DL's terms of use forbid "scripts, spiders or other robotic activity
# to automatically download articles or harvest metadata", on pain of the
# institution losing its downloading rights. No request may reach it, not even
# through a DOI link, which redirects there.
ACM_DOI_PREFIXES = ("10.1145",)
ACM_HOSTS = ("acm.org",)

# IEEE reports how a document may be read. These two are free for everyone;
# Creative Commons papers come back as "CCBY - IEEE is not the copyright
# holder ..." and are free as well, so any CC* type also counts (see
# ieee_access_is_free).
IEEE_FREE_ACCESS_TYPES = frozenset({"OPEN_ACCESS", "EPHEMERA"})
# The API's ``pdf_url`` is an HTML page that embeds the PDF in an iframe; this
# endpoint serves the file itself for the same article number.
IEEE_PDF_URL = "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}"

# Human-readable names for the failure message.
PROVIDER_LABELS = {
    "arxiv-id": "arXiv",
    "arxiv-search": "arXiv",
    "unpaywall": "Unpaywall",
    "openalex": "OpenAlex",
    "semantic-scholar": "Semantic Scholar",
    "scopus": "Scopus",
    "elsevier": "Elsevier",
    "ieee": "IEEE Xplore",
}


@dataclass(frozen=True)
class PdfQuery:
    """Everything we know about a source that could lead to its PDF."""

    doi: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    url: str | None = None

    @property
    def clean_doi(self) -> str | None:
        if not self.doi:
            return None
        doi = _DOI_PREFIX.sub("", self.doi.strip()).strip()
        return doi.lower() or None

    @property
    def clean_title(self) -> str | None:
        title = (self.title or "").strip()
        return title or None


@dataclass
class PdfCandidate:
    """A URL that might serve the PDF, plus where we learned about it."""

    url: str
    provider: str
    license: str | None = None
    version: str | None = None
    #: A direct file link, or a landing page we still have to dig the PDF out of.
    kind: str = "pdf"
    #: Repository copies are reachable; publisher copies are usually bot-blocked.
    is_repository: bool = False
    #: False for the speculative probe of the source's own URL, which is not a
    #: known open-access copy and must not be reported as one.
    is_open_access: bool = True
    #: Extra request headers, e.g. a publisher API key. Kept out of ``url`` so
    #: credentials never end up in stored provenance or in a link we show.
    headers: dict[str, str] = field(default_factory=dict)
    #: The page a human should open when the download fails. Publisher API
    #: endpoints are useless to a reader, so they carry the DOI link here.
    public_url: str | None = None

    @property
    def link(self) -> str:
        """The URL worth showing to a person."""
        return self.public_url or self.url

    def rank(self) -> tuple[int, int]:
        """Sort key: repositories before publishers, direct PDFs before landing pages."""
        return (0 if self.is_repository else 1, 0 if self.kind == "pdf" else 1)


def order_candidates(candidates: list[PdfCandidate]) -> list[PdfCandidate]:
    """Try the copies that actually let us download first.

    Publishers (MDPI, Elsevier, IEEE, ...) answer automated requests with 403
    even for gold open-access papers, while institutional repositories serve
    the same file without complaint — so repository copies go first.
    """
    return sorted(candidates, key=lambda c: c.rank())


@dataclass
class DownloadedPdf:
    """PDF bytes plus the URL they were finally served from."""

    content: bytes
    url: str


@dataclass
class PdfRetrievalResult:
    found: bool
    provider: str | None = None
    source_url: str | None = None
    license: str | None = None
    content: bytes | None = None
    reason: str | None = None
    providers_tried: list[str] = field(default_factory=list)
    candidates: list[PdfCandidate] = field(default_factory=list)

    def open_access_urls(self) -> list[str]:
        """Known open copies we could not download — worth offering as links."""
        urls: list[str] = []
        for candidate in self.candidates:
            if candidate.is_open_access and candidate.link not in urls:
                urls.append(candidate.link)
        return urls


def is_index_host(url: str) -> bool:
    """True for bibliographic indexes that only ever serve a record page.

    OpenAlex sometimes lists a Scopus record URL as an open-access location; it
    is neither downloadable nor open, so it must not cost a request or show up
    in a failure message.
    """
    host = urlparse(url).hostname or ""
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in INDEX_HOSTS)


def is_forbidden_url(url: str) -> bool:
    """True for the ACM DL, and for DOI links that redirect to it."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if any(host == h or host.endswith(f".{h}") for h in ACM_HOSTS):
        return True
    if host == "doi.org" or host.endswith(".doi.org"):
        doi = unquote(parsed.path).lstrip("/").lower()
        return doi.startswith(tuple(f"{prefix}/" for prefix in ACM_DOI_PREFIXES))
    return False


async def _refuse_forbidden_request(request: httpx.Request) -> None:
    """httpx request hook; it runs before every request, redirect hops included."""
    if is_forbidden_url(str(request.url)):
        raise httpx.RequestError(f"Not requesting {request.url}: ACM forbids automated access")


def ieee_access_is_free(access_type: str | None) -> tuple[bool, str | None]:
    """Whether an IEEE ``access_type`` means anyone may read the paper.

    Returns ``(is_free, license)``. Only the part before " - " matters: the
    API pads Creative Commons types with a sentence of legal text.
    """
    head = str(access_type or "").split(" - ", 1)[0].strip().upper().replace(" ", "_")
    if head in IEEE_FREE_ACCESS_TYPES:
        return True, "IEEE Open Access"
    if head.startswith("CC"):
        # "CCBY" -> "CC BY", "CCBYNCND" -> "CC BYNCND"
        return True, "CC " + head[2:] if len(head) > 2 else "CC"
    return False, None


def belongs_to_publisher(query: PdfQuery, doi_prefixes: tuple[str, ...], hosts: tuple[str, ...]) -> bool:
    """True when the record is published by the given publisher.

    Matched on the DOI prefix, or on the source URL for records imported without
    a DOI. Anything else must not cost a call against a metered publisher key.
    """
    doi = query.clean_doi
    if doi and doi.startswith(tuple(f"{prefix}/" for prefix in doi_prefixes)):
        return True
    host = urlparse((query.url or "").strip()).hostname or ""
    return any(host == h or host.endswith(f".{h}") for h in hosts)


def describe_failure(
    candidates: list[PdfCandidate], providers_tried: list[str] | None = None
) -> str:
    """Explain a miss in terms the person screening sources can act on."""
    open_access = [c for c in candidates if c.is_open_access]
    if not open_access:
        checked = []
        for provider in providers_tried or []:
            label = PROVIDER_LABELS.get(provider)
            if label and label not in checked:
                checked.append(label)
        listed = ", ".join(checked) or "Unpaywall, OpenAlex, Semantic Scholar, arXiv"
        return f"No downloadable copy found (checked {listed})"

    if all(is_forbidden_url(c.link) for c in open_access):
        return (
            "No copy outside the ACM DL found, and ACM forbids automated downloads. "
            "Open the ACM DL link, download the PDF yourself and upload it"
        )

    hosts = sorted({urlparse(c.link).hostname or "?" for c in open_access})
    return (
        f"Open-access copy listed at {', '.join(hosts[:3])} but the host refused the download. "
        "Upload the PDF manually"
    )


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def titles_match(a: str | None, b: str | None, threshold: float = TITLE_MATCH_THRESHOLD) -> bool:
    """Guard against title searches returning a different paper."""
    if not a or not b:
        return False
    left, right = normalize_title(a), normalize_title(b)
    if not left or not right:
        return False
    if left == right:
        return True
    return SequenceMatcher(None, left, right).ratio() >= threshold


def extract_arxiv_id(*values: str | None) -> str | None:
    """Pull an arXiv id out of a DOI (10.48550/arXiv.1234.5678) or a URL."""
    for value in values:
        if not value:
            continue
        match = _ARXIV_NEW_ID.search(value)
        if match:
            return match.group(1) + (match.group(2) or "")
        match = _ARXIV_OLD_ID.search(value)
        if match:
            return match.group(1) + (match.group(2) or "")
    return None


class PdfRetrievalService:
    """Resolves and downloads open-access PDFs.

    The httpx client is injected so tests can supply a MockTransport.
    """

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        hooks = client.event_hooks
        client.event_hooks = {**hooks, "request": [*hooks["request"], _refuse_forbidden_request]}
        self.contact_email = (settings.oa_contact_email or "").strip()
        self.semantic_scholar_api_key = (settings.semantic_scholar_api_key or "").strip()
        self.elsevier_api_key = (settings.elsevier_api_key or "").strip()
        self.elsevier_inst_token = (settings.elsevier_inst_token or "").strip()
        self.ieee_api_key = (settings.ieee_api_key or "").strip()
        self.max_bytes = settings.pdf_retrieval_max_mb * 1024 * 1024

    # ---------------------------------------------------------------- public

    async def retrieve(self, query: PdfQuery) -> PdfRetrievalResult:
        """Walk the providers in order and return the first real PDF.

        Downloads are attempted per provider rather than after collecting every
        candidate, so a source resolved by the first provider costs one API call
        instead of five — which matters across a few hundred imported records.
        """
        query, providers_tried = await self._with_looked_up_doi(query)
        all_candidates: list[PdfCandidate] = []
        seen: set[str] = set()

        for name, resolver in self._resolvers():
            try:
                candidates = await resolver(query)
            except Exception as exc:  # a flaky provider must not kill the batch
                logger.warning("PDF resolver %s failed: %s", name, exc)
                providers_tried.append(name)
                continue

            # None means the resolver does not cover this record. It never
            # asked, so it must not be reported as having been checked.
            if candidates is None:
                continue
            providers_tried.append(name)

            for candidate in candidates:
                if candidate.url in seen or is_index_host(candidate.url):
                    continue
                seen.add(candidate.url)
                if is_forbidden_url(candidate.url):
                    # Never fetched; offered as a link for manual download instead.
                    all_candidates.append(replace(candidate, is_open_access=True))
                    continue
                all_candidates.append(candidate)

                downloaded = await self.download_pdf(candidate.url, candidate.headers)
                if downloaded:
                    return PdfRetrievalResult(
                        found=True,
                        provider=candidate.provider,
                        # Publisher API endpoints are no use to a reader, so
                        # provenance keeps the public link where there is one.
                        source_url=candidate.public_url or downloaded.url,
                        license=candidate.license,
                        content=downloaded.content,
                        providers_tried=providers_tried,
                        candidates=all_candidates,
                    )

        # Scopus imports carry a Scopus record URL, so build the ACM page from the DOI.
        doi = query.clean_doi
        if doi and doi.startswith(tuple(f"{prefix}/" for prefix in ACM_DOI_PREFIXES)):
            all_candidates.append(
                PdfCandidate(url=f"https://dl.acm.org/doi/{doi}", provider="acm", kind="landing")
            )

        return PdfRetrievalResult(
            found=False,
            reason=describe_failure(all_candidates, providers_tried),
            providers_tried=providers_tried,
            candidates=all_candidates,
        )

    def _resolvers(self):
        """The provider chain, cheapest and most likely to succeed first.

        Publisher APIs are only in the chain when their key is configured; the
        resolver itself then decides whether the record is one of theirs.
        """
        resolvers = [("arxiv-id", self._from_arxiv_id)]
        if self.elsevier_api_key:
            resolvers.append(("elsevier", self._from_elsevier))
        resolvers += [
            ("unpaywall", self._from_unpaywall),
            ("openalex", self._from_openalex),
            ("semantic-scholar", self._from_semantic_scholar),
            ("arxiv-search", self._from_arxiv_search),
        ]
        if self.ieee_api_key:
            resolvers.append(("ieee", self._from_ieee))
        resolvers.append(("landing-page", self._from_landing_page))
        return tuple(resolvers)

    async def resolve_candidates(self, query: PdfQuery) -> tuple[list[PdfCandidate], list[str]]:
        """Ask every provider in order, collecting PDF URLs without downloading."""
        candidates: list[PdfCandidate] = []
        query, providers_tried = await self._with_looked_up_doi(query)

        for name, resolver in self._resolvers():
            try:
                found = await resolver(query)
            except Exception as exc:  # a flaky provider must not kill the batch
                logger.warning("PDF resolver %s failed: %s", name, exc)
                providers_tried.append(name)
                continue

            if found is None:
                continue
            providers_tried.append(name)
            candidates.extend(found)

        seen: set[str] = set()
        unique: list[PdfCandidate] = []
        for candidate in candidates:
            if candidate.url in seen or is_index_host(candidate.url):
                continue
            seen.add(candidate.url)
            unique.append(candidate)

        return unique, providers_tried

    async def download_pdf(
        self, url: str, headers: dict[str, str] | None = None, _depth: int = 0
    ) -> DownloadedPdf | None:
        """Download a URL and return the bytes only if they really are a PDF.

        Open-access links frequently point at a landing page instead of the
        file, so one HTML response is followed via its ``citation_pdf_url``.
        """
        try:
            async with (
                _limiter("download"),
                self.client.stream(
                    "GET",
                    url,
                    headers={**self._headers(accept="application/pdf,*/*"), **(headers or {})},
                    follow_redirects=True,
                    timeout=settings.pdf_retrieval_timeout_seconds,
                ) as response,
            ):
                if response.status_code >= 400:
                    logger.info("PDF download %s returned %s", url, response.status_code)
                    return None

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self.max_bytes:
                        logger.info("PDF at %s exceeds %s bytes, skipping", url, self.max_bytes)
                        return None
                    chunks.append(chunk)
                data = b"".join(chunks)
        except Exception as exc:
            logger.info("PDF download %s failed: %s", url, exc)
            return None

        if data.startswith(PDF_MAGIC):
            return DownloadedPdf(content=data, url=url)

        if _depth == 0:
            embedded = self._find_citation_pdf_url(data)
            if embedded and embedded != url:
                # Same publisher, so any credential still applies.
                return await self.download_pdf(embedded, headers, _depth=1)

        logger.info("PDF download %s returned a non-PDF response", url)
        return None

    # ------------------------------------------------------------- resolvers

    async def _from_arxiv_id(self, query: PdfQuery) -> list[PdfCandidate]:
        arxiv_id = extract_arxiv_id(query.doi, query.url)
        if not arxiv_id:
            return []
        return [
            PdfCandidate(
                url=f"https://arxiv.org/pdf/{arxiv_id}",
                provider="arxiv",
                license="arXiv (see paper)",
            )
        ]

    async def _from_unpaywall(self, query: PdfQuery) -> list[PdfCandidate]:
        doi = query.clean_doi
        if not doi:
            return []
        if not self.contact_email:
            logger.info("Skipping Unpaywall: OA_CONTACT_EMAIL is not configured")
            return []

        async with _limiter("unpaywall"):
            response = await self.client.get(
                f"https://api.unpaywall.org/v2/{quote(doi, safe='')}",
                params={"email": self.contact_email},
                headers=self._headers(),
                timeout=settings.pdf_retrieval_timeout_seconds,
            )
        if response.status_code != 200:
            return []

        payload = response.json()
        locations = []
        best = payload.get("best_oa_location")
        if best:
            locations.append(best)
        locations.extend(payload.get("oa_locations") or [])

        candidates: list[PdfCandidate] = []
        for location in locations:
            if not isinstance(location, dict):
                continue
            is_repository = location.get("host_type") == "repository"
            for url, kind in (
                (location.get("url_for_pdf"), "pdf"),
                (location.get("url_for_landing_page") or location.get("url"), "landing"),
            ):
                if not url:
                    continue
                candidates.append(
                    PdfCandidate(
                        url=url,
                        provider="unpaywall",
                        license=location.get("license"),
                        version=location.get("version"),
                        kind=kind,
                        is_repository=is_repository,
                    )
                )
        return order_candidates(candidates)

    async def _from_openalex(self, query: PdfQuery) -> list[PdfCandidate]:
        work = await self._openalex_work(query)
        if not work:
            return []

        candidates: list[PdfCandidate] = []
        locations = [work.get("best_oa_location"), work.get("primary_location")]
        locations.extend(work.get("locations") or [])

        for location in locations:
            if not isinstance(location, dict):
                continue
            is_repository = ((location.get("source") or {}) or {}).get("type") == "repository"

            if location.get("pdf_url"):
                candidates.append(
                    PdfCandidate(
                        url=location["pdf_url"],
                        provider="openalex",
                        license=location.get("license"),
                        version=location.get("version"),
                        kind="pdf",
                        is_repository=is_repository,
                    )
                )

            # Repositories usually list only a landing page; the PDF is behind
            # its citation_pdf_url meta tag, which download_pdf follows.
            if location.get("is_oa") and location.get("landing_page_url"):
                candidates.append(
                    PdfCandidate(
                        url=location["landing_page_url"],
                        provider="openalex",
                        license=location.get("license"),
                        version=location.get("version"),
                        kind="landing",
                        is_repository=is_repository,
                    )
                )

        oa_url = (work.get("open_access") or {}).get("oa_url")
        if oa_url:
            candidates.append(PdfCandidate(url=oa_url, provider="openalex", kind="landing"))
        return order_candidates(candidates)

    async def _openalex_work(self, query: PdfQuery) -> dict | None:
        params: dict[str, str] = {}
        if self.contact_email:
            params["mailto"] = self.contact_email

        doi = query.clean_doi
        if doi:
            async with _limiter("openalex"):
                response = await self.client.get(
                    f"https://api.openalex.org/works/doi:{quote(doi, safe='')}",
                    params=params,
                    headers=self._headers(),
                    timeout=settings.pdf_retrieval_timeout_seconds,
                )
            if response.status_code == 200:
                return response.json()

        title = query.clean_title
        if not title:
            return None

        search_params = dict(params)
        search_params["filter"] = f"title.search:{title}"
        search_params["per-page"] = "5"
        async with _limiter("openalex"):
            response = await self.client.get(
                "https://api.openalex.org/works",
                params=search_params,
                headers=self._headers(),
                timeout=settings.pdf_retrieval_timeout_seconds,
            )
        if response.status_code != 200:
            return None

        for work in response.json().get("results") or []:
            if titles_match(title, work.get("title") or work.get("display_name")):
                return work
        return None

    async def _from_semantic_scholar(self, query: PdfQuery) -> list[PdfCandidate]:
        paper = await self._semantic_scholar_paper(query)
        if not paper:
            return []

        open_pdf = paper.get("openAccessPdf") or {}
        url = open_pdf.get("url")
        if not url:
            return []
        return [
            PdfCandidate(
                url=url,
                provider="semantic-scholar",
                license=open_pdf.get("license") or paper.get("license"),
                kind="pdf",
            )
        ]

    async def _semantic_scholar_get(self, url: str, params: dict) -> httpx.Response | None:
        """GET with one backoff retry — the unauthenticated API 429s readily."""
        headers = self._headers()
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        for attempt in range(2):
            async with _limiter("semantic-scholar"):
                response = await self.client.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=settings.pdf_retrieval_timeout_seconds,
                )
            if response.status_code != 429:
                return response
            if attempt == 0:
                await asyncio.sleep(SEMANTIC_SCHOLAR_BACKOFF_SECONDS)

        logger.info("Semantic Scholar is rate limiting; skipping this record")
        return None

    async def _semantic_scholar_paper(self, query: PdfQuery) -> dict | None:
        fields = "title,openAccessPdf,externalIds"
        doi = query.clean_doi
        if doi:
            response = await self._semantic_scholar_get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{quote(doi, safe='')}",
                {"fields": fields},
            )
            if response is not None and response.status_code == 200:
                return response.json()

        title = query.clean_title
        if not title:
            return None

        response = await self._semantic_scholar_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": title, "fields": fields, "limit": 5},
        )
        if response is None or response.status_code != 200:
            return None

        for paper in response.json().get("data") or []:
            if titles_match(title, paper.get("title")):
                return paper
        return None

    async def _from_arxiv_search(self, query: PdfQuery) -> list[PdfCandidate]:
        title = query.clean_title
        if not title:
            return []

        escaped = title.replace('"', "")
        async with _limiter("arxiv"):
            response = await self.client.get(
                "http://export.arxiv.org/api/query",
                params={"search_query": f'ti:"{escaped}"', "max_results": 5},
                headers=self._headers(),
                timeout=settings.pdf_retrieval_timeout_seconds,
            )
        if response.status_code != 200:
            return []

        feed = BeautifulSoup(response.text, "xml")
        for entry in feed.find_all("entry"):
            entry_title = entry.find("title")
            if not titles_match(title, entry_title.get_text() if entry_title else None):
                continue
            entry_id = entry.find("id")
            arxiv_id = extract_arxiv_id(entry_id.get_text() if entry_id else None)
            if arxiv_id:
                return [
                    PdfCandidate(
                        url=f"https://arxiv.org/pdf/{arxiv_id}",
                        provider="arxiv",
                        license="arXiv (see paper)",
                    )
                ]
        return []

    async def _from_elsevier(self, query: PdfQuery) -> list[PdfCandidate] | None:
        """Ask the ScienceDirect Article Retrieval API for the PDF.

        Elsevier serves the full text only for its open-access articles, or for
        subscribed ones when an institutional token is configured. For anything
        else the API still answers 200 with a one-page preview and says so in
        the ``X-ELS-Status`` header, which is why the candidate is not
        advertised as an open copy.
        """
        doi = query.clean_doi
        if not doi or not self.elsevier_api_key:
            return None
        if not belongs_to_publisher(query, ELSEVIER_DOI_PREFIXES, ELSEVIER_HOSTS):
            return None

        return [
            PdfCandidate(
                url=f"https://api.elsevier.com/content/article/doi/{quote(doi, safe='/')}",
                provider="elsevier",
                kind="pdf",
                is_open_access=False,
                headers={**self._elsevier_headers(), "Accept": "application/pdf"},
                public_url=f"https://doi.org/{doi}",
            )
        ]

    async def _from_ieee(self, query: PdfQuery) -> list[PdfCandidate] | None:
        """Look the record up in the IEEE Xplore metadata API.

        The API hands back the article number and whether the document is free
        (``OPEN_ACCESS``/``EPHEMERA``/``CC*``) or locked; the PDF itself is
        served by ``getPDF.jsp`` for that number. Xplore often refuses automated
        downloads even for free papers, so an open hit that fails to download is
        still reported as a link the user can open.
        """
        if not self.ieee_api_key:
            return None

        doi = query.clean_doi
        title = query.clean_title
        params: dict[str, str | int] = {
            "apikey": self.ieee_api_key,
            "format": "json",
            "max_records": 5,
        }
        if doi:
            if not belongs_to_publisher(query, IEEE_DOI_PREFIXES, IEEE_HOSTS):
                return None
            params["doi"] = doi
        elif title:
            # No DOI to gate on, so a title search is the only way to find out
            # whether IEEE has the paper at all.
            params["article_title"] = title
        else:
            return None

        async with _limiter("ieee"):
            response = await self.client.get(
                "https://ieeexploreapi.ieee.org/api/v1/search/articles",
                params=params,
                headers=self._headers(),
                timeout=settings.pdf_retrieval_timeout_seconds,
            )
        if response.status_code != 200:
            logger.info("IEEE Xplore API returned %s for doi=%s", response.status_code, doi)
            return []

        candidates: list[PdfCandidate] = []
        for article in response.json().get("articles") or []:
            if not doi and not titles_match(title, article.get("title")):
                continue

            is_free, license = ieee_access_is_free(
                article.get("access_type") or article.get("accessType")
            )
            landing = article.get("html_url") or article.get("abstract_url")
            stamp_url = article.get("pdf_url")
            arnumber = str(article.get("article_number") or "").strip()
            # Prefer the endpoint that returns the file; the stamp page is HTML.
            pdf_url = IEEE_PDF_URL.format(arnumber=arnumber) if arnumber else stamp_url
            if not pdf_url:
                continue

            candidates.append(
                PdfCandidate(
                    url=pdf_url,
                    provider="ieee",
                    license=license,
                    kind="pdf",
                    is_open_access=is_free,
                    public_url=landing or stamp_url or pdf_url,
                )
            )
        return candidates

    async def _with_looked_up_doi(self, query: PdfQuery) -> tuple[PdfQuery, list[str]]:
        """Fill in a missing DOI from Scopus so every later provider can use it.

        Records imported from a database export without a DOI are otherwise
        stuck with title search everywhere. Returns the query unchanged, plus
        the providers already consumed, when there is nothing to look up.
        """
        if query.clean_doi or not query.clean_title or not self.elsevier_api_key:
            return query, []

        try:
            doi = await self._scopus_doi(query)
        except Exception as exc:  # a flaky provider must not kill the batch
            logger.warning("Scopus DOI lookup failed: %s", exc)
            return query, ["scopus"]

        if not doi:
            return query, ["scopus"]

        logger.info("Scopus resolved '%s' to DOI %s", (query.title or "")[:60], doi)
        return replace(query, doi=doi), ["scopus"]

    async def _scopus_doi(self, query: PdfQuery) -> str | None:
        """Search Scopus by title and return the DOI of a confident match."""
        title = query.clean_title
        if not title:
            return None

        # Quotes would terminate the TITLE() clause; braces are Scopus' literal
        # string form and cope with punctuation better than a loose phrase.
        escaped = title.replace("{", " ").replace("}", " ").strip()
        params: dict[str, str | int] = {
            "query": f"TITLE({{{escaped}}})",
            "count": 5,
            "field": "dc:title,prism:doi,prism:coverDate",
        }
        async with _limiter("elsevier"):
            response = await self.client.get(
                "https://api.elsevier.com/content/search/scopus",
                params=params,
                headers={**self._elsevier_headers(), "Accept": "application/json"},
                timeout=settings.pdf_retrieval_timeout_seconds,
            )
        if response.status_code != 200:
            logger.info("Scopus search returned %s", response.status_code)
            return None

        entries = (response.json().get("search-results") or {}).get("entry") or []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("prism:doi"):
                continue
            if titles_match(title, entry.get("dc:title")):
                return str(entry["prism:doi"])
        return None

    def _elsevier_headers(self) -> dict[str, str]:
        """Elsevier authenticates with a key header, plus a token when off-campus."""
        headers = {**self._headers(), "X-ELS-APIKey": self.elsevier_api_key}
        if self.elsevier_inst_token:
            headers["X-ELS-Insttoken"] = self.elsevier_inst_token
        return headers

    async def _from_landing_page(self, query: PdfQuery) -> list[PdfCandidate] | None:
        """Last resort: read ``citation_pdf_url`` off the source's own URL.

        This is a guess, not a known open-access copy, so it is flagged as such
        — otherwise every Scopus-imported record would look like it had an
        open-access link that failed.
        """
        url = (query.url or "").strip()
        if not url or not url.startswith("http"):
            return None

        return [
            PdfCandidate(url=url, provider="landing-page", kind="landing", is_open_access=False)
        ]

    # ---------------------------------------------------------------- helpers

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        contact = f"; mailto:{self.contact_email}" if self.contact_email else ""
        return {
            "User-Agent": f"SMS-Assistant/0.1 (systematic mapping study tool{contact})",
            "Accept": accept,
        }

    def _find_citation_pdf_url(self, body: bytes) -> str | None:
        try:
            soup = BeautifulSoup(body[:500_000], "html.parser")
        except Exception:
            return None
        meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if meta and meta.get("content"):
            return str(meta["content"]).strip()
        return None
