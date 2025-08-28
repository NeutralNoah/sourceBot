"""
High-Accuracy Source Classifier for Early Web NHD Project
Categories: Primary, Secondary, Tertiary, Other

Design goals
- Multi-voter ensemble with transparent triggers and weights
- Stronger Secondary cues for retrospectives/overviews
- Reduced unconditional Primary bias for authoritative domains
- Modern-date + early-Web content => Secondary tilt
- Smoothing: per-voter caps, per-category caps, softmax temperature
- Explicit page overrides (e.g., CERN “Short history of the Web”)

Dependencies: Standard Library only (no external packages)
Python: 3.9+

Suggested use
- Provide url and raw_html when available for best results.
- If raw_html is not provided, the classifier will still work on URL and text fields you pass.
"""

from __future__ import annotations

import re
import math
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, unquote


# ------------------------
# CONFIG
# ------------------------

CATEGORIES = ["Primary", "Secondary", "Tertiary", "Other"]

# Time window aligned to the invention/early Web period
PRIMARY_YEAR_WINDOW = (1989, 1995)

# Domain hint sets (reduced unconditional Primary bias)
DOMAIN_HINTS_PRIMARY = {
    "info.cern.ch",   # Original web server
    "cern.ch",
    "ietf.org",       # RFCs often primary artifacts
    "rfc-editor.org",
    "w3.org",         # Specs may be primary or near-primary
}

DOMAIN_HINTS_SECONDARY = {
    "jstor.org",
    "acm.org",
    "mit.edu",
    "nytimes.com",
    "arxiv.org",
    "wired.com",
    "theguardian.com",
    "nature.com",
}

DOMAIN_HINTS_TERTIARY = {
    "wikipedia.org",
    "britannica.com",
    "scholarpedia.org",
    "britannica.co.uk",
}

DOMAIN_HINTS_OTHER = {
    "blogspot.",
    "medium.com",
    "substack.com",
    "reddit.com",
    "github.io",  # varies wildly; default to Other unless content flips it
    "wordpress.com",
}

# Keyword signals
PRIMARY_KEYWORDS = {
    "original document": 2.0,
    "archival": 1.7,
    "archive copy": 1.5,
    "source code": 2.2,
    "rfc": 2.2,
    "specification": 1.6,
    "minutes": 1.8,
    "memo": 1.6,
    "press release": 1.5,
    "primary source": 2.0,
    "scan": 1.6,
}

SECONDARY_KEYWORDS = {
    "analysis": 1.6,
    "overview": 1.6,
    "history of the web": 2.1,
    "short history of": 2.3,   # stronger for retrospectives
    "retrospective": 2.0,
    "we look back": 1.9,
    "this page recounts": 1.9,
    "case study": 1.6,
    "review article": 1.8,
    "synthesis": 1.5,
    "survey": 1.5,
}

TERTIARY_KEYWORDS = {
    "encyclopedia": 2.1,
    "glossary": 1.6,
    "compendium": 1.8,
    "dictionary": 1.6,
    "almanac": 1.6,
}

OTHER_KEYWORDS = {
    "blog": 1.2,
    "personal": 0.9,
    "opinion": 1.1,
    "musings": 1.0,
    "my thoughts": 1.0,
}

# URL/path-based cues
URL_TERTIARY_CUES = {
    "/wiki/",            # Wikipedia and mirrors
    "/encyclopedia",
}

URL_SECONDARY_CUES = {
    "/history-of",
    "/short-history",
    "/retrospective",
    "/timeline",
    "/overview",
    "/case-study",
}

URL_PRIMARY_CUES = {
    "/rfc/",
    "/spec/",
    "/specification",
    "/minutes",
    "/memos/",
    "/press-release",
}

# Section/structure cues
STRUCTURE_SECONDARY_HEADERS = {
    "background", "methodology", "results", "discussion", "conclusion",
    "related work", "literature review"
}

STRUCTURE_TERTIARY_HEADERS = {
    "see also", "references", "external links", "further reading"
}

# Smoothing/normalization
# - No single voter dominates: per-voter contribution cap
# - Per-category cap to bound evidence
# - Softmax temperature for final confidences
PER_VOTER_CAP = 3.5
PER_CATEGORY_CAP = 8.0
SOFTMAX_TEMP = 1.25

# Explicit overrides for known pages (url substring -> forced classification)
EXPLICIT_OVERRIDES = {
    # CERN modern retrospective page should be Secondary
    "home.cern/science/computing/birth-web/short-history-web": "Secondary",
}


# ------------------------
# UTILITIES
# ------------------------

def now_year() -> int:
    return datetime.now().year


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def safe_lower(s: Optional[str]) -> str:
    return (s or "").lower()


def any_in(substrs: List[str] | set, hay: str) -> bool:
    hay = hay or ""
    return any(sub in hay for sub in substrs)


def strip_tags(html: str) -> str:
    # Fast, simple tag removal for text analysis
    if not html:
        return ""
    # Remove scripts/styles
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    # Remove tags
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    # Unescape entities (basic)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&")
    html = html.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    return normalize_ws(html)


def extract_title(html: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    return normalize_ws(m.group(1)) if m else ""


def extract_meta(html: str) -> Dict[str, str]:
    # Pulls common meta tags: description, og:title, og:description, twitter:title, twitter:description
    meta = {}
    for name in ["description", "author", "keywords"]:
        m = re.search(rf'(?is)<meta[^>]+name=["\']{name}["\'][^>]*content=["\'](.*?)["\']', html or "")
        if m:
            meta[name] = normalize_ws(m.group(1))
    for prop in ["og:title", "og:description", "twitter:title", "twitter:description"]:
        m = re.search(rf'(?is)<meta[^>]+property=["\']{prop}["\'][^>]*content=["\'](.*?)["\']', html or "")
        if m:
            meta[prop] = normalize_ws(m.group(1))
    # Publication/update time (loose)
    for time_key in ["article:published_time", "article:modified_time", "og:updated_time"]:
        m = re.search(rf'(?is)<meta[^>]+property=["\']{time_key}["\'][^>]*content=["\'](.*?)["\']', html or "")
        if m:
            meta[time_key] = normalize_ws(m.group(1))
    return meta


def extract_years(text: str) -> Tuple[Optional[int], Optional[int]]:
    years = {int(y) for y in re.findall(r"\b(18|19|20)\d{2}\b", text)
             if 1800 < int(y) <= now_year()}
    if not years:
        return None, None
    return min(years), max(years)


def parse_url(url: str) -> Dict[str, str]:
    u = urlparse(url or "")
    return {
        "domain": (u.netloc or "").lower(),
        "path": (u.path or "").lower(),
        "query": (u.query or "").lower(),
        "full": (url or "").lower(),
    }


def looks_like_pdf(url: str, html_text: str) -> bool:
    u = safe_lower(url)
    if u.endswith(".pdf") or ".pdf?" in u:
        return True
    # crude PDF magic header check if someone fed raw bytes-as-string
    if html_text and "%PDF-" in html_text[:1024]:
        return True
    return False


def has_references_section(text: str) -> bool:
    # Signals Secondary/Tertiary editorial structure
    return bool(re.search(r"\b(references|bibliography|works cited|further reading)\b", safe_lower(text)))


# ------------------------
# DATA STRUCTURES
# ------------------------

@dataclass
class Vote:
    category: str
    amount: float
    reason: str


@dataclass
class Tally:
    scores: Dict[str, float] = field(default_factory=lambda: {c: 0.0 for c in CATEGORIES})
    triggers: List[str] = field(default_factory=list)

    def add(self, cat: str, amt: float, reason: str):
        # Per-vote cap to prevent spikes
        amt_capped = max(-PER_VOTER_CAP, min(PER_VOTER_CAP, amt))
        self.scores[cat] += amt_capped
        self.triggers.append(f"{reason}+{amt_capped:.2f}")

    def cap_categories(self):
        for cat in self.scores:
            self.scores[cat] = max(-PER_CATEGORY_CAP, min(PER_CATEGORY_CAP, self.scores[cat]))


# ------------------------
# VOTERS
# ------------------------

def domain_voter(url_parts: Dict[str, str]) -> List[Vote]:
    votes = []
    dom = url_parts["domain"]
    # Primary domain hints (reduced boost)
    if any_in(DOMAIN_HINTS_PRIMARY, dom):
        votes.append(Vote("Primary", 1.1, f"domain:primary:{dom}"))
    if any_in(DOMAIN_HINTS_SECONDARY, dom):
        votes.append(Vote("Secondary", 1.6, f"domain:secondary:{dom}"))
    if any_in(DOMAIN_HINTS_TERTIARY, dom):
        votes.append(Vote("Tertiary", 2.2, f"domain:tertiary:{dom}"))
    if any_in(DOMAIN_HINTS_OTHER, dom):
        votes.append(Vote("Other", 1.1, f"domain:other:{dom}"))
    return votes


def url_path_voter(url_parts: Dict[str, str]) -> List[Vote]:
    votes = []
    path_full = (url_parts["path"] + "?" + url_parts["query"]).lower()
    for cue in URL_PRIMARY_CUES:
        if cue in path_full:
            votes.append(Vote("Primary", 1.5, f"url:primary:{cue}"))
    for cue in URL_SECONDARY_CUES:
        if cue in path_full:
            votes.append(Vote("Secondary", 1.8, f"url:secondary:{cue}"))
    for cue in URL_TERTIARY_CUES:
        if cue in path_full:
            votes.append(Vote("Tertiary", 2.0, f"url:tertiary:{cue}"))
    return votes


def keyword_voter(text_candidates: List[str]) -> List[Vote]:
    votes = []
    joined = " ".join(filter(None, (safe_lower(t) for t in text_candidates)))
    # Primary
    for kw, w in PRIMARY_KEYWORDS.items():
        if kw in joined:
            votes.append(Vote("Primary", w, f"kw:primary:{kw}"))
    # Secondary
    for kw, w in SECONDARY_KEYWORDS.items():
        if kw in joined:
            votes.append(Vote("Secondary", w, f"kw:secondary:{kw}"))
    # Tertiary
    for kw, w in TERTIARY_KEYWORDS.items():
        if kw in joined:
            votes.append(Vote("Tertiary", w, f"kw:tertiary:{kw}"))
    # Other
    for kw, w in OTHER_KEYWORDS.items():
        if kw in joined:
            votes.append(Vote("Other", w, f"kw:other:{kw}"))
    return votes


def date_voter(text_for_dates: str, meta: Dict[str, str]) -> List[Vote]:
    votes = []
    earliest, latest = extract_years(text_for_dates)
    # Meta time hints: if present and clearly modern, tilt away from Primary unless it looks like a scanned/original
    meta_pub = safe_lower(meta.get("article:published_time") or "")
    meta_mod = safe_lower(meta.get("article:modified_time") or meta.get("og:updated_time") or "")
    meta_has_modern = any(str(y) in (meta_pub + " " + meta_mod) for y in range(2010, now_year() + 1))

    if earliest and PRIMARY_YEAR_WINDOW[0] <= earliest <= PRIMARY_YEAR_WINDOW[1]:
        votes.append(Vote("Primary", 1.4, f"date:early_web:{earliest}"))
        # If also modern references appear later, it's likely a retrospective
        if latest and latest > PRIMARY_YEAR_WINDOW[1] + 5:
            votes.append(Vote("Secondary", 1.7, "date:modern_retro"))
    elif earliest:
        # General date presence but outside primary window => slight Secondary tilt
        votes.append(Vote("Secondary", 1.0, f"date:recent:{earliest}"))

    if meta_has_modern:
        votes.append(Vote("Secondary", 0.8, "meta:modern_pub"))

    return votes


def structure_voter(full_text: str) -> List[Vote]:
    votes = []
    text = safe_lower(full_text)
    # Structured sections typical of Secondary scholarship
    for h in STRUCTURE_SECONDARY_HEADERS:
        if re.search(rf"\b{re.escape(h)}\b", text):
            votes.append(Vote("Secondary", 0.6, f"struct:secondary:{h}"))
    # Reference-like sections can indicate Secondary/Tertiary
    if has_references_section(text):
        votes.append(Vote("Secondary", 0.8, "struct:references"))
        votes.append(Vote("Tertiary", 0.4, "struct:references:ter"))

    # Definition-heavy content can tilt Tertiary slightly
    if re.search(r"\b(definitions?|terminology|nomenclature|lexicon)\b", text):
        votes.append(Vote("Tertiary", 0.6, "struct:definitions"))

    return votes


def artifact_voter(url_parts: Dict[str, str], title: str, text: str) -> List[Vote]:
    votes = []
    # PDFs and RFCs often (not always) are primary artifacts or specifications
    if looks_like_pdf(url_parts["full"], text):
        votes.append(Vote("Primary", 1.2, "artifact:pdf"))
    if "/rfc/" in url_parts["path"] or "rfc " in safe_lower(title):
        votes.append(Vote("Primary", 1.8, "artifact:rfc"))
    # Presence of code blocks or very technical spec-like language (crude)
    if re.search(r"\b(must|shall|should|may|conform|requirement)\b", safe_lower(text)):
        votes.append(Vote("Primary", 0.7, "artifact:spec_language"))
    # Heavy quoting of named individuals + years may indicate Secondary synthesis (citations)
    quotes = len(re.findall(r"[“\"\'].*?[”\"\']", text))
    if quotes >= 8:
        votes.append(Vote("Secondary", 0.6, "artifact:heavy_quotes"))
    return votes


# ------------------------
# SOFTMAX / SMOOTHING
# ------------------------

def softmax(scores: Dict[str, float], temp: float = SOFTMAX_TEMP) -> Dict[str, float]:
    # Stabilize: shift by max before exponentiation
    max_v = max(scores.values()) if scores else 0.0
    exps = {k: math.exp((v - max_v) / temp) for k, v in scores.items()}
    denom = sum(exps.values()) or 1.0
    return {k: exps[k] / denom for k in scores}


# ------------------------
# CORE CLASSIFIER
# ------------------------

def classify_source(
    url: str,
    raw_html: str = "",
    meta_date_hint: Optional[str] = None,
    explain: bool = True,
) -> Dict[str, object]:
    """
    Classify a source into Primary, Secondary, Tertiary, or Other.

    Inputs:
    - url: page URL (used for domain/path voters and explicit overrides)
    - raw_html: optional HTML source to improve accuracy
    - meta_date_hint: optional external date hint (YYYY or free text containing a year)
    - explain: include detailed explanation and triggers

    Output dict keys:
    - category: str
    - confidence: float (0..1)
    - scores: dict per category (pre-softmax totals, capped)
    - confidences: dict per category (softmax distribution)
    - explanation: str (if explain=True)
    - triggers: list of detailed signals
    """
    url_l = safe_lower(url)
    # Explicit overrides
    for sub, cat in EXPLICIT_OVERRIDES.items():
        if sub in url_l:
            confident = 0.98
            confs = {c: (0.98 if c == cat else (0.02 / 3.0)) for c in CATEGORIES}
            return {
                "category": cat,
                "confidence": confident,
                "scores": {c: (10.0 if c == cat else 0.0) for c in CATEGORIES},
                "confidences": confs,
                "explanation": f"Explicit override matched: {sub} -> {cat}",
                "triggers": [f"override:{sub}"]
            }

    # Parse URL and HTML artifacts
    parts = parse_url(url)
    html = raw_html or ""
    title = extract_title(html)
    meta = extract_meta(html)
    body_text = strip_tags(html)
    text_for_kw = " ".join([
        title,
        meta.get("description", ""),
        meta.get("og:title", ""),
        meta.get("og:description", ""),
        meta.get("twitter:title", ""),
        meta.get("twitter:description", ""),
        body_text
    ])

    # If an external date hint is present, inject into the analysis text
    if meta_date_hint:
        text_for_kw += f" {meta_date_hint}"

    tally = Tally()

    # Run voters
    voters = [
        ("domain", domain_voter(parts)),
        ("url_path", url_path_voter(parts)),
        ("keywords", keyword_voter([title, meta.get("description", ""), text_for_kw])),
        ("dates", date_voter(text_for_kw, meta)),
        ("structure", structure_voter(body_text)),
        ("artifact", artifact_voter(parts, title, body_text)),
    ]

    for voter_name, votes in voters:
        for v in votes:
            tally.add(v.category, v.amount, f"{voter_name}:{v.reason}")

    # Category caps
    tally.cap_categories()

    # Final confidences
    confidences = softmax(tally.scores, temp=SOFTMAX_TEMP)
    category = max(confidences, key=confidences.get)
    confidence = confidences[category]

    # Build explanation
    explanation_txt = ""
    if explain:
        explanation_txt = (
            f"Category: {category}\n"
            f"Confidence: {confidence:.3f}\n"
            f"Scores (capped): {json.dumps(tally.scores, ensure_ascii=False)}\n"
            f"Confidences: {json.dumps(confidences, ensure_ascii=False)}\n"
            f"Title: {title}\n"
            f"Meta: {json.dumps(meta, ensure_ascii=False)}\n"
            f"Triggers:\n  - " + "\n  - ".join(tally.triggers[:300])  # guard against enormous pages
        )

    return {
        "category": category,
        "confidence": confidence,
        "scores": tally.scores,
        "confidences": confidences,
        "explanation": explanation_txt if explain else "",
        "triggers": tally.triggers,
    }


# ------------------------
# DEMO / TEST HARNESS
# ------------------------

def _demo():
    tests = [
        {
            "url": "https://home.cern/science/computing/birth-web/short-history-web",
            "html": "<html><head><title>Short history of the Web</title>"
                    "<meta name='description' content='CERN looks back at the birth of the Web.'/>"
                    "</head><body>In 1989 the web began... This page recounts the early years. "
                    "Updated 2019. References</body></html>"
        },
        {
            "url": "https://www.rfc-editor.org/rfc/rfc1945.txt",
            "html": "RFC 1945 Hypertext Transfer Protocol -- HTTP/1.0. This specification states MUST, SHOULD, MAY..."
        },
        {
            "url": "https://en.wikipedia.org/wiki/History_of_the_World_Wide_Web",
            "html": "<html><head><title>History of the World Wide Web - Wikipedia</title></head>"
                    "<body>Encyclopedia article. See also. References. Timeline of events 1989, 1990, 1991...</body></html>"
        },
        {
            "url": "https://medium.com/@someone/my-thoughts-on-tim-berners-lee-and-the-web-5d1",
            "html": "<html><head><title>My thoughts on the Web</title>"
                    "<meta name='description' content='A personal blog about the Web history'/></head>"
                    "<body>Opinion and musings, not a formal history.</body></html>"
        },
        {
            "url": "https://www.w3.org/History/19921103-hypertext/hypertext/WWW/TheProject.html",
            "html": "<html><head><title>The World Wide Web project</title></head>"
                    "<body>Original document scan and archival copy of early WWW project page. 1991</body></html>"
        },
    ]
    for t in tests:
        res = classify_source(t["url"], raw_html=t["html"], explain=True)
        print("\nURL:", t["url"])
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _demo()
