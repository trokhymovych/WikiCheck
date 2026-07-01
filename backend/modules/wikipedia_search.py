from __future__ import annotations

from typing import Dict, List

import mediawiki

from ..schemas import Passage

_wiki_cache: Dict[str, mediawiki.MediaWiki] = {}

RESULTS_PER_QUERY = 3


def _get_wiki(lang: str) -> mediawiki.MediaWiki:
    if lang not in _wiki_cache:
        _wiki_cache[lang] = mediawiki.MediaWiki(lang=lang)
    return _wiki_cache[lang]


def _fetch_passages(title: str, lang: str) -> List[Passage]:
    wiki = _get_wiki(lang)
    try:
        page = wiki.page(title)
    except Exception:
        return []

    url = page.url
    article = page.title
    summary = page.summary or ""
    sentences = [s.strip() for s in summary.replace("\n", " ").split(". ") if len(s.strip()) > 20]
    return [
        Passage(article=article, text=s, section="Summary", url=url, language=lang.upper())
        for s in sentences
    ]


def search(queries: List[str], languages: List[str]) -> List[Passage]:
    seen: set = set()
    results: List[Passage] = []

    for lang in languages:
        wiki = _get_wiki(lang)
        for query in queries:
            try:
                titles = wiki.search(query, results=RESULTS_PER_QUERY)
            except Exception:
                continue
            for title in titles:
                passages = _fetch_passages(title, lang)
                for p in passages:
                    key = (p.article, p.text)
                    if key not in seen:
                        seen.add(key)
                        results.append(p)

    return results
