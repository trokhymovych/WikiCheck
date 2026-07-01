from __future__ import annotations

from typing import Dict, Generator, List, Tuple

import mediawiki
from bs4 import BeautifulSoup, NavigableString, Tag

from ..schemas import Passage

_wiki_cache: Dict[str, mediawiki.MediaWiki] = {}

RESULTS_PER_QUERY = 3
MAX_SENTENCES_PER_PAGE = 80


def _get_wiki(lang: str) -> mediawiki.MediaWiki:
    if lang not in _wiki_cache:
        _wiki_cache[lang] = mediawiki.MediaWiki(lang=lang)
    return _wiki_cache[lang]


def _build_ref_map(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """Map cite_note-N IDs to lists of external reference URLs."""
    ref_map: Dict[str, List[str]] = {}
    for li in soup.find_all("li", id=lambda x: x and x.startswith("cite_note")):
        urls = [
            a["href"]
            for a in li.find_all("a", href=True)
            if a["href"].startswith("http") and "wikipedia.org/wiki" not in a["href"]
        ]
        ref_map[li["id"]] = urls
    return ref_map


def _walk_tokens(p_tag: Tag) -> Generator[Tuple[str, str], None, None]:
    """Yield ('text', str) or ('ref', cite_id) tokens from a <p> tag."""
    for child in p_tag.children:
        if isinstance(child, NavigableString):
            yield ("text", str(child))
        elif isinstance(child, Tag):
            if child.name == "sup" and "reference" in child.get("class", []):
                a = child.find("a")
                if a:
                    href = a.get("href", "")
                    if href.startswith("#cite_note"):
                        yield ("ref", href[1:])
            elif child.name not in ("style", "script"):
                yield ("text", child.get_text())


def _sentences_from_paragraph(
    p_tag: Tag, ref_map: Dict[str, List[str]]
) -> List[Tuple[str, List[str]]]:
    """
    Parse a <p> tag into (sentence_text, reference_urls) pairs.
    References are attached to the sentence they immediately follow in the HTML.
    """
    sentences: List[Tuple[str, List[str]]] = []
    current_text = ""
    current_refs: List[str] = []

    for kind, val in _walk_tokens(p_tag):
        if kind == "ref":
            current_refs.append(val)
        else:
            # Split on ". " to detect sentence boundaries
            remaining = val
            while True:
                dot_idx = remaining.find(". ")
                if dot_idx == -1:
                    current_text += remaining
                    break
                current_text += remaining[:dot_idx]
                sentence = current_text.strip()
                if len(sentence) > 20:
                    urls = _dedup(
                        url
                        for ref_id in current_refs
                        for url in ref_map.get(ref_id, [])
                    )
                    sentences.append((sentence, urls))
                current_text = ""
                current_refs = []
                remaining = remaining[dot_idx + 2:]

    # Flush the last partial sentence
    sentence = current_text.strip()
    if len(sentence) > 20:
        urls = _dedup(
            url for ref_id in current_refs for url in ref_map.get(ref_id, [])
        )
        sentences.append((sentence, urls))

    return sentences


def _dedup(iterable) -> List[str]:
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _fetch_passages(title: str, lang: str) -> List[Passage]:
    wiki = _get_wiki(lang)
    try:
        page = wiki.page(title)
        html = page.html
    except Exception:
        return []

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["style", "script"]):
        tag.decompose()
    ref_map = _build_ref_map(soup)

    url = page.url
    article = page.title
    results: List[Passage] = []
    seen: set = set()
    current_section = "Summary"

    content = soup.find("div", class_="mw-parser-output") or soup

    for elem in content.children:
        if not isinstance(elem, Tag):
            continue

        if elem.name in ("h2", "h3", "h4"):
            headline = elem.find(class_="mw-headline")
            current_section = headline.get_text() if headline else elem.get_text()

        elif elem.name == "p":
            for text, refs in _sentences_from_paragraph(elem, ref_map):
                if text not in seen:
                    seen.add(text)
                    results.append(
                        Passage(
                            article=article,
                            text=text,
                            section=current_section,
                            url=url,
                            language=lang.upper(),
                            references=refs,
                        )
                    )
                    if len(results) >= MAX_SENTENCES_PER_PAGE:
                        return results

    return results


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
