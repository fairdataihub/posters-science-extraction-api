"""
Match spreadsheet rows to extracted posters when the sheet has no file name
column (Curious today, EasyChair when files come from another platform).

Runs after extraction, on the title and authors poster2json read from each
poster. Tiers, strongest first:

1. Title alone. ROUGE-L F1 of the two titles (word level, accents and case
   ignored). A match is automatic when the score is at least 0.95 and no other
   row or poster comes within 0.10 of it.
2. Title plus authors. When titles are close but not decisive (a shared or
   reworded title, a long title cut short on the poster), the author family
   names break the tie. These matches are always shown for the organizer to
   check.
3. Nothing clears the bar: the row and the file are listed as unmatched.

Assignment is one to one, best score first. Stateless; no database, no model.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

TITLE_AUTO = 0.95  # title score for an automatic match
TITLE_MIN = 0.60  # below this the titles are not the same poster
MARGIN = 0.10  # lead over the next candidate, on both the row and the file side
AUTHORS_MIN = 0.50  # share of family names in common for the authors tier
TITLE_WEIGHT = 0.75  # combined score = 0.75 title + 0.25 authors


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", text)


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def title_score(a: str, b: str) -> float:
    """ROUGE-L F1 between two titles, on normalized words."""
    wa, wb = words(a), words(b)
    lcs = _lcs(wa, wb)
    if not lcs:
        return 0.0
    p, r = lcs / len(wa), lcs / len(wb)
    return 2 * p * r / (p + r)


def family_name(name: str) -> str:
    """Last word of "Given Family", or the part before the comma in "Family, Given"."""
    name = (name or "").strip()
    if "," in name:
        name = name.split(",", 1)[0]
    w = words(name)
    return w[-1] if w else ""


def author_score(a: list[str], b: list[str]) -> float | None:
    """
    Share of family names in common (Dice), with a small allowance for
    spelling and OCR differences. None when either side has no authors.
    """
    fa = [f for f in (family_name(n) for n in a) if f]
    fb = [f for f in (family_name(n) for n in b) if f]
    if not fa or not fb:
        return None
    left = list(fb)
    hits = 0
    for f in fa:
        best = max(left, key=lambda g: difflib.SequenceMatcher(None, f, g).ratio(), default=None)
        if best is not None and (best == f or (len(f) > 3 and difflib.SequenceMatcher(None, f, best).ratio() >= 0.85)):
            hits += 1
            left.remove(best)
    return 2 * hits / (len(fa) + len(fb))


def _score(row: dict, poster: dict, t: float) -> dict:
    au = author_score(row.get("authors") or [], poster.get("authors") or [])
    combined = t if au is None else TITLE_WEIGHT * t + (1 - TITLE_WEIGHT) * au
    return {"title_score": round(t, 3), "author_score": None if au is None else round(au, 3), "score": round(combined, 3)}


def match_rows_to_posters(rows: list[dict], posters: list[dict]) -> dict:
    """
    rows: [{row_key, title, authors: [names]}] from the sheet.
    posters: [{file_name, title, authors: [names]}] from extraction.
    Returns matches (with method and confidence), unmatched_rows, unmatched_files.
    """
    rows = [r for r in rows if r.get("row_key")]
    posters = [p for p in posters if p.get("file_name") and (p.get("title") or p.get("authors"))]
    row_words = [words(r.get("title", "")) for r in rows]
    poster_words = [words(p.get("title", "")) for p in posters]
    row_sets = [set(w) for w in row_words]
    poster_sets = [set(w) for w in poster_words]

    # Score every pair that could reach TITLE_MIN. The LCS is never longer than
    # the shared words, so 2 * shared / (len a + len b) bounds the F1 from above.
    scores: dict[tuple[int, int], dict] = {}
    for i, r in enumerate(rows):
        for j, p in enumerate(posters):
            la, lb = len(row_words[i]), len(poster_words[j])
            shared = len(row_sets[i] & poster_sets[j])
            if not la or not lb or 2 * shared / (la + lb) < TITLE_MIN:
                continue
            lcs = _lcs(row_words[i], poster_words[j])
            t = 2 * lcs / (la + lb)
            scores[(i, j)] = _score(r, p, t)

    by_row: dict[int, list[tuple[int, dict]]] = {}
    by_file: dict[int, list[tuple[int, dict]]] = {}
    for (i, j), s in scores.items():
        by_row.setdefault(i, []).append((j, s))
        by_file.setdefault(j, []).append((i, s))

    def runner_up(i: int, j: int, key: str) -> float:
        others = [s[key] or 0 for k, s in by_row.get(i, []) if k != j]
        others += [s[key] or 0 for k, s in by_file.get(j, []) if k != i]
        return max(others, default=0.0)

    candidates = []
    for (i, j), s in scores.items():
        if s["title_score"] < TITLE_MIN:
            continue
        if s["title_score"] >= TITLE_AUTO and s["title_score"] - runner_up(i, j, "title_score") >= MARGIN:
            method, confidence = "title", "high"
        elif s["author_score"] is not None and s["author_score"] >= AUTHORS_MIN and s["score"] - runner_up(i, j, "score") >= MARGIN:
            method, confidence = "title_authors", "review"
        else:
            continue  # includes two near-identical titles with no authors to tell them apart
        candidates.append((s["score"], i, j, method, confidence))

    matches = []
    used_rows: set[int] = set()
    used_files: set[int] = set()
    for score, i, j, method, confidence in sorted(candidates, key=lambda c: -c[0]):
        if i in used_rows or j in used_files:
            continue
        used_rows.add(i)
        used_files.add(j)
        s = scores[(i, j)]
        matches.append(
            {
                "row_key": rows[i]["row_key"],
                "file_name": posters[j]["file_name"],
                "method": method,
                "confidence": confidence,
                **s,
                "row_title": rows[i].get("title", ""),
                "poster_title": posters[j].get("title", ""),
            }
        )

    unmatched_rows = []
    for i, r in enumerate(rows):
        if i in used_rows:
            continue
        best = max(((s, j) for j, s in by_row.get(i, []) if j not in used_files), key=lambda x: x[0]["score"], default=None)
        unmatched_rows.append(
            {
                "row_key": r["row_key"],
                "title": r.get("title", ""),
                "closest": {"file_name": posters[best[1]]["file_name"], **best[0]} if best else None,
            }
        )
    return {
        "matches": matches,
        "unmatched_rows": unmatched_rows,
        "unmatched_files": [p["file_name"] for j, p in enumerate(posters) if j not in used_files],
        "thresholds": {"title_auto": TITLE_AUTO, "title_min": TITLE_MIN, "margin": MARGIN, "authors_min": AUTHORS_MIN},
    }
