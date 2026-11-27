"""
Match spreadsheet rows to extracted posters when a row has no file link
(Curious has no file name column; at ISMB and BOSC the poster files come from
another platform than EasyChair).

Runs after extraction, on the title and authors poster2json read from each
poster. Tiers, strictest first:

1. title: ROUGE-L F1 of the two titles (word level, accents and case ignored)
   of at least 0.95, with no other row or poster within 0.10. Applied as is.
2. title_authors: titles at least 0.60 and author family names at least 0.50
   in common; the combined score (0.75 title, 0.25 authors) leads by 0.10.
3. words_fuzzy: bag of words, order ignored, with a small allowance for
   spelling and OCR slips in each word, and a title cut short on the poster
   counted by containment. At least 0.70 (or 0.55 with authors at least 0.50),
   leading by 0.10.
4. best_remaining: among what is left, the row and the poster are each
   other's best candidate, the combined score is at least 0.50 and leads by
   0.05.

The tiers run in a loop. Each match takes its row and poster out of play,
which can make another pair decisive, so after any match the loop starts again
at tier 1. It stops when every poster has a row, or when no tier can place
anything more. Two equal candidates are never guessed at any tier (the margin
is zero). Tier 1 matches are "high" confidence; the rest are "review" and are
shown to the organizer to check. A poster left without a row keeps its
extracted metadata only.

Before the loop, a row whose suggested file (its number appears in the file
name but the name does not start with it) has an agreeing title takes that
file, for review ("number_suggested"), unless another row or poster fits as
well or better by title (within FINAL_MARGIN).

Inputs are read defensively: a field of the wrong type is read as empty, and
a poster extraction read nothing from is listed with the files without a row
(and in blank_files), never dropped.

Rows the sheet already paired with a file (file name column, close file name,
number) are checked too: their extracted title has to agree with the sheet
title (PAIR_AGREE), and authors, when both sides have them, must share a
family name. A pair that does not agree is flagged for the organizer however
it was paired, so a wrong file is caught before its metadata goes out.

Stateless; no database, no model.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

TITLE_AUTO = 0.95  # tier 1: title score for an automatic match
TITLE_MIN = 0.60  # tier 2: title floor when authors agree
AUTHORS_MIN = 0.50  # share of family names in common to count as agreeing
WORDS_MIN = 0.70  # tier 3: bag of words floor
WORDS_WITH_AUTHORS = 0.55  # tier 3: bag of words floor when authors agree
FLOOR = 0.50  # tier 4: combined score floor
MARGIN = 0.10  # lead over the next candidate, on both the row and the poster side
FINAL_MARGIN = 0.05  # tier 4 lead
TITLE_WEIGHT = 0.75  # combined score = 0.75 title + 0.25 authors
WORD_FUZZ = 0.80  # two words count as the same when their spelling ratio is this high
PREFILTER = 0.25  # pairs sharing less than this (Dice of near-same words) are not scored
CANDIDATES_PER_ROW = 25  # each row scores at most this many posters, best prefilter first
# Title check on rows already paired with a file (file name column, close file
# name, number). Calibrated on real titles from 53 conferences: a title against
# a noisy copy of itself (cut short, OCR slips, a subtitle added, words
# swapped) vs two different posters from the same conference. At 0.50, 0.8% of
# true pairs are flagged for a look and 0.1% of different posters pass (a
# French and an English title of one project). A flagged true pair costs one
# click; a wrong pair that passes publishes the wrong metadata.
PAIR_AGREE = 0.50
PAIR_TITLE_SURE = 0.90  # with PAIR_TITLE_SURE_WORDS, an author mismatch alone does not flag a pair
PAIR_TITLE_SURE_WORDS = 6

STOPWORDS = frozenset(
    "a an and are as at by de der die for from in into is of on or the to via vs with".split()
)


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z0-9]+", text)


def content_words(ws: list[str]) -> list[str]:
    out = [w for w in ws if w not in STOPWORDS and len(w) > 1]
    return out or ws


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
    return 2 * lcs / (len(wa) + len(wb))


_RATIO_CACHE: dict[tuple[str, str], float] = {}


def _word_ratio(a: str, b: str) -> float:
    key = (a, b) if a < b else (b, a)
    hit = _RATIO_CACHE.get(key)
    if hit is None:
        if len(_RATIO_CACHE) > 200_000:
            _RATIO_CACHE.clear()
        hit = _RATIO_CACHE[key] = difflib.SequenceMatcher(None, a, b).ratio()
    return hit


def words_score(a: list[str], b: list[str]) -> float:
    """
    Bag of words similarity of two titles (content words, order ignored).
    Words match exactly, or when longer than three letters with a spelling
    ratio of 0.80 or more. The score is the Dice of matched words, or 0.9 of
    the containment when the shorter title has five words or more (a title
    cut short, or a subtitle added).
    """
    if not a or not b:
        return 0.0
    left = list(b)
    hits = 0
    rest = []
    for w in a:
        if w in left:
            left.remove(w)
            hits += 1
        else:
            rest.append(w)
    for w in rest:
        if len(w) <= 3 or not left:
            continue
        best = max(left, key=lambda g: _word_ratio(w, g))
        if len(best) > 3 and _word_ratio(w, best) >= WORD_FUZZ:
            left.remove(best)
            hits += 1
    dice = 2 * hits / (len(a) + len(b))
    shorter = min(len(a), len(b))
    contain = 0.9 * hits / shorter if shorter >= 5 else 0.0
    return max(dice, contain)


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
        best = max(left, key=lambda g: _word_ratio(f, g), default=None)
        if best is not None and (best == f or (len(f) > 3 and _word_ratio(f, best) >= 0.85)):
            hits += 1
            left.remove(best)
    return 2 * hits / (len(fa) + len(fb))


def _combine(x: float, au: float | None) -> float:
    return x if au is None else TITLE_WEIGHT * x + (1 - TITLE_WEIGHT) * au


def _pair_scores(row: dict, poster: dict, rw: list[str], pw: list[str], rc: list[str], pc: list[str]) -> dict:
    t = 2 * _lcs(rw, pw) / (len(rw) + len(pw)) if rw and pw else 0.0
    w = max(words_score(rc, pc), t)
    au = author_score(row.get("authors") or [], poster.get("authors") or [])
    return {
        "title_score": round(t, 3),
        "words_score": round(w, 3),
        "author_score": None if au is None else round(au, 3),
        "score": round(_combine(t, au), 3),
        "words_combined": round(_combine(w, au), 3),
    }


def _agree(s: dict) -> bool:
    return s["author_score"] is not None and s["author_score"] >= AUTHORS_MIN


# Each tier: name, confidence, the score its margin is measured on, the margin,
# and whether a pair qualifies.
TIERS = (
    ("title", "high", "title_score", MARGIN, lambda s: s["title_score"] >= TITLE_AUTO),
    ("title_authors", "review", "score", MARGIN, lambda s: s["title_score"] >= TITLE_MIN and _agree(s)),
    (
        "words_fuzzy",
        "review",
        "words_combined",
        MARGIN,
        lambda s: s["words_score"] >= WORDS_MIN or (s["words_score"] >= WORDS_WITH_AUTHORS and _agree(s)),
    ),
    ("best_remaining", "review", "words_combined", FINAL_MARGIN, lambda s: s["words_combined"] >= FLOOR),
)


def check_pair(row: dict, poster: dict) -> dict:
    """
    Does the extracted poster look like the sheet row it is paired with? The
    titles agree when their bag of words score (spelling slips, a title cut
    short and a subtitle allowed) reaches PAIR_AGREE. Authors with no family
    name in common also count as a disagreement, when both sides have authors
    and the titles are not already near identical.
    """
    rw, pw = words(row.get("title", "")), words(poster.get("title", ""))
    if not rw or not pw:
        return {"status": "unchecked", "reason": "no_title"}
    s = _pair_scores(row, poster, rw, pw, content_words(rw), content_words(pw))
    out = {k: s[k] for k in ("title_score", "words_score", "author_score")}
    if s["words_score"] < PAIR_AGREE:
        return {**out, "status": "disagree", "reason": "title"}
    # A long, near-identical title settles it; otherwise authors with no family
    # name in common flag the pair (a short generic title can repeat).
    title_sure = s["words_score"] >= PAIR_TITLE_SURE and len(content_words(rw)) >= PAIR_TITLE_SURE_WORDS
    if s["author_score"] == 0.0 and not title_sure:
        return {**out, "status": "disagree", "reason": "authors"}
    return {**out, "status": "agree", "reason": None}


def check_pairs(pairs: list[dict], posters: list[dict]) -> list[dict]:
    """
    pairs: [{row_key, file_name, title, authors, method}] rows the sheet already
    paired with a file. Each comes back with its check; a pair whose poster is
    not in `posters` (not extracted yet) is "unchecked".
    """
    by_file = {p.get("file_name"): p for p in posters if p.get("file_name")}
    out = []
    for pair in pairs:
        poster = by_file.get(pair.get("file_name"))
        check = check_pair(pair, poster) if poster else {"status": "unchecked", "reason": "not_extracted"}
        out.append(
            {
                "row_key": pair.get("row_key", ""),
                "file_name": pair.get("file_name", ""),
                "method": pair.get("method", ""),
                **check,
                "row_title": pair.get("title", ""),
                "poster_title": (poster or {}).get("title", ""),
            }
        )
    return out


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _names(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def normalize_items(items) -> list[dict]:
    """
    Keep each item's fields in the shape the matcher reads: strings for
    row_key, file_name, title, method and suggested_file, a list of name
    strings for authors. A wrong type (a number for a title, one string for
    the authors) is read or dropped instead of failing the request.
    """
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        clean = {k: _text(item.get(k)) for k in ("row_key", "file_name", "title", "method", "suggested_file", "number")}
        clean["authors"] = _names(item.get("authors"))
        out.append(clean)
    return out


def word_keys(w: str) -> set[str]:
    """
    Lookup keys for a word: the word, and for words longer than three letters
    the word with any one letter dropped. Two words that differ by one OCR
    slip (a letter swapped, added or lost) share a key, so pairs that only
    agree once spelling slips are allowed are still found as candidates.
    """
    if len(w) <= 3:
        return {w}
    return {w} | {w[:k] + w[k + 1:] for k in range(len(w))}


def _title_keys(item: dict) -> tuple[list[str], set[str]]:
    cw = content_words(words(item.get("title", "")))
    keys: set[str] = set()
    for w in cw:
        keys |= word_keys(w)
    return cw, keys


def _best_other(target: dict, others: list[dict], keyed=None) -> float:
    """Highest bag of words title score between target and any of others that
    shares a word with it, OCR slips allowed. keyed caches _title_keys by id."""
    keyed = keyed if keyed is not None else {}

    def get(item):
        hit = keyed.get(id(item))
        if hit is None:
            hit = keyed[id(item)] = _title_keys(item)
        return hit

    tw, tkeys = get(target)
    best = 0.0
    for o in others:
        ow, okeys = get(o)
        if not tkeys & okeys:
            continue
        best = max(best, words_score(tw, ow))
    return best


def match_rows_to_posters(rows: list[dict], posters: list[dict], pairs: list[dict] | None = None) -> dict:
    """
    rows: [{row_key, title, authors: [names], suggested_file?}] from the sheet,
      still without a file. suggested_file is a file whose name carries the
      row's number but does not start with it; it is taken (for review) only
      when its title agrees.
    posters: [{file_name, title, authors: [names]}] from extraction.
    pairs: [{row_key, file_name, title, authors, method}] rows already paired
      by file name or number; their files are out of play, and each pair gets
      a title check (pair_checks; disagreements also in flagged).
    Returns matches (with method, confidence and the round that made them),
    unmatched_rows (with the closest poster left), unmatched_files,
    pair_checks and flagged.
    """
    rows, posters = normalize_items(rows), normalize_items(posters)
    pairs = [p for p in normalize_items(pairs) if p["file_name"]]
    pair_checks = check_pairs(pairs, posters)
    taken = {p["file_name"] for p in pairs}
    rows = [r for r in rows if r["row_key"]]
    posters = [p for p in posters if p["file_name"] and p["file_name"] not in taken]
    # A poster extraction read nothing from (no title, no authors) cannot be
    # matched, but it is still a file without a row, so it is reported.
    blank_files = [p["file_name"] for p in posters if not (p["title"] or p["authors"])]
    posters = [p for p in posters if p["title"] or p["authors"]]

    # A file suggested by its number is taken first, if its title agrees and
    # no other row or poster fits better: the number in a file name ("(2)", a
    # version) is weak evidence, so it never wins against a clearer title.
    suggested_matches: list[dict] = []
    keyed: dict[int, tuple[list[str], set[str]]] = {}
    by_file = {p["file_name"]: p for p in posters}
    for r in rows:
        poster = by_file.get(r.get("suggested_file") or "")
        if poster is None:
            continue
        check = check_pair(r, poster)
        if check["status"] == "agree":
            other_rows = [o for o in rows if o is not r]
            other_posters = [p for p in by_file.values() if p is not poster]
            rival = max(_best_other(poster, other_rows, keyed), _best_other(r, other_posters, keyed))
            if check["words_score"] - rival < FINAL_MARGIN:
                continue
            suggested_matches.append(
                {
                    "row_key": r["row_key"],
                    "file_name": poster["file_name"],
                    "method": "number_suggested",
                    "confidence": "review",
                    "round": 0,
                    "title_score": check["title_score"],
                    "words_score": check["words_score"],
                    "author_score": check["author_score"],
                    "score": check["words_score"],
                    "row_title": r.get("title", ""),
                    "poster_title": poster.get("title", ""),
                }
            )
            del by_file[poster["file_name"]]
    if suggested_matches:
        done_rows = {m["row_key"] for m in suggested_matches}
        done_files = {m["file_name"] for m in suggested_matches}
        rows = [r for r in rows if r["row_key"] not in done_rows]
        posters = [p for p in posters if p["file_name"] not in done_files]
    row_words = [words(r.get("title", "")) for r in rows]
    poster_words = [words(p.get("title", "")) for p in posters]
    row_content = [content_words(w) for w in row_words]
    poster_content = [content_words(w) for w in poster_words]

    # Candidate pairs share content words, the same or one OCR slip apart
    # (word_keys). Score only those whose shared words reach PREFILTER, and
    # for each row only its CANDIDATES_PER_ROW best by that share, so a batch
    # of near-identical titles cannot make the scoring quadratic.
    # Ties on that share are broken by words shared exactly: "1001" is one
    # slip from "101", "100" and "1011", and without the tie break the poster
    # with the same title could rank past the cap behind those.
    index: dict[str, set[int]] = {}
    exact_index: dict[str, set[int]] = {}
    for j, ws in enumerate(poster_content):
        for w in set(ws):
            exact_index.setdefault(w, set()).add(j)
            for k in word_keys(w):
                index.setdefault(k, set()).add(j)
    scores: dict[tuple[int, int], dict] = {}
    for i, ws in enumerate(row_content):
        shared: dict[int, int] = {}
        exact: dict[int, int] = {}
        for w in set(ws):
            hit: set[int] = set()
            for k in word_keys(w):
                hit |= index.get(k, set())
            for j in hit:
                shared[j] = shared.get(j, 0) + 1
            for j in exact_index.get(w, ()):
                exact[j] = exact.get(j, 0) + 1
        rset = len(set(ws))
        ranked = sorted(
            ((2 * n / (rset + len(set(poster_content[j]))), exact.get(j, 0), j) for j, n in shared.items()),
            key=lambda t: (-t[0], -t[1], t[2]),
        )
        for dice, _, j in ranked[:CANDIDATES_PER_ROW]:
            if dice < PREFILTER:
                break
            scores[(i, j)] = _pair_scores(rows[i], posters[j], row_words[i], poster_words[j], ws, poster_content[j])

    by_row: dict[int, list[int]] = {}
    by_file: dict[int, list[int]] = {}
    for i, j in scores:
        by_row.setdefault(i, []).append(j)
        by_file.setdefault(j, []).append(i)

    used_rows: set[int] = set()
    used_files: set[int] = set()
    matches: list[dict] = []

    # Each row's and each poster's candidates, best first per score key, so
    # the runner up and best candidate are found by walking past the few
    # already used instead of scanning every alternative.
    ranked_cache: dict[tuple[str, str, int], list[tuple[float, int]]] = {}

    def ranked(side: str, x: int, key: str) -> list[tuple[float, int]]:
        hit = ranked_cache.get((side, key, x))
        if hit is None:
            if side == "row":
                hit = sorted(((scores[(x, k)][key], k) for k in by_row.get(x, [])), reverse=True)
            else:
                hit = sorted(((scores[(k, x)][key], k) for k in by_file.get(x, [])), reverse=True)
            ranked_cache[(side, key, x)] = hit
        return hit

    def _first(side: str, x: int, key: str, skip: int | None, used: set[int]) -> tuple[float, int] | None:
        for v, k in ranked(side, x, key):
            if k != skip and k not in used:
                return v, k
        return None

    def runner_up(i: int, j: int, key: str) -> float:
        a = _first("row", i, key, j, used_files)
        b = _first("file", j, key, i, used_rows)
        return max(a[0] if a else 0.0, b[0] if b else 0.0)

    def best_of_row(i: int, key: str) -> int | None:
        hit = _first("row", i, key, None, used_files)
        return hit[1] if hit else None

    def best_of_file(j: int, key: str) -> int | None:
        hit = _first("file", j, key, None, used_rows)
        return hit[1] if hit else None

    rounds = 0
    while len(used_rows) < len(rows) and len(used_files) < len(posters):
        rounds += 1
        placed = False
        for name, confidence, key, margin, qualifies in TIERS:
            found = []
            for (i, j), s in scores.items():
                if i in used_rows or j in used_files or not qualifies(s):
                    continue
                if s[key] - runner_up(i, j, key) < margin:
                    continue
                if name == "best_remaining" and (best_of_row(i, key) != j or best_of_file(j, key) != i):
                    continue
                found.append((s[key], i, j))
            for _, i, j in sorted(found, reverse=True):
                if i in used_rows or j in used_files:
                    continue
                used_rows.add(i)
                used_files.add(j)
                s = scores[(i, j)]
                matches.append(
                    {
                        "row_key": rows[i]["row_key"],
                        "file_name": posters[j]["file_name"],
                        "method": name,
                        "confidence": confidence,
                        "round": rounds,
                        "title_score": s["title_score"],
                        "words_score": s["words_score"],
                        "author_score": s["author_score"],
                        "score": s[key],
                        "row_title": rows[i].get("title", ""),
                        "poster_title": posters[j].get("title", ""),
                    }
                )
                placed = True
            if placed:
                break  # start again at tier 1 with these taken out of play
        if not placed:
            break

    unmatched_rows = []
    for i, r in enumerate(rows):
        if i in used_rows:
            continue
        j = best_of_row(i, "words_combined")
        closest = None
        if j is not None:
            s = scores[(i, j)]
            closest = {
                "file_name": posters[j]["file_name"],
                "title_score": s["title_score"],
                "words_score": s["words_score"],
                "author_score": s["author_score"],
                "score": s["words_combined"],
            }
        unmatched_rows.append({"row_key": r["row_key"], "title": r.get("title", ""), "closest": closest})
    return {
        "matches": suggested_matches + matches,
        "unmatched_rows": unmatched_rows,
        "unmatched_files": [p["file_name"] for j, p in enumerate(posters) if j not in used_files] + blank_files,
        "blank_files": blank_files,
        "pair_checks": pair_checks,
        "flagged": [c for c in pair_checks if c["status"] == "disagree"],
        "rounds": rounds,
        "thresholds": {
            "pair_agree": PAIR_AGREE,
            "pair_title_sure": PAIR_TITLE_SURE,
            "title_auto": TITLE_AUTO,
            "title_min": TITLE_MIN,
            "authors_min": AUTHORS_MIN,
            "words_min": WORDS_MIN,
            "words_with_authors": WORDS_WITH_AUTHORS,
            "floor": FLOOR,
            "margin": MARGIN,
            "final_margin": FINAL_MARGIN,
        },
    }
