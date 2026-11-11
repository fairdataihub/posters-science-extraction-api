"""Tests for poster_matcher: rows without a file name matched to extracted posters."""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import poster_matcher as pm  # noqa: E402


def row(key, title, authors=()):
    return {"row_key": key, "title": title, "authors": list(authors)}


def poster(name, title, authors=()):
    return {"file_name": name, "title": title, "authors": list(authors)}


def got(out):
    return {m["row_key"]: (m["file_name"], m["method"], m["confidence"]) for m in out["matches"]}


def test_title_score_ignores_case_accents_and_punctuation():
    assert pm.title_score("Caf\u00e9 data: a FAIR approach!", "cafe data a fair approach") == 1.0


def test_clear_titles_match_automatically():
    out = pm.match_rows_to_posters(
        [row("r1", "Open tools for genome assembly"), row("r2", "Measuring trust in AI models")],
        [poster("b.pdf", "Measuring Trust in AI Models"), poster("a.pdf", "Open tools for genome assembly.")],
    )
    assert got(out) == {"r1": ("a.pdf", "title", "high"), "r2": ("b.pdf", "title", "high")}
    assert out["unmatched_rows"] == [] and out["unmatched_files"] == []


def test_same_title_is_split_by_authors_and_flagged_for_review():
    rows = [row("r1", "Annual lab report", ["Ana Silva", "Bo Chen"]), row("r2", "Annual lab report", ["Dee Fox", "Eve Ng"])]
    posters = [poster("x.pdf", "Annual lab report", ["E. Ng", "D. Fox"]),
               poster("y.pdf", "Annual lab report", ["A. Silva", "B. Chen"])]
    out = pm.match_rows_to_posters(rows, posters)
    assert got(out) == {"r1": ("y.pdf", "title_authors", "review"), "r2": ("x.pdf", "title_authors", "review")}


def test_same_title_without_authors_is_not_guessed():
    out = pm.match_rows_to_posters(
        [row("r1", "Annual lab report"), row("r2", "Annual lab report")],
        [poster("x.pdf", "Annual lab report"), poster("y.pdf", "Annual lab report")],
    )
    assert out["matches"] == []
    assert {r["row_key"] for r in out["unmatched_rows"]} == {"r1", "r2"}


def test_reworded_title_needs_authors():
    sheet_title = "Deep learning for retinal image classification in clinics"
    poster_title = "Deep learning for retinal image classification"
    without = pm.match_rows_to_posters([row("r1", sheet_title)], [poster("a.pdf", poster_title)])
    assert without["matches"] == []
    assert without["unmatched_rows"][0]["closest"]["file_name"] == "a.pdf"  # shown as a suggestion

    with_authors = pm.match_rows_to_posters(
        [row("r1", sheet_title, ["Jun Park", "Mia Lopez"])],
        [poster("a.pdf", poster_title, ["Jun Park", "Mia L\u00f3pez"])],
    )
    assert got(with_authors) == {"r1": ("a.pdf", "title_authors", "review")}


def test_different_posters_do_not_match():
    out = pm.match_rows_to_posters([row("r1", "Protein folding with graphs", ["Ana Silva"])],
                                   [poster("a.pdf", "Urban heat islands in summer", ["Ana Silva"])])
    assert out["matches"] == [] and out["unmatched_files"] == ["a.pdf"]


def test_family_name_forms():
    assert pm.family_name("Silva, Ana") == "silva"
    assert pm.family_name("Ana M. Silva") == "silva"
    assert pm.author_score(["Ana Silva", "Bo Chen"], ["A. Silva", "B. Chen", "C. Wu"]) == 0.8
    assert pm.author_score([], ["A. Silva"]) is None


def test_a_few_hundred_posters_match_quickly():
    rnd = random.Random(1)
    vocab = [f"w{i}" for i in range(400)]
    titles = [" ".join(rnd.sample(vocab, 9)) for _ in range(300)]
    rows = [row(f"r{i}", t) for i, t in enumerate(titles)]
    posters = [poster(f"{i}.pdf", t) for i, t in enumerate(titles)]
    start = time.time()
    out = pm.match_rows_to_posters(rows, posters)
    assert time.time() - start < 5
    assert len(out["matches"]) == 300
