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


def test_reworded_title_is_matched_for_review():
    sheet_title = "Deep learning for retinal image classification in clinics"
    poster_title = "Deep learning for retinal image classification"
    without = pm.match_rows_to_posters([row("r1", sheet_title)], [poster("a.pdf", poster_title)])
    assert got(without) == {"r1": ("a.pdf", "words_fuzzy", "review")}  # bag of words, title cut short

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


def test_ocr_slips_and_word_order_match_by_bag_of_words():
    out = pm.match_rows_to_posters(
        [row("r1", "Single cell atlas of the human retina"), row("r2", "Graph models for drug repurposing")],
        [poster("a.pdf", "Human retina: a single-ce1l atlas"), poster("b.pdf", "Graph rnodels for drug repurposlng")],
    )
    assert got(out) == {"r1": ("a.pdf", "words_fuzzy", "review"), "r2": ("b.pdf", "words_fuzzy", "review")}


def test_loop_restarts_after_each_match():
    # r2's title is close to both posters until r1 takes x.pdf by an exact title.
    rows = [row("r1", "Soil microbes under drought stress"), row("r2", "Soil microbes under drought")]
    posters = [poster("x.pdf", "Soil microbes under drought stress"), poster("y.pdf", "Soil microbes in drought")]
    out = pm.match_rows_to_posters(rows, posters)
    assert got(out)["r1"] == ("x.pdf", "title", "high")
    assert got(out)["r2"][0] == "y.pdf"
    assert out["rounds"] >= 2 and out["unmatched_files"] == []


def test_best_remaining_needs_mutual_best_and_a_floor():
    out = pm.match_rows_to_posters(
        [row("r1", "Coral reef recovery after bleaching events in the Pacific")],
        [poster("a.pdf", "Pacific coral reef surveys 2025")],
    )
    assert got(out) == {"r1": ("a.pdf", "best_remaining", "review")}
    far = pm.match_rows_to_posters([row("r1", "Coral reef recovery after bleaching")], [poster("a.pdf", "Reef fish counts")])
    assert far["matches"] == [] and far["unmatched_rows"][0]["closest"]["file_name"] == "a.pdf"


def test_identical_titles_without_authors_stay_unmatched_after_others_match():
    rows = [row("r1", "Annual lab report"), row("r2", "Annual lab report"), row("r3", "Ocean acidity sensors")]
    posters = [poster("x.pdf", "Annual lab report"), poster("y.pdf", "Annual lab report"), poster("z.pdf", "Ocean acidity sensors")]
    out = pm.match_rows_to_posters(rows, posters)
    assert got(out) == {"r3": ("z.pdf", "title", "high")}
    assert sorted(out["unmatched_files"]) == ["x.pdf", "y.pdf"]


def test_paired_rows_get_a_title_check():
    pairs = [
        {"row_key": "r1", "file_name": "1.pdf", "title": "Ocean sensor networks for coastal monitoring", "method": "column"},
        {"row_key": "r2", "file_name": "2.pdf", "title": "Deep learning for protein folding", "method": "number"},
        {"row_key": "r3", "file_name": "3.pdf", "title": "Soil carbon maps", "method": "number"},
    ]
    posters = [
        # cut short and an OCR slip: still the same poster
        {"file_name": "1.pdf", "title": "Ocean sensr networks for coastal"},
        # the wrong file
        {"file_name": "2.pdf", "title": "A survey of library open access policies"},
    ]
    out = pm.match_rows_to_posters([], posters, pairs)
    status = {c["row_key"]: (c["status"], c["reason"]) for c in out["pair_checks"]}
    assert status == {"r1": ("agree", None), "r2": ("disagree", "title"), "r3": ("unchecked", "not_extracted")}
    assert [c["row_key"] for c in out["flagged"]] == ["r2"]
    # Paired files are out of play for the matching loop.
    assert out["unmatched_files"] == []


def test_pair_with_no_author_in_common_is_flagged():
    pair = {"row_key": "r", "file_name": "a.pdf", "title": "Ocean sensor networks", "authors": ["Jane Smith"]}
    poster = {"file_name": "a.pdf", "title": "Ocean sensor networks", "authors": ["Wei Zhang", "Ana Costa"]}
    assert pm.check_pair(pair, poster)["reason"] == "authors"


def test_suggested_file_taken_for_review_when_title_agrees():
    rows = [
        {"row_key": "r1", "title": "Ocean sensor networks for coastal monitoring", "suggested_file": "poster (1).pdf"},
        {"row_key": "r2", "title": "Deep learning for protein folding", "suggested_file": "poster (2).pdf"},
    ]
    posters = [
        {"file_name": "poster (1).pdf", "title": "Ocean sensor networks for coastal monitoring"},
        {"file_name": "poster (2).pdf", "title": "A survey of library open access policies"},
    ]
    out = pm.match_rows_to_posters(rows, posters)
    by_row = {m["row_key"]: m for m in out["matches"]}
    assert by_row["r1"]["method"] == "number_suggested" and by_row["r1"]["confidence"] == "review"
    assert "r2" not in by_row
