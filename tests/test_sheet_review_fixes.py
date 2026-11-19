"""Fixes from the review of #21: decisions, names, file links, limits, matching."""

import io
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import poster_matcher as pm  # noqa: E402
import sheet_mapper as sm  # noqa: E402
from test_sheet_mapper import CURIOUS_HEADERS, curious_rows, xlsx  # noqa: E402


def about(result):
    """Linked rows only: {row_key: file_name}."""
    return {r["row_key"]: r["file_name"] for r in result["sheets"]["About the Poster"] if r["file_name"]}


def codes(result):
    return {w["code"] for w in result["warnings"]}


# Decisions


@pytest.mark.parametrize("decision", ["Not yet accepted", "accept?", "Accepted (pending)", "waitlist"])
def test_pending_decisions_are_not_accepted(decision):
    assert sm._accepted(decision) is False


def test_rejected_rows_are_skipped_when_nothing_is_accepted_yet():
    data = "#,Title,Decision\n1,One,reject\n2,Two,pending\n3,Three,\n4,Four,withdrawn\n5,Five,Not yet accepted\n"
    result = sm.map_sheet(data.encode(), "s.csv", [])
    titles = [r["title"] for r in result["sheets"]["About the Poster"]]
    assert titles == ["Two", "Three", "Five"]
    assert result["skipped"]["not_accepted"] == 2


def test_not_yet_accepted_row_is_dropped_once_others_are_accepted():
    data = "#,Title,Decision\n1,One,accept\n2,Two,Not yet accepted\n"
    result = sm.map_sheet(data.encode(), "s.csv", [])
    assert [r["title"] for r in result["sheets"]["About the Poster"]] == ["One"]


# Names, titles, affiliations


@pytest.mark.parametrize("cell, names", [
    ("Smith, John", ["Smith, John"]),
    ("van Dijk, J. R.", ["van Dijk, J. R."]),
    ("Lee, Ann1", ["Lee, Ann"]),
    ("John Smith, Jane Doe", ["John Smith", "Jane Doe"]),
    ("Smith, Jane Doe", ["Smith", "Jane Doe"]),
    ("Smith, John and Ann Lee", ["Smith", "John", "Ann Lee"]),
])
def test_single_inverted_name(cell, names):
    assert [a["name"] for a in sm.parse_authors(cell)] == names


def test_year_stays_in_title_even_when_it_is_the_poster_number():
    assert sm.strip_position_prefixes(["2026: A review"], ["2026"]) == ["2026: A review"]
    assert sm.strip_position_prefixes(["12 - A review"], ["12"]) == ["A review"]


def test_numbered_affiliations_without_author_numbers_are_kept_with_a_warning():
    warnings = []
    rows = sm._creator_rows({"file_name": "a.pdf", "row_key": "row-2"}, "Ann Lee, Bo Chen",
                            "1 Lab A\n2 Lab B", lambda *a: warnings.append(a[0]))
    assert [r.get("contributor_affiliation_2_name") for r in rows] == ["Lab B", "Lab B"]
    assert warnings == ["affiliations_unassigned"]


# Columns


def test_user_choice_wins_over_an_earlier_automatic_column():
    data = "Title,Poster title,File\nAuto,Chosen,a.pdf\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["a.pdf"], {"Poster title": "title"})
    assert result["sheets"]["About the Poster"][0]["title"] == "Chosen"


def test_position_column_matching_file_names_is_marked_as_from_values():
    result = sm.map_sheet(xlsx(CURIOUS_HEADERS, curious_rows(False)), "c.xlsx", ["1.pdf", "2.pdf"])
    position = next(m for m in result["mapping"] if m["column"] == "Position")
    assert position["source"] == "values" and result["needs_confirmation"] is True


# File links


def test_uploads_that_share_a_key_link_only_by_exact_name():
    data = "File,Title\nSmith Poster.pdf,One\nSMITH-POSTER.pdf,Two\nother.pdf,Three\n"
    files = ["Smith Poster.pdf", "smith_poster.pdf", "other.pdf"]
    result = sm.map_sheet(data.encode(), "s.csv", files)
    assert about(result) == {"row-2": "Smith Poster.pdf", "row-4": "other.pdf"}
    assert [r["title"] for r in result["match_rows"]] == ["Two"]
    assert "ambiguous_file" in codes(result)


def test_file_names_before_any_upload_are_unverified():
    data = "File,Title\nfolder/a.pdf,One\n"
    result = sm.map_sheet(data.encode(), "s.csv", [])
    assert about(result) == {"row-2": "a.pdf"}
    assert result["pairs"][0]["method"] == "column_unverified"
    assert "files_not_uploaded" in codes(result) and result["needs_confirmation"] is True


def test_a_number_shared_with_a_file_linked_row_links_nothing():
    data = "#,File,Title\n7,7.pdf,One\n7,,Two\n8,,Three\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["7.pdf", "8.pdf", "x.pdf", "y.pdf", "z.pdf"])
    assert about(result) == {"row-2": "7.pdf", "row-4": "8.pdf"}
    assert [r["title"] for r in result["match_rows"]] == ["Two"]


def test_plain_number_links_hold_when_most_files_are_named_otherwise():
    data = "#,Title\n17,One\n18,Two\n"
    files = ["17_smith.pdf", "a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    result = sm.map_sheet(data.encode(), "s.csv", files)
    assert about(result)["row-2"] == "17_smith.pdf"


def test_shared_number_does_not_pull_authors_or_topics_by_file_name():
    sub = "#,title,authors,decision\n7,One,Ann Lee,accept\n7,Two,Bo Chen,accept\n"
    au = "submission #,first name,last name,affiliation\n7,Wrong,Person,Lab X\n"
    result = sm.map_sheet(sub.encode(), "submission.csv", ["7.pdf"], extra_files=[(au.encode(), "author.csv")])
    names = {m["title"]: m["authors"] for m in result["match_rows"]}
    assert names == {"One": ["Ann Lee"], "Two": ["Bo Chen"]}
    assert "Creators" not in result["sheets"] or all(c["family_name"] != "Person" for c in result["sheets"]["Creators"])


# File types and limits


@pytest.mark.parametrize("names, fmt", [
    (["c.csv"], "curious"),
    (["a.xlsx", "b.xlsx"], "curious"),
    (["a.xlsx", "b.xlsx", "c.xlsx"], "easychair"),
    ([f"{n}.csv" for n in "abcde"], "easychair"),
])
def test_file_types_and_counts_per_format_are_refused(names, fmt):
    with pytest.raises(sm.SheetFormatError):
        sm.check_files(names, fmt)


def test_file_limits_that_pass():
    sm.check_files(["c.xlsx"], "curious")
    sm.check_files(["a.xlsx", "b.xlsx"], "easychair")
    sm.check_files(["submission.csv", "author.csv", "submission_topic.csv", "topics.csv"], "easychair")
    with pytest.raises(ValueError):
        sm.check_files([f"{n}.csv" for n in "abcdefg"], None)


def test_curious_csv_is_refused_by_map_sheet():
    with pytest.raises(sm.SheetFormatError):
        sm.map_sheet(b"Position,Title\n1,A\n", "c.csv", [], expected_format="curious")


def test_unpacked_size_counts_all_workbooks_together(monkeypatch):
    one = xlsx(["Title"], [["A"]])
    monkeypatch.setattr(sm, "MAX_XLSX_UNPACKED", int(sm._xlsx_unpacked(one) * 1.5))
    sm.read_tables([(one, "a.xlsx")])
    with pytest.raises(ValueError, match="unpacks"):
        sm.read_tables([(one, "a.xlsx"), (one, "b.xlsx")])


def test_rows_count_across_all_files(monkeypatch):
    monkeypatch.setattr(sm, "MAX_TOTAL_ROWS", 5)
    data = b"Title\nA\nB\nC\n"
    sm.read_tables([(data, "a.csv")])
    with pytest.raises(ValueError, match="together"):
        sm.read_tables([(data, "a.csv"), (data, "b.csv")])


# Matching


def test_ocr_slips_are_found_without_shared_exact_words():
    rows = [{"row_key": "r1", "title": "Microbial genomes of coastal wetlands", "authors": []}]
    posters = [{"file_name": "p.pdf", "title": "Microbiai genomcs of coasta1 wetlands", "authors": []},
               {"file_name": "q.pdf", "title": "Deep learning for protein folding", "authors": []}]
    out = pm.match_rows_to_posters(rows, posters)
    assert [(m["row_key"], m["file_name"]) for m in out["matches"]] == [("r1", "p.pdf")]


def test_near_identical_titles_stay_fast():
    rows = [{"row_key": f"r{i}", "title": f"Genome analysis of sample {i} in coastal wetland soils", "authors": []}
            for i in range(1500)]
    posters = [{"file_name": f"{i}.pdf", "title": f"Genome analysis of sample {i} in coastal wetland soils", "authors": []}
               for i in range(1500)]
    start = time.time()
    pm.match_rows_to_posters(rows, posters)
    assert time.time() - start < 30


# API


def test_api_refuses_oversized_bodies_and_too_many_files():
    pytest.importorskip("torch")
    pytest.importorskip("poster2json")
    import api
    client = api.app.test_client()
    big = {"sheet": (io.BytesIO(b"x" * (12 * 1024 * 1024)), "big.csv")}
    assert client.post("/sheets/map", data=big, content_type="multipart/form-data").status_code == 413
    many = {"sheet": [(io.BytesIO(b"Title\nA\n"), f"{n}.csv") for n in "abcdefg"], "file_names": "[]"}
    assert client.post("/sheets/map", data=many, content_type="multipart/form-data").status_code == 400
    rows = [{"row_key": f"r{i}", "title": f"T {i}"} for i in range(2500)]
    assert client.post("/sheets/match", json={"rows": rows, "posters": []}).status_code == 200
