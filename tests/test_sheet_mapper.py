"""Tests for sheet_mapper: template detection, file column, value parsers."""

import io
import os
import sys

import openpyxl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sheet_mapper as sm  # noqa: E402

CURIOUS_HEADERS = [
    "Position", "Track", "Title", "Authors", "Contact Author", "Contact E-Mail", "Affiliation",
    "Abstract", "Website", "LinkedIn", "Instagram", "Bluesky", "X", "Comment",
]


def xlsx(headers, rows, title="Posters"):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title
    ws.append(headers)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def curious_rows(with_file=True):
    rows = [
        [1, "Human Health", "1 \u2013 Poster Title 1", "Jane Do, Another Name", "Jane Do", "jane@test.de",
         "Test University, Department of Science, Germany", "A summary.", "www.test-website.de",
         "www.linkedin.com/in/jane-do", "no", "no", "no", None],
        [2, "Vibrant Digital", "2 \u2013 Making AI efficient", "John Do1, Ann Lee2, Bo Chen1,2", "John Do", None,
         "1 Department of Digitalization, University of Sciences, Mainz\n2 Max Planck Institute, Germany",
         "A maximum of 250 words.", "no", "no", "no", "no", "@JohnDo", None],
        [3] + [None] * 13,  # template filler: position only
    ]
    if with_file:
        rows[0].append("1_jane.pdf")
        rows[1].append("posters/2_john.PDF")
        rows[2].append(None)
    return rows


def test_curious_without_file_column_waits_for_title_matching():
    result = sm.map_sheet(xlsx(CURIOUS_HEADERS, curious_rows(False)), "c.xlsx", ["jane.pdf", "john.pdf"])
    assert result["template"]["id"] == "curious-2027"
    assert result["file_column"] is None
    assert result["ok"] is True
    assert result["needs_confirmation"] is True
    assert any(w["code"] == "file_column_missing" and w["severity"] == "warning" for w in result["warnings"])
    assert [(m["row_key"], m["title"], m["authors"]) for m in result["match_rows"]] == [
        ("row-2", "Poster Title 1", ["Jane Do", "Another Name"]),
        ("row-3", "Making AI efficient", ["John Do", "Ann Lee", "Bo Chen"]),
    ]
    about = result["sheets"]["About the Poster"]
    assert [(r["row_key"], r["file_name"]) for r in about] == [("row-2", ""), ("row-3", "")]
    assert {r["row_key"] for r in result["sheets"]["Creators"]} == {"row-2", "row-3"}


def test_curious_with_file_column_maps_to_grid_rows():
    headers = CURIOUS_HEADERS + ["Poster file"]
    files = ["1_jane.pdf", "2_john.pdf", "3_extra.pdf"]
    result = sm.map_sheet(xlsx(headers, curious_rows()), "c.xlsx", files)
    assert result["ok"] is True
    assert result["template"]["known"] is True
    assert result["file_column"] == {"column": "Poster file", "method": "column", "methods": {"column": 2}, "matched_rows": 2, "rows": 2}
    assert result["match_rows"] == []

    about = {r["file_name"]: r for r in result["sheets"]["About the Poster"]}
    assert about["1_jane.pdf"]["title"] == "Poster Title 1"  # position prefix removed
    assert about["2_john.pdf"]["title"] == "Making AI efficient"
    assert about["1_jane.pdf"]["keywords"] == "Human Health"  # track kept as a keyword

    creators = [r for r in result["sheets"]["Creators"] if r["file_name"] == "2_john.pdf"]
    assert [(c["given_name"], c["family_name"]) for c in creators] == [("John", "Do"), ("Ann", "Lee"), ("Bo", "Chen")]
    assert creators[0]["contributor_affiliation_1_name"].startswith("Department of Digitalization")
    assert creators[1]["contributor_affiliation_1_name"] == "Max Planck Institute, Germany"
    assert creators[2]["contributor_affiliation_2_name"] == "Max Planck Institute, Germany"

    jane = [r for r in result["sheets"]["Creators"] if r["file_name"] == "1_jane.pdf"]
    assert all(r["contributor_affiliation_1_name"].startswith("Test University") for r in jane)

    extra = {r["file_name"]: r for r in result["sheets"]["Additional information"]}
    assert extra["1_jane.pdf"]["submission_abstract"] == "A summary."

    related = result["sheets"]["Related resources"]
    assert related == [{
        "row_key": "row-2", "file_name": "1_jane.pdf", "related_identifier": "https://www.test-website.de",
        "related_identifier_type": "URL", "relation_type": "References", "resource_type_general": "Other",
    }]  # "no" in row 2 is a placeholder, not a website

    assert result["unmatched_files"] == ["3_extra.pdf"]
    stored = {m["column"]: m["stored"] for m in result["mapping"]}
    assert stored["LinkedIn"] is False and stored["Contact E-Mail"] is False and stored["Title"] is True


def test_file_column_found_by_values_not_header():
    data = "Paper ID,Title,Authors,Upload\n7,A study,Kim Park,C:\\talks\\Poster_7.pdf\n8,B study,Li Wu,poster_8.png\n"
    result = sm.map_sheet(data.encode(), "easy.csv", ["poster_7.pdf", "Poster_8.png"])
    assert result["file_column"]["column"] == "Upload"
    names = [r["file_name"] for r in result["sheets"]["About the Poster"]]
    assert names == ["poster_7.pdf", "Poster_8.png"]  # returned as uploaded


def test_rows_without_a_matching_file_are_reported_not_guessed():
    data = "File name\tTitle\na.pdf\tFirst\nmissing.pdf\tSecond\n"
    result = sm.map_sheet(data.encode(), "s.tsv", ["a.pdf", "b.pdf"])
    assert [r["file_name"] for r in result["sheets"]["About the Poster"]] == ["a.pdf", ""]
    assert result["unmatched_rows"] == [{"row": 3, "value": "missing.pdf", "title": "Second"}]
    assert [m["row_key"] for m in result["match_rows"]] == ["row-3"]  # left for title matching
    assert result["needs_confirmation"] is True


def test_duplicate_file_rows_keep_the_first():
    data = "File name,Title\na.pdf,First\nA.pdf,Second\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["a.pdf"])
    assert result["sheets"]["About the Poster"] == [{"row_key": "row-2", "file_name": "a.pdf", "title": "First"}]
    assert any(w["code"] == "duplicate_file" for w in result["warnings"])


def test_header_row_below_a_title_banner():
    rows = [["Curious 2027 poster list"], [], ["Filename", "Poster title", "Keywords"], ["x.pdf", "T", "a; b, c"]]
    wb = openpyxl.Workbook()
    for r in rows:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    result = sm.map_sheet(buf.getvalue(), "s.xlsx", ["x.pdf"])
    assert result["header_row"] == 3
    assert result["sheets"]["About the Poster"] == [{"row_key": "row-4", "file_name": "x.pdf", "title": "T", "keywords": "a; b; c"}]


def test_user_overrides_win():
    data = "File name,Heading,Blurb\nx.pdf,My title,Some text\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["x.pdf"], {"Heading": "title", "Blurb": "abstract"})
    assert result["sheets"]["About the Poster"][0]["title"] == "My title"
    assert result["sheets"]["Additional information"][0]["submission_abstract"] == "Some text"


def test_llm_is_off_without_url(monkeypatch):
    monkeypatch.delenv("SHEET_MAPPER_LLM_URL", raising=False)
    monkeypatch.setattr(sm.config, "get_env", lambda key, default=None: None)
    data = "File name,Heading\nx.pdf,My title\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["x.pdf"])
    heading = next(m for m in result["mapping"] if m["column"] == "Heading")
    assert heading["field"] is None
    assert result["template"]["known"] is False


def test_llm_answer_is_limited_to_our_fields(monkeypatch):
    monkeypatch.setattr(sm.config, "get_env", lambda key, default=None: "http://llm" if key == "SHEET_MAPPER_LLM_URL" else None)

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"Heading": "title", "Blurb": "made_up_field"}'}}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    data = "File name,Heading,Blurb\nx.pdf,My title,Some text\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["x.pdf"])
    by_col = {m["column"]: m for m in result["mapping"]}
    assert by_col["Heading"]["field"] == "title" and by_col["Heading"]["source"] == "llm"
    assert by_col["Blurb"]["field"] is None
    assert result["needs_confirmation"] is True


@pytest.mark.parametrize(
    "cell,expected",
    [
        ("John Do1, Ann Lee2", [("John Do", [1]), ("Ann Lee", [2])]),
        ("Ann Lee1,2, Bo Chen3", [("Ann Lee", [1, 2]), ("Bo Chen", [3])]),
        ("Ann Lee\u00b9\u00b2; Bo Chen\u00b3", [("Ann Lee", [12]), ("Bo Chen", [3])]),
        ("Ann Lee* and Bo Chen", [("Ann Lee", []), ("Bo Chen", [])]),
        ("Author  Six6", [("Author Six", [6])]),
    ],
)
def test_parse_authors(cell, expected):
    assert [(a["name"], a["numbers"]) for a in sm.parse_authors(cell)] == expected


def test_run_together_numbers_split_only_when_needed():
    assert sm.resolve_numbers([12], {1, 2, 3}) == [1, 2]
    assert sm.resolve_numbers([12], set(range(1, 13))) == [12]
    rows = sm._creator_rows({"row_key": "row-2", "file_name": "a.pdf"}, "Ann Lee¹²", "¹ Lab A \n² Lab B", lambda *a: None)
    assert rows[0]["contributor_affiliation_1_name"] == "Lab A"
    assert rows[0]["contributor_affiliation_2_name"] == "Lab B"


@pytest.mark.parametrize(
    "name,expected",
    [("Jane Do", ("Jane", "Do")), ("Do, Jane", ("Jane", "Do")), ("Anna van der Berg", ("Anna", "van der Berg")), ("Plato", ("", "Plato"))],
)
def test_split_name(name, expected):
    assert sm.split_name(name) == expected


def test_parse_affiliations():
    numbered, plain = sm.parse_affiliations("1. Dept A, Uni X; 2. Dept B\n3) Lab C")
    assert numbered == {1: "Dept A, Uni X", 2: "Dept B", 3: "Lab C"} and plain == []
    numbered, plain = sm.parse_affiliations("Test University, Department of Science, 69120 Heidelberg, Germany")
    assert numbered == {} and plain == ["Test University, Department of Science, 69120 Heidelberg, Germany"]


def test_position_prefix_only_when_most_titles_have_one():
    assert sm.strip_position_prefixes(["1 - A", "2 \u2013 B"]) == ["A", "B"]
    assert sm.strip_position_prefixes(["3D printing in 2024", "Plain"]) == ["3D printing in 2024", "Plain"]


def test_file_cell_with_a_typo_links_to_the_close_file_for_a_check():
    data = "File name,Title\nsmith_poster_v2.pdf,First\njones-poster.pdf,Second\nzzz.pdf,Third\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["smith_poster_v3.pdf", "jones_poster.pdf", "lee_poster.pdf"])
    names = [r["file_name"] for r in result["sheets"]["About the Poster"]]
    assert names == ["smith_poster_v3.pdf", "jones_poster.pdf", ""]
    assert result["file_column"]["methods"] == {"column": 1, "column_fuzzy": 1}
    assert any(w["code"] == "files_linked_fuzzy" for w in result["warnings"])
    assert result["needs_confirmation"] is True
    assert [m["row_key"] for m in result["match_rows"]] == ["row-4"]  # left for title matching


def test_rows_with_blank_file_cells_fall_back_to_numbers():
    data = "Poster number,File name,Title\n1,a.pdf,First\n2,,Second\n3,,Third\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["a.pdf", "2_jones.pdf", "submission_3.pdf"])
    names = [r["file_name"] for r in result["sheets"]["About the Poster"]]
    assert names == ["a.pdf", "2_jones.pdf", "submission_3.pdf"]
    assert result["file_column"]["methods"] == {"column": 1, "number": 2}
    assert result["match_rows"] == []
    # Every paired row goes back for a title check after extraction.
    assert [(p["row_key"], p["method"]) for p in result["pairs"]] == [
        ("row-2", "column"), ("row-3", "number"), ("row-4", "number"),
    ]


def test_number_inside_a_name_is_only_a_suggestion():
    # "poster (2).pdf" is a browser's download copy, not submission 2.
    data = "Poster number,File name,Title\n1,a.pdf,First\n2,,Second\n3,,Third\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["a.pdf", "Smith_poster (2).pdf", "ISMB20263Lee.pdf"])
    names = [r["file_name"] for r in result["sheets"]["About the Poster"]]
    assert names == ["a.pdf", "", ""]
    assert [(m["row_key"], m.get("suggested_file")) for m in result["match_rows"]] == [
        ("row-3", "Smith_poster (2).pdf"), ("row-4", "ISMB20263Lee.pdf"),
    ]
    assert any(w["code"] == "files_suggested_by_number" for w in result["warnings"])
    assert result["needs_confirmation"] is True


def test_easychair_number_links_only_plain_names():
    data = (
        "#,Title,Authors,Submitted,Last updated,Keywords,Decision,Abstract\n"
        "17,First,A B,x,x,k,accept poster,abs\n"
        "18,Second,C D,x,x,k,accept poster,abs\n"
        "19,Third,E F,x,x,k,accept poster,abs\n"
    )
    files = ["17_smith.pdf", "bosc_submission_18.pdf", "poster (19).pdf"]
    result = sm.map_sheet(data.encode(), "s.csv", files)
    names = [r["file_name"] for r in result["sheets"]["About the Poster"]]
    assert names == ["17_smith.pdf", "bosc_submission_18.pdf", ""]
    assert result["match_rows"][0]["suggested_file"] == "poster (19).pdf"


@pytest.mark.parametrize("name, number", [
    ("17.pdf", "17"), ("17_patel.pdf", "17"), ("017-x.pdf", "17"), ("bosc_submission_17.pdf", "17"),
    ("Smith_poster (2).pdf", None), ("ISMB20262153Patel.pdf", None), ("2026_smith_poster.pdf", None),
    ("poster.pdf", None),
])
def test_file_number_only_for_plain_names(name, number):
    assert sm.file_number(name) == number


def test_doi_column_is_the_posters_own_doi():
    data = "File name,Title,DOI\na.pdf,First,https://doi.org/10.5281/zenodo.123\nb.pdf,Second,\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["a.pdf", "b.pdf"])
    assert result["hints"] == {"row-2": {"doi": "10.5281/zenodo.123"}}
    assert "Related resources" not in result["sheets"]
    assert any(w["code"] == "existing_doi" for w in result["warnings"])


def test_any_authors_table_is_joined():
    posters = "Poster ID,Title\n7,First\n8,Second\n"
    authors = "Poster ID,Name,Institution,ORCID\n7,Jane Doe,Uni X,0000-0002-1825-0097\n7,John Roe,Uni Y,\n8,Ann Lee,Uni Z,\n"
    result = sm.map_sheet(posters.encode(), "posters.csv", ["7.pdf", "8.pdf"], extra_files=[(authors.encode(), "authors.csv")])
    creators = result["sheets"]["Creators"]
    assert [(c["file_name"], c["given_name"], c["family_name"]) for c in creators] == [
        ("7.pdf", "Jane", "Doe"), ("7.pdf", "John", "Roe"), ("8.pdf", "Ann", "Lee"),
    ]
    assert creators[0]["name_identifier"] == "https://orcid.org/0000-0002-1825-0097"
    assert creators[0]["contributor_affiliation_1_name"] == "Uni X"
    assert result["authors_table"] == "authors"


def test_authors_table_keyed_by_file_name():
    posters = "File name,Title\na.pdf,First\n"
    authors = "File name,First name,Last name\na.pdf,Jane,Doe\n"
    result = sm.map_sheet(posters.encode(), "posters.csv", ["a.pdf"], extra_files=[(authors.encode(), "authors.csv")])
    assert [(c["given_name"], c["family_name"]) for c in result["sheets"]["Creators"]] == [("Jane", "Doe")]


def test_huge_unpacked_workbook_is_refused(monkeypatch):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/big.xml", b"0" * 2_000_000)
    monkeypatch.setattr(sm, "MAX_XLSX_UNPACKED", 1_000_000)
    with pytest.raises(ValueError, match="unpacks to"):
        sm.map_sheet(buf.getvalue(), "s.xlsx", [])


def test_row_cap(monkeypatch):
    monkeypatch.setattr(sm, "MAX_ROWS", 3)
    data = "File name,Title\n" + "".join(f"{i}.pdf,T{i}\n" for i in range(5))
    with pytest.raises(ValueError, match="more than 3 rows"):
        sm.map_sheet(data.encode(), "s.csv", [])


def test_title_index_dropped_when_it_is_the_rows_own_number():
    titles = ["12 - Ocean sensors", "P13: Soil maps", "2024: a review", "Plain title"]
    assert sm.strip_position_prefixes(titles, ["12", "13", "14", "15"]) == [
        "Ocean sensors", "Soil maps", "2024: a review", "Plain title",
    ]


def test_single_name_goes_to_family_name():
    from sheet_mapper import split_name

    assert split_name("Madonna") == ("", "Madonna")
    assert split_name(", Madonna") == ("", "Madonna")
    assert split_name("Smith, Jane") == ("Jane", "Smith")


def test_title_prefix_cut_only_for_real_indexes():
    from sheet_mapper import strip_position_prefixes

    titles = ["1 - Alpha study", "2 - Beta study", "3 - Gamma study",
              "4 - Delta study", "2019-2023 trends in care", "2026: A review"]
    out = strip_position_prefixes(titles)
    assert out[:4] == ["Alpha study", "Beta study", "Gamma study", "Delta study"]
    assert out[4] == "2019-2023 trends in care"  # a range, not an index
    assert out[5] == "2026: A review"  # a year, not an index
    # A row's own number: only that number is cut.
    assert strip_position_prefixes(["12 - Title", "7 - Other"], ["12", "8"]) == ["Title", "7 - Other"]
    assert strip_position_prefixes(["3.5 T MRI at scale"], ["3"]) == ["3.5 T MRI at scale"]
