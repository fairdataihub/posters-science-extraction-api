"""EasyChair exports: known template, filters, form fields, author table, file numbers."""

import io
import json
import os
import sys

import openpyxl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sheet_mapper as sm  # noqa: E402
from test_sheet_mapper import CURIOUS_HEADERS, curious_rows, xlsx  # noqa: E402

# Column headings of a real EasyChair basic submission download (BOSC 2026).
EASYCHAIR_HEADERS = [
    "#", "Track #", "Track name", "Title", "Authors", "Submitted", "Last updated", "Form fields",
    "Keywords", "Decision", "Notified", "Reviews sent", "Abstract", "Deleted",
]

PRIVATE_FORM = "(fee-reason) SECRET-FEE-REASON\n(diversity) SECRET-DIVERSITY\n(comments) SECRET-COMMENT"


def easychair_row(num, title, authors, decision="accept", form="", keywords="", deleted=""):
    return [num, 3, "BOSC", title, authors, "2026-03-01 10:00", "2026-03-02 10:00", form,
            keywords, decision, "yes", "yes", f"Abstract {num}.", deleted]


def easychair_basic():
    return xlsx(EASYCHAIR_HEADERS, [
        easychair_row(17, "Open tools for genomes", "Ana Silva, Bo Chen and Carl de Vries",
                      form="(doi) (2024) 10.1000/xyz123.\n(source) https://github.com/x/y\n(license) MIT\n" + PRIVATE_FORM,
                      keywords="workflows\nAI in Law, Justice\nreproducibility"),
        easychair_row(18, "Rejected work", "Dee Fox", decision="reject"),
        easychair_row(19, "Deleted work", "Eve Ng", deleted="yes"),
        easychair_row(20, "A talk only", "Fay Wu", form="(Submission Type) TALK ONLY"),
        easychair_row(21, "Accepted as poster", "Gus Ito", decision="accept as poster"),
    ], title="Submissions")


def test_basic_download_is_a_known_template():
    result = sm.map_sheet(easychair_basic(), "bosc.xlsx", ["bosc2026_submission_17.pdf", "21_ito.pdf"])
    assert result["template"]["id"] == "easychair" and result["template"]["known"] is True
    assert result["skipped"] == {"deleted": 1, "not_accepted": 1, "talk_only": 1}
    assert result["file_column"] == {"column": "#", "method": "number", "matched_rows": 2, "rows": 2}
    assert result["match_rows"] == []
    assert result["needs_confirmation"] is False  # numbered files are EasyChair's documented link

    about = {r["file_name"]: r for r in result["sheets"]["About the Poster"]}
    assert set(about) == {"bosc2026_submission_17.pdf", "21_ito.pdf"}
    assert about["bosc2026_submission_17.pdf"]["keywords"] == "workflows; AI in Law, Justice; reproducibility; BOSC"

    creators = [r for r in result["sheets"]["Creators"] if r["file_name"] == "bosc2026_submission_17.pdf"]
    assert [(c["given_name"], c["family_name"]) for c in creators] == [
        ("Ana", "Silva"), ("Bo", "Chen"), ("Carl", "de Vries")]

    related = {(r["relation_type"], r["related_identifier"]) for r in result["sheets"]["Related resources"]}
    assert related == {("IsDerivedFrom", "10.1000/xyz123"), ("IsSupplementedBy", "https://github.com/x/y")}
    assert result["hints"] == {"row-2": {"project_license": "MIT"}}  # a hint, never the poster license


def test_private_form_fields_never_leave_the_mapper():
    result = sm.map_sheet(easychair_basic(), "bosc.xlsx", ["17.pdf"])
    assert "SECRET" not in json.dumps(result)


def test_conference_data_workbook_joins_authors():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Submissions"
    ws.append(EASYCHAIR_HEADERS)
    ws.append(easychair_row(5, "Graphs of proteins", "Ana Silva and Bo Chen"))
    authors = wb.create_sheet("Authors")
    authors.append(["Submission #", "First name", "Last name", "Email", "Country", "Affiliation", "Web page",
                    "Person #", "Corresponding?"])
    authors.append([5, "Ana", "Silva", "a@x.org", "Brazil", "Univ A", "https://orcid.org/0000-0002-1825-009x", 1, "yes"])
    authors.append([5, "Bo", "Chen", "", "China", "Inst B", "https://bo.example.org", 2, "no"])
    buf = io.BytesIO()
    wb.save(buf)
    result = sm.map_sheet(buf.getvalue(), "conf.xlsx", ["005_poster.pdf"])
    assert result["sheet_name"] == "Submissions" and result["authors_table"] == "Authors"
    creators = result["sheets"]["Creators"]
    assert [(c["given_name"], c["family_name"], c["contributor_affiliation_1_name"]) for c in creators] == [
        ("Ana", "Silva", "Univ A"), ("Bo", "Chen", "Inst B")]
    assert creators[0]["name_identifier"] == "https://orcid.org/0000-0002-1825-009X"
    assert "name_identifier" not in creators[1]  # a personal web page is not an identifier
    assert "a@x.org" not in json.dumps(result)


def test_csv_pair():
    sub = (
        "#,title,authors,submitted,last updated,form fields,keywords,decision,notified,reviews sent,abstract,deleted?\n"
        '3,A study,"Kim Park and Li Wu",d,d,,"[\'x\', \'y\']",accept,no,no,Text,no\n'
    )
    au = (
        "submission #,first name,last name,email,country,affiliation,Web page,person #,corresponding?\n"
        "3,Kim,Park,k@x.org,Korea,Lab K,,1,yes\n3,Li,Wu,,China,Lab L,,2,no\n"
    )
    result = sm.map_sheet(sub.encode(), "submission.csv", [], extra_files=[(au.encode(), "author.csv")])
    assert result["template"]["id"] == "easychair"
    assert result["sheets"]["About the Poster"][0]["keywords"] == "x; y"
    assert [m["authors"] for m in result["match_rows"]] == [["Kim Park", "Li Wu"]]
    assert [c["contributor_affiliation_1_name"] for c in result["sheets"]["Creators"]] == ["Lab K", "Lab L"]


@pytest.mark.parametrize("name, expected", [
    ("bosc2026_submission_17.pdf", ["17"]),
    ("17_smith.pdf", ["17"]),
    ("ISMB20262153PatelPoster.pdf", ["20262153", "2153"]),
    ("ISMBECCB2025_879_Portillo_poster.pdf", ["2025", "879"]),
    ("poster.pdf", []),
])
def test_file_numbers(name, expected):
    assert sm.file_numbers(name) == expected


def test_number_link_skips_a_file_that_fits_two_rows():
    data = "#,Title\n2025,First\n879,Second\n"
    result = sm.map_sheet(data.encode(), "s.csv", ["ISMBECCB2025_879_x.pdf", "879.pdf"])
    about = {r["row_key"]: r["file_name"] for r in result["sheets"]["About the Poster"]}
    assert about == {"row-2": "", "row-3": "879.pdf"}


def test_number_link_outside_easychair_needs_a_check():
    result = sm.map_sheet(xlsx(CURIOUS_HEADERS, curious_rows(False)), "c.xlsx", ["01_jane.pdf", "02_john.pdf"])
    assert result["file_column"] == {"column": "Position", "method": "number", "matched_rows": 2, "rows": 2}
    assert result["needs_confirmation"] is True


def test_position_values_equal_to_file_names_need_a_check():
    result = sm.map_sheet(xlsx(CURIOUS_HEADERS, curious_rows(False)), "c.xlsx", ["1.pdf", "2.pdf"])
    assert result["file_column"]["column"] == "Position" and result["file_column"]["method"] == "column"
    assert result["needs_confirmation"] is True


def test_form_field_samples_show_questions_only():
    result = sm.map_sheet(easychair_basic(), "bosc.xlsx", ["17.pdf"])
    form = next(m for m in result["mapping"] if m["column"] == "Form fields")
    assert form["samples"] == ["comments, diversity, doi, fee reason, license, source, submission type (answers not shown)"]
