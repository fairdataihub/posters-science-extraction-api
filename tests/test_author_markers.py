"""Affiliation marks around author names: brackets, carets and footnote symbols.

A worksheet is refused when an author doesn't read as one person, so marks
like "Smith (1)" must come out as the name plus its affiliation numbers, not
as part of the name ("Smith (1)") or a stray author ("2)").
"""

import io
import os
import sys

import openpyxl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sheet_mapper as sm  # noqa: E402


@pytest.mark.parametrize("cell, expected", [
    ("John Smith (1), Ann Lee (2)", [("John Smith", [1]), ("Ann Lee", [2])]),
    ("John Smith (1,2), Ann Lee (2)", [("John Smith", [1, 2]), ("Ann Lee", [2])]),
    ("John Smith ( 1, 2 ), Ann Lee (2)", [("John Smith", [1, 2]), ("Ann Lee", [2])]),
    ("John Smith [1;2]; Ann Lee [2]", [("John Smith", [1, 2]), ("Ann Lee", [2])]),
    ("John Smith^1,2, Ann Lee^2", [("John Smith", [1, 2]), ("Ann Lee", [2])]),
    ("Smith, John (1); Doe, Jane (2)", [("Smith, John", [1]), ("Doe, Jane", [2])]),
    ("J. Smith*, A. Lee†, B. Chen‡", [("J. Smith", []), ("A. Lee", []), ("B. Chen", [])]),
    ("John Smith¹§, Ann Lee²¶", [("John Smith", [1]), ("Ann Lee", [2])]),
])
def test_bracketed_and_caret_marks_are_affiliation_numbers(cell, expected):
    authors = sm.parse_authors(cell)
    assert [(a["name"], a["numbers"]) for a in authors] == expected
    assert not any(sm._not_one_person(a["name"]) for a in authors)


@pytest.mark.parametrize("cell, names", [
    # Brackets around words are not marks; they stay in the name.
    ("John Smith (Harvard), Ann Lee", ["John Smith (Harvard)", "Ann Lee"]),
    # Plain trailing marks still work as before.
    ("John Do1, Ann Lee2, Bo Chen1,2", ["John Do", "Ann Lee", "Bo Chen"]),
    ("Smith, John and Ann Lee", ["Smith, John", "Ann Lee"]),
    ("Lee, Chen and Wu", ["Lee", "Chen", "Wu"]),
])
def test_other_author_cells_are_unchanged(cell, names):
    assert [a["name"] for a in sm.parse_authors(cell)] == names


def _curious_xlsx(authors, affiliations):
    headers = ["Position", "Track", "Title", "Authors", "Contact Author", "Contact E-Mail", "Affiliation",
               "Abstract", "Website", "LinkedIn", "Instagram", "Bluesky", "X", "Comment"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)
    ws.append([1, "Health", "A poster", authors, None, None, affiliations, "A summary.",
               None, None, None, None, None, None])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_worksheet_with_bracketed_marks_is_not_refused():
    data = _curious_xlsx("John Smith (1), Ann Lee (1,2)", "1 University A\n2 Institute B")
    result = sm.map_sheet(data, "posters.xlsx", [], expected_format="curious")
    creators = result["sheets"][sm.SHEET_CREATORS]
    people = [(c["given_name"], c["family_name"]) for c in creators]
    assert people == [("John", "Smith"), ("Ann", "Lee")]
    assert "Institute B" in str(creators[1])
