"""
Map an organizer's poster spreadsheet (Curious, EasyChair, ...) onto
the posters.science bulk metadata grid.

The pipeline only ever works on our own format. Everything else is converted
here, before extraction starts:

1. Read the sheet (.xlsx, .xls, .csv, .tsv) and find the header row.
2. Map each column to one of our fields, in three tiers:
   a. known template: the header set matches a saved template (exact, no guessing)
   b. header aliases: normalized and fuzzy matching against a synonym list
   c. small LLM, only for columns still unmapped, and only when
      SHEET_MAPPER_LLM_URL is set. Output is constrained to our field names.
3. Link rows to poster files, each tier taking only what is still unlinked:
   a. the file column, found by its values (cells that match the uploaded file
      names, by base name: folders, extension, case and spacing ignored)
   b. a file column cell that is close to one file name only (a typo)
   c. the submission or poster number a file name starts with (or has after
      "submission"); a number elsewhere in a name is only a suggestion
   d. after extraction, title and authors (poster_matcher.py, POST /sheets/match)
   Every linked row is also title checked after extraction (pairs).
   A poster no row reaches keeps its extracted metadata only.
4. Convert values with deterministic parsers (authors with superscript
   affiliation numbers, numbered affiliation lists, position prefixes on
   titles, "no" placeholders). Nothing else is changed: a value the sheet gives
   replaces the extracted one, and an empty cell leaves extraction to fill it.

The result is keyed the way the metadata grid saves rows: sheet name to a list
of rows keyed by fillKey, each carrying the poster's file_name. The app swaps
file_name for its poster id.

No database, no storage, no model on the GPU: safe to call while extraction runs.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from dataclasses import field as dc_field

import config

# ---------------------------------------------------------------------------
# Our fields
# ---------------------------------------------------------------------------

SHEET_ABOUT = "About the Poster"
SHEET_CREATORS = "Creators"
SHEET_ADDITIONAL = "Additional information"
SHEET_RELATED = "Related resources"

# field -> what it means. Order is the order shown to the LLM and the UI.
FIELDS: dict[str, str] = {
    "file_name": "poster file name or path that links the row to an uploaded poster file",
    "title": "poster title",
    "authors": "all authors of the poster in one cell, possibly with affiliation numbers",
    "affiliations": "author affiliations, possibly a numbered list",
    "abstract": "submitted abstract or summary of the poster",
    "keywords": "keywords or tags",
    "track": "conference track, session, topic or category",
    "language": "language of the poster",
    "domain": "research field or discipline",
    "website": "project or research website URL",
    "doi": "an existing DOI for the poster",
    "contact_name": "name of the contact or presenting author",
    "contact_email": "email address of the contact author",
    "poster_number": "the organizer's own poster number, position or submission id",
    "social": "social media handle or profile link (LinkedIn, X, Bluesky, Instagram, Mastodon)",
    "comment": "free-text comment or note for the organizer",
    "decision": "review decision; only accepted rows are kept",
    "deleted": "deleted or withdrawn flag; flagged rows are skipped",
    "form_fields": "EasyChair form fields, one '(name) value' per line",
    "ignored": "a known column with nothing for the poster record (dates, review status, country)",
}

# Fields that are recognized but have no place in the poster record. They are
# reported so the user sees they were read, then dropped.
NOT_STORED = {"contact_name", "contact_email", "poster_number", "social", "comment", "decision", "deleted", "form_fields", "ignored"}

# Values that mean "empty" in organizer sheets.
PLACEHOLDERS = {"", "no", "none", "n/a", "na", "-", "--", "\u2013", "\u2014", "null", "nil", "tbd", "x"}

POSTER_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg")


def _norm_header(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    text = text.replace("#", " number ")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# Header synonyms per field (normalized form). Keep these plain and general.
ALIASES: dict[str, list[str]] = {
    "file_name": [
        "file name", "filename", "file", "poster file", "poster file name", "poster filename",
        "pdf", "pdf file", "pdf name", "file path", "path", "poster path", "upload", "attachment",
        "document", "poster pdf",
    ],
    "title": ["title", "poster title", "paper title", "submission title", "abstract title", "name of poster"],
    "authors": ["authors", "author", "author list", "author names", "authors list", "all authors", "creators", "co authors"],
    "affiliations": ["affiliation", "affiliations", "institution", "institutions", "organization", "organisation", "author affiliations"],
    "abstract": ["abstract", "summary", "submission abstract", "short abstract", "description", "poster abstract"],
    "keywords": ["keywords", "keyword", "tags", "key words", "subjects", "topics list"],
    "track": ["track", "session", "topic", "category", "theme", "topics", "area", "track name", "session name"],
    "language": ["language", "lang", "poster language"],
    "domain": ["domain", "field", "field of study", "discipline", "research field", "research area", "subject area"],
    "website": ["website", "web site", "url", "homepage", "project website", "link", "web"],
    "doi": ["doi", "existing doi"],
    "contact_name": ["contact author", "contact", "presenter", "presenting author", "corresponding author", "contact name"],
    "contact_email": ["contact e mail", "contact email", "email", "e mail", "corresponding email", "presenter email"],
    "poster_number": [
        "position", "poster number", "poster no", "no", "number", "submission id", "paper id", "id", "board",
        "board number", "submission", "submission number", "paper number",
    ],
    "social": ["linkedin", "instagram", "bluesky", "x", "twitter", "mastodon", "facebook", "social", "social media"],
    "comment": ["comment", "comments", "note", "notes", "remarks"],
    "decision": ["decision", "status", "acceptance", "review decision"],
    "deleted": ["deleted", "is deleted", "withdrawn"],
    "form_fields": ["form fields", "custom fields"],
    "ignored": [
        "submitted", "last updated", "notified", "reviews sent", "track number", "created", "updated",
        "timestamp", "date submitted", "country", "person number",
    ],
}

_ALIAS_INDEX = {alias: fld for fld, aliases in ALIASES.items() for alias in aliases}

# ---------------------------------------------------------------------------
# Known templates (tier a). Add one entry per confirmed organizer format.
# `headers` must all be present; extra columns (a file name column) are fine.
#
# Keys a template can have:
#   id, name        Template id (FORMATS maps a choosable format to it) and
#                   the name used in messages.
#   headers         {column: field}, all required in the poster table.
#   optional        {column: field}, mapped when present.
#   files           {"excel": n, "text": n}: how many Excel and CSV/TSV files
#                   the format may upload (0 refuses that type).
#   main_file       CSV file name of the poster table, when the export can
#                   come as several CSV files; it is then the required one.
#   main_sheet      Workbook sheet name of the poster table.
#   multi_file      True when only the files named here may be uploaded.
#   tables          The export's other tables: [{name, file, headers,
#                   optional}], checked by their columns.
#   file_number     True when files named by submission number link directly.
# ---------------------------------------------------------------------------

TEMPLATES: list[dict] = [
    {
        "id": "curious-2027",
        "name": "Curious poster information template",
        "headers": {
            "Position": "poster_number",
            "Track": "track",
            "Title": "title",
            "Authors": "authors",
            "Contact Author": "contact_name",
            "Contact E-Mail": "contact_email",
            "Affiliation": "affiliations",
            "Abstract": "abstract",
            "Website": "website",
            "LinkedIn": "social",
            "Instagram": "social",
            "Bluesky": "social",
            "X": "social",
            "Comment": "comment",
        },
        # Curious gives its own Excel template only.
        "files": {"excel": 1, "text": 0},
    },
    {
        # Basic submission download (one sheet) and the Submissions sheet of the
        # Conference data download. Headers from a real BOSC 2026 export.
        "id": "easychair",
        "name": "EasyChair submissions download",
        "headers": {
            "#": "poster_number",
            "Title": "title",
            "Authors": "authors",
            "Submitted": "ignored",
            "Last updated": "ignored",
            "Keywords": "keywords",
            "Decision": "decision",
            "Abstract": "abstract",
        },
        "optional": {
            "Track #": "ignored",
            "Track name": "track",
            "Form fields": "form_fields",
            "Notified": "ignored",
            "Reviews sent": "ignored",
            "Deleted": "deleted",
        },
        # Poster files named by submission number (17.pdf, bosc_submission_17.pdf)
        # are the documented way to link files, so a number link needs no check.
        "file_number": True,
        # The Excel download (the conference data workbook, sometimes with a
        # separate authors copy) or up to four CSV files.
        "files": {"excel": 2, "text": 4},
        # The Conference data workbook, or its sheets as CSV files dropped
        # together: submission.csv is this table, the rest are below.
        "main_file": "submission.csv",
        "main_sheet": "Submissions",
        "multi_file": True,
        "tables": [
            {
                "name": "Authors",
                "file": "author.csv",
                "headers": ["submission #", "first name", "last name", "affiliation"],
                "optional": ["email", "country", "Web page", "person #", "corresponding?"],
            },
            {"name": "Submission topics", "file": "submission_topic.csv", "headers": ["submission #", "topic"]},
            {"name": "Topics", "file": "topics.csv", "headers": ["topic", "header?"]},
        ],
    },
]


# Formats the organizer can choose in the app, by template id. With a format
# chosen, a sheet without that template's columns is refused.
FORMATS: dict[str, str] = {"easychair": "easychair", "curious": "curious-2027"}


# File types a sheet can come as. What each format may upload is its
# template's `files`.
EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")
TEXT_EXTENSIONS = (".csv", ".tsv")
MAX_SHEET_FILES = 6


def _template_for(fmt: str) -> dict:
    return next(t for t in TEMPLATES if t["id"] == FORMATS[fmt])


def _csv_file_names(tpl: dict) -> list[str]:
    """The CSV files a template's export splits into, main table first."""
    names = [tpl.get("main_file")] + [t.get("file") for t in tpl.get("tables", [])]
    return [n for n in names if n]


def _files_wording(tpl: dict) -> str:
    """'one Excel file (.xlsx) ...' for a format's `files`, for messages."""
    limits = tpl["files"]
    parts = []
    if limits["excel"]:
        n = limits["excel"]
        parts.append("one Excel file (.xlsx)" if n == 1 else f"up to {n} Excel files (.xlsx)")
    if limits["text"]:
        n = limits["text"]
        csv = "one CSV file" if n == 1 else f"up to {n} CSV files"
        names = _csv_file_names(tpl)
        parts.append(f"{csv} ({', '.join(names)})" if names else csv)
    wording = " or ".join(parts)
    if not limits["text"]:
        wording += ". CSV files are not accepted"
    elif not limits["excel"]:
        wording += ". Excel files are not accepted"
    return wording


def check_files(filenames: list[str], fmt: str | None) -> None:
    """Refuse file types or counts the chosen format does not use."""
    if len(filenames) > MAX_SHEET_FILES:
        raise ValueError(f"Upload at most {MAX_SHEET_FILES} sheet files at once.")
    if not fmt:
        return
    tpl = _template_for(fmt)
    limits = tpl["files"]
    excel = sum(1 for n in filenames if n.lower().endswith(EXCEL_EXTENSIONS))
    text = sum(1 for n in filenames if n.lower().endswith(TEXT_EXTENSIONS))
    other = [n for n in filenames if not n.lower().endswith(EXCEL_EXTENSIONS + TEXT_EXTENSIONS)]
    if other or excel > limits["excel"] or text > limits["text"]:
        message = f"The {tpl['name']} takes {_files_wording(tpl)}."
        issues = [sheet_issue(message)]
        issues += [sheet_issue("This file type is not a sheet this format takes.", file=os.path.basename(n))
                   for n in other]
        if not limits["text"]:
            issues += [sheet_issue("CSV files are not accepted for this format.", file=os.path.basename(n))
                       for n in filenames if n.lower().endswith(TEXT_EXTENSIONS)]
        if not limits["excel"]:
            issues += [sheet_issue("Excel files are not accepted for this format.", file=os.path.basename(n))
                       for n in filenames if n.lower().endswith(EXCEL_EXTENSIONS)]
        raise SheetFormatError(message, [], None, issues)


class SheetFormatError(ValueError):
    """The sheet is not the format the organizer chose."""

    def __init__(
        self, message: str, missing: list[str], detected: str | None, issues: list[dict] | None = None
    ):
        super().__init__(message)
        self.missing = missing
        self.detected = detected
        # Where each problem is, for the app to list: [{file, sheet, row,
        # message, missing, found}]. file, sheet and row are None when the
        # problem is not at one place (a file that is not there).
        self.issues = issues or [sheet_issue(message, missing=missing)]


class SheetReadError(ValueError):
    """A sheet file could not be read (corrupt, too large); names the file."""

    def __init__(self, message: str, file: str | None):
        super().__init__(message)
        self.file = file


def sheet_issue(
    message: str,
    *,
    file: str | None = None,
    sheet: str | None = None,
    row: int | None = None,
    missing: list[str] | None = None,
    found: list[str] | None = None,
) -> dict:
    """One rejection reason and where it is (row is 1-based, as in Excel)."""
    return {
        "message": message,
        "file": file,
        "sheet": sheet,
        "row": row,
        "missing": missing or [],
        "found": [h for h in (found or []) if str(h).strip()],
    }


def _columns_wording(missing: list[str]) -> str:
    return f"{', '.join(missing)} column{'s' if len(missing) > 1 else ''}"


def check_format(headers: list[str], fmt: str, table: "Table | None" = None) -> None:
    """Raise SheetFormatError when `headers` lack a column of format `fmt`.
    With the table, the error says which file, sheet and row the headers are on."""
    tpl = _template_for(fmt)
    present = {_norm_header(h) for h in headers}
    missing = [h for h in tpl["headers"] if _norm_header(h) not in present]
    if not missing:
        return
    other = match_template(headers)
    detected = next((k for k, v in FORMATS.items() if other and v == other["id"]), None)
    message = f"This sheet is not the {tpl['name']}: it has no {_columns_wording(missing)}."
    if detected:
        message += f" It looks like the {other['name']}; choose that format instead."
    issue = sheet_issue(
        f"The header row has no {_columns_wording(missing)}.",
        file=table.file_name or None if table else None,
        sheet=table.sheet_name if table else None,
        row=table.header_row if table else None,
        missing=missing,
        found=headers,
    )
    raise SheetFormatError(message, missing, detected, [issue])


def _first_header_row(rows: list[list[str]]) -> tuple[int, list[str]] | tuple[None, None]:
    """The first row (of the first 15) naming at least two columns, and its
    1-based row number."""
    return next(
        ((i + 1, r) for i, r in enumerate(rows[:15]) if sum(1 for c in r if str(c).strip()) >= 2), (None, None)
    )


def check_tables(files: list[tuple[bytes, str]], fmt: str) -> None:
    """
    Check the columns of a format's other tables (its template's `tables`,
    as sheets or CSV files), that every CSV file is one of the format's when
    the template is `multi_file`, and that its `main_file` is there when only
    CSV files were sent. File types and counts are check_files; the main
    table's columns are check_format. Formats without `tables` or
    `main_file` pass. Raises SheetFormatError with every problem found, not
    only the first.
    """
    tpl = _template_for(fmt)
    tables = tpl.get("tables", [])

    def stem(name: str) -> str:
        return _norm_header(os.path.splitext(os.path.basename(name))[0])

    main_names = {stem(tpl.get("main_file", "")), _norm_header(tpl.get("main_sheet", ""))} - {""}
    main_headers = {_norm_header(h) for h in tpl["headers"]}
    csv_names = _csv_file_names(tpl)
    all_csv = True
    main_found = False
    issues: list[dict] = []
    for data, filename in files:
        is_csv = filename.lower().endswith(TEXT_EXTENSIONS)
        all_csv = all_csv and is_csv
        base = os.path.basename(filename)
        for sheet_name, rows in read_sheet_file(data, filename):
            name = stem(filename) if is_csv else _norm_header(sheet_name)
            label = base if is_csv else f"The {sheet_name} sheet"
            row_no, header = _first_header_row(rows)
            present = {_norm_header(h) for h in header or []}
            if name in main_names or (present and main_headers <= present):
                main_found = True
                continue
            # Recognized by its columns first, so a renamed file still counts.
            if any(present and {_norm_header(h) for h in t["headers"]} <= present for t in tables):
                continue
            table = next(
                (t for t in tables if name in {_norm_header(t["name"]), stem(t.get("file", ""))}), None
            )
            where = {"file": base, "sheet": None if is_csv else sheet_name}
            if table:
                missing = [h for h in table["headers"] if _norm_header(h) not in present]
                issues.append(sheet_issue(
                    f"{label} is not the {table['name']} table of the {tpl['name']}: it has no "
                    f"{_columns_wording(missing)}.",
                    **where, row=row_no, missing=missing, found=header,
                ))
            elif is_csv and tpl.get("multi_file"):
                issues.append(sheet_issue(
                    f"{label} is not one of the {tpl['name']} CSV files ({', '.join(csv_names)}).",
                    **where, row=row_no, found=header,
                ))
            # Any other sheet in a workbook is left alone.
    if all_csv and tpl.get("main_file") and not main_found:
        issues.append(sheet_issue(f"Add {tpl['main_file']}: it is the one CSV file that is required."))
    if issues:
        message = issues[0]["message"]
        if len(issues) > 1:
            message += f" ({len(issues) - 1} more problem{'s' if len(issues) > 2 else ''})"
        missing = [h for issue in issues for h in issue["missing"]]
        raise SheetFormatError(message, missing, None, issues)


def read_sheet_file(data: bytes, filename: str, budget: "_Budget | None" = None) -> list[tuple[str, list[list[str]]]]:
    """_read_rows, with any read error naming the file it came from."""
    try:
        return _read_rows(data, filename, budget)
    except SheetReadError:
        raise
    except ValueError as e:
        raise SheetReadError(str(e), os.path.basename(filename)) from e
    except Exception as e:
        # openpyxl and xlrd raise their own errors for corrupt, encrypted or
        # unsupported files.
        raise SheetReadError(
            "This file could not be opened. It may be damaged or password protected; save it again from "
            "Excel (without a password) and retry.",
            os.path.basename(filename),
        ) from e


def header_fingerprint(headers: list[str]) -> str:
    """Stable id for a header row: order and case do not matter."""
    keys = sorted({_norm_header(h) for h in headers if _norm_header(h)})
    return hashlib.sha1("|".join(keys).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Reading the sheet
# ---------------------------------------------------------------------------


@dataclass
class Table:
    sheet_name: str
    header_row: int  # 1-based row number in the file
    headers: list[str]
    rows: list[tuple[int, list[str]]]  # (1-based row number, cells)
    file_name: str = ""  # the uploaded file it was read from


def _cell_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# An .xlsx is a zip, so the 10 MB upload limit says little about what openpyxl
# would unpack. A real poster sheet is a few MB at most once unpacked.
MAX_XLSX_UNPACKED = 100 * 1024 * 1024
# The largest export we expect (every EasyChair submission of a big meeting,
# rejected ones included) is a few thousand rows; 20,000 leaves ample room.
MAX_ROWS = 20_000
# Several files are read as one workbook, so the limits also hold for all of
# them together, not only for each one.
MAX_TOTAL_ROWS = 100_000
MAX_XLS_CELLS = 2_000_000


def _xlsx_unpacked(data: bytes) -> int:
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile:
        raise ValueError("This .xlsx file could not be opened. Save it again from Excel and retry.") from None


def _check_xlsx_size(data: bytes, already: int = 0) -> int:
    """Raise when this workbook, plus what earlier uploads unpacked to, is over
    the limit. Returns the new running total."""
    unpacked = already + _xlsx_unpacked(data)
    if unpacked > MAX_XLSX_UNPACKED:
        raise ValueError(
            f"This workbook unpacks to {unpacked // (1024 * 1024)} MB, over the {MAX_XLSX_UNPACKED // (1024 * 1024)} MB "
            "limit. Remove sheets or images that are not the poster list and retry."
        )
    return unpacked


class _Budget:
    """What all uploads of one request have used so far (unpacked bytes, rows)."""

    def __init__(self):
        self.unpacked = 0
        self.rows = 0


def _capped(rows, sheet_name: str, budget: _Budget | None = None) -> list:
    out = []
    for r in rows:
        if len(out) == MAX_ROWS:
            raise ValueError(f'Sheet "{sheet_name}" has more than {MAX_ROWS:,} rows. Split it or remove rows that are not posters.')
        if budget is not None:
            budget.rows += 1
            if budget.rows > MAX_TOTAL_ROWS:
                raise ValueError(
                    f"These files have more than {MAX_TOTAL_ROWS:,} rows together. Upload only the poster list "
                    "(and its authors and topics) and retry."
                )
        out.append(r)
    return out


def _read_rows(data: bytes, filename: str, budget: _Budget | None = None) -> list[tuple[str, list[list[str]]]]:
    """Return [(sheet name, rows of cell text)] for every sheet in the file.
    With a budget, the size and row limits count every file read with it."""
    budget = budget or _Budget()
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".xls" and data[:2] == b"PK":
        ext = ".xlsx"  # an .xlsx saved under an .xls name
    if ext == ".xls" and data.lstrip()[:1] == b"<":
        raise ValueError(
            "This .xls file is a web page or XML export, not an Excel workbook. Open it in Excel and save it as "
            ".xlsx or .csv, then retry."
        )
    if ext in (".csv", ".tsv", ".txt"):
        # A long abstract can pass the csv module's 128 KB default cell limit.
        csv.field_size_limit(max(csv.field_size_limit(), 10 * 1024 * 1024))
        text = data.decode("utf-8-sig", errors="replace")
        if ext == ".tsv":
            delimiter = "\t"
        else:
            try:
                delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
            except csv.Error:
                delimiter = ","
        name = os.path.splitext(os.path.basename(filename))[0]
        rows = _capped(([_cell_text(c) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)), name, budget)
        return [(name, rows)]
    if ext in (".xlsx", ".xlsm"):
        import openpyxl

        budget.unpacked = _check_xlsx_size(data, budget.unpacked)
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        return [
            (ws.title, _capped(([_cell_text(c) for c in r] for r in ws.iter_rows(values_only=True)), ws.title, budget))
            for ws in wb.worksheets
        ]
    if ext == ".xls":
        import xlrd

        # Sheets load one at a time and are released after reading, so only one
        # sheet's cells are in memory, and a sheet over the cell limit is
        # refused before its cells are copied out.
        book = xlrd.open_workbook(file_contents=data, on_demand=True)
        out = []
        try:
            for k in range(book.nsheets):
                sh = book.sheet_by_index(k)
                if sh.nrows > MAX_ROWS or sh.nrows * sh.ncols > MAX_XLS_CELLS:
                    raise ValueError(
                        f'Sheet "{sh.name}" is too large ({sh.nrows:,} rows by {sh.ncols:,} columns). Remove rows or '
                        "columns that are not the poster list and retry."
                    )
                rows = _capped(([_cell_text(sh.cell_value(i, j)) for j in range(sh.ncols)] for i in range(sh.nrows)), sh.name, budget)
                out.append((sh.name, rows))
                book.unload_sheet(k)
        finally:
            book.release_resources()
        return out
    raise ValueError(f"Unsupported sheet type: {ext or filename}. Use .xlsx, .xls, .csv or .tsv.")


def _header_score(cells: list[str]) -> int:
    """
    How much a row looks like a header row: the number of different fields
    its cells name. Placeholder words ("no", "x", "-") never count, so a first
    data row of "no" cells is not taken for the header.
    """
    fields = set()
    for c in cells:
        key = _norm_header(c)
        if key and c.strip().lower() not in PLACEHOLDERS and key in _ALIAS_INDEX:
            fields.add(_ALIAS_INDEX[key])
    return len(fields)


def _table_at(sheet_name: str, rows: list[list[str]], idx: int) -> Table:
    headers = [c.strip() for c in rows[idx]]
    while headers and not headers[-1]:
        headers.pop()
    width = len(headers)
    body = []
    for n, cells in enumerate(rows[idx + 1 :], start=idx + 2):
        cells = (cells + [""] * width)[:width]
        body.append((n, cells))
    return Table(sheet_name, idx + 1, headers, body)


# A per-author table: one row per author per poster (EasyChair's "Authors"
# sheet or author.csv, or any organizer's authors tab). It needs a column that
# names the poster (a number or a file name), a person's name, and no title
# column (that would make it the poster list itself).
POSTER_LIST_FIELDS = {"title", "abstract", "keywords", "decision", "form_fields"}
AUTHOR_KEY_NUMBER = {
    "submission number", "submission", "submission id", "paper", "paper id", "paper number", "poster number",
    "poster", "poster id", "position", "abstract id", "abstract number", "id",
}
AUTHOR_KEY_FILE = {"file name", "filename", "file", "poster file"}
AUTHOR_GIVEN = {"first name", "given name", "firstname", "given names", "forename"}
AUTHOR_FAMILY = {"last name", "family name", "lastname", "surname"}
AUTHOR_FULL = {"name", "author", "author name", "full name", "person"}
AUTHOR_AFFILIATION = {"affiliation", "affiliations", "institution", "organization", "organisation"}
AUTHOR_ORCID = {"orcid", "orcid id", "orcid url", "web page"}


def _author_columns(headers: list[str]) -> dict[str, int] | None:
    """Column index per role for an authors table, or None when it is not one."""
    idx = {_norm_header(h): i for i, h in reversed(list(enumerate(headers))) if _norm_header(h)}
    # A column only a poster list has ("Title", "Poster Title", "Abstract")
    # makes this the poster list, however it names its authors.
    if any(_ALIAS_INDEX.get(k) in POSTER_LIST_FIELDS or k in ("title", "topic") for k in idx):
        return None

    def pick(names: set[str]) -> int | None:
        return next((idx[n] for n in names if n in idx), None)

    cols = {
        "number": pick(AUTHOR_KEY_NUMBER),
        "file": pick(AUTHOR_KEY_FILE),
        "given": pick(AUTHOR_GIVEN),
        "family": pick(AUTHOR_FAMILY),
        "full": pick(AUTHOR_FULL),
        "affiliation": pick(AUTHOR_AFFILIATION),
        "orcid": pick(AUTHOR_ORCID),
    }
    has_key = cols["number"] is not None or cols["file"] is not None
    has_name = cols["family"] is not None or cols["full"] is not None
    if not (has_key and has_name):
        return None
    return {k: v for k, v in cols.items() if v is not None}


def is_author_table(headers: list[str]) -> bool:
    return _author_columns(headers) is not None


def _topic_columns(headers: list[str]) -> tuple[str, int, int] | None:
    """
    ("per_poster", number col, topic col) for EasyChair's submission_topic.csv
    (submission #, topic), ("list", topic col, header? col) for topics.csv,
    else None.
    """
    idx = {_norm_header(h): i for i, h in reversed(list(enumerate(headers))) if _norm_header(h)}
    if "topic" not in idx or "title" in idx:
        return None
    number = next((idx[n] for n in AUTHOR_KEY_NUMBER if n in idx), None)
    if number is not None and len(idx) <= 3:
        return ("per_poster", number, idx["topic"])
    if "header" in idx and len(idx) <= 3:
        return ("list", idx["topic"], idx["header"])
    return None


def read_tables(files: list[tuple[bytes, str]]) -> tuple[Table, Table | None, dict[str, list[str]]]:
    """
    Pick the poster table (the sheet and header row with the most recognizable
    headers) and, when one is there, a per-author table to join to it. Several
    files are read as one workbook, so an EasyChair export can come as one
    .xlsx or as submission.csv plus author.csv (plus submission_topic.csv and
    topics.csv, whose topics become keywords). Returns the poster table, the
    authors table and {submission number: [topics]}.
    """
    best = None
    authors = None
    per_poster: list[tuple[str, str]] = []
    group_headings: set[str] = set()
    budget = _Budget()
    for data, filename in files:
        for sheet_name, rows in read_sheet_file(data, filename, budget):
            for i, cells in enumerate(rows[:15]):
                topic = _topic_columns(cells)
                if topic:
                    kind, a, b = topic
                    for row in rows[i + 1:]:
                        row = list(row) + [""] * (max(a, b) + 1 - len(row))
                        if kind == "per_poster" and clean(row[a]) and clean(row[b]):
                            per_poster.append((clean(row[a]).lstrip("0"), clean(row[b])))
                        elif kind == "list" and clean(row[b]).lower() == "yes":
                            group_headings.add(clean(row[a]).lower())
                    break
                if is_author_table(cells):
                    if authors is None:
                        authors = _table_at(sheet_name, rows, i)
                        authors.file_name = os.path.basename(filename)
                    break
                score = _header_score(cells)
                if best is None or score > best[0]:
                    best = (score, sheet_name, i, rows, os.path.basename(filename))
    if best is None or best[0] == 0:
        raise ValueError("Could not find a header row. The first rows should name the columns.")
    _, sheet_name, idx, rows, best_file = best
    topics: dict[str, list[str]] = {}
    for number, topic in per_poster:
        if topic.lower() not in group_headings:
            topics.setdefault(number, []).append(topic)
    table = _table_at(sheet_name, rows, idx)
    table.file_name = best_file
    return table, authors, topics


def read_table(data: bytes, filename: str) -> Table:
    return read_tables([(data, filename)])[0]


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


@dataclass
class ColumnMap:
    index: int
    header: str
    field: str | None
    source: str  # template | alias | fuzzy | llm | values | none
    samples: list[str] = dc_field(default_factory=list)
    renamed: bool = False  # became the file column from another field (a number column)


def match_template(headers: list[str]) -> dict | None:
    present = {_norm_header(h) for h in headers}
    for tpl in TEMPLATES:
        need = {_norm_header(h) for h in tpl["headers"]}
        if need <= present:
            return tpl
    return None


def _template_map(tpl: dict | None) -> dict[str, str]:
    if not tpl:
        return {}
    both = {**tpl.get("optional", {}), **tpl["headers"]}
    return {_norm_header(h): f for h, f in both.items()}


def _alias_field(header: str) -> tuple[str | None, str]:
    key = _norm_header(header)
    if not key:
        return None, "none"
    if key in _ALIAS_INDEX:
        return _ALIAS_INDEX[key], "alias"
    close = difflib.get_close_matches(key, list(_ALIAS_INDEX), n=1, cutoff=0.86)
    if close:
        return _ALIAS_INDEX[close[0]], "fuzzy"
    return None, "none"


def _samples(table: Table, index: int, k: int = 3) -> list[str]:
    out = []
    for _, cells in table.rows:
        v = cells[index]
        if v and v.lower() not in PLACEHOLDERS:
            out.append(v[:160])
            if len(out) == k:
                break
    return out


def _llm_map(columns: list[ColumnMap]) -> tuple[dict[int, str | None], str | None]:
    """
    Ask a small local model to map the leftover columns. Off unless
    SHEET_MAPPER_LLM_URL points at an OpenAI-compatible server (llama.cpp
    llama-server, Ollama /v1, vLLM). A 1-2B instruct model on CPU is enough:
    it only sees headers and three sample values, and the JSON schema limits
    each answer to one of our field names or null.
    """
    base = (config.get_env("SHEET_MAPPER_LLM_URL") or "").rstrip("/")
    if not base or not columns:
        return {}, None
    import requests

    allowed = list(FIELDS)
    field_list = "\n".join(f"- {k}: {v}" for k, v in FIELDS.items())
    cols = [{"column": c.header, "samples": c.samples} for c in columns]
    schema = {
        "type": "object",
        "properties": {
            c.header: {"anyOf": [{"type": "string", "enum": allowed}, {"type": "null"}]} for c in columns
        },
        "required": [c.header for c in columns],
        "additionalProperties": False,
    }
    prompt = (
        "Map each spreadsheet column to one field, or null if none fits.\n"
        f"Fields:\n{field_list}\n\nColumns (with sample values):\n{json.dumps(cols, ensure_ascii=False)}\n\n"
        "Answer with a JSON object from column name to field name or null."
    )
    body = {
        "model": config.get_env("SHEET_MAPPER_LLM_MODEL") or "qwen2.5-1.5b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_schema", "json_schema": {"name": "mapping", "schema": schema}},
    }
    try:
        resp = requests.post(f"{base}/v1/chat/completions", json=body, timeout=30)
        resp.raise_for_status()
        answer = json.loads(resp.json()["choices"][0]["message"]["content"])
    except Exception as exc:  # the LLM is a helper; never fail the request over it
        return {}, f"Automatic column matching is unavailable ({exc.__class__.__name__})."
    out: dict[int, str | None] = {}
    for c in columns:
        value = answer.get(c.header) if isinstance(answer, dict) else None
        out[c.index] = value if value in FIELDS else None
    return out, None


def map_columns(table: Table, overrides: dict[str, str | None] | None = None) -> tuple[list[ColumnMap], dict | None, list[str]]:
    notes: list[str] = []
    tpl = match_template(table.headers)
    tpl_map = _template_map(tpl)
    columns: list[ColumnMap] = []
    for i, header in enumerate(table.headers):
        col = ColumnMap(i, header, None, "none", _samples(table, i))
        key = _norm_header(header)
        if key in tpl_map:
            col.field, col.source = tpl_map[key], "template"
        else:
            col.field, col.source = _alias_field(header)
        columns.append(col)

    leftover = [c for c in columns if c.field is None and c.header and c.samples]
    llm, note = _llm_map(leftover)
    if note:
        notes.append(note)
    for c in leftover:
        if llm.get(c.index):
            c.field, c.source = llm[c.index], "llm"

    # Confirmed choices from the user win over everything.
    for c in columns:
        if overrides and c.header in overrides:
            want = overrides[c.header]
            c.field, c.source = (want if want in FIELDS else None), "user"

    # One column per stored field, except multi-column fields.
    # A field the user chose for a column belongs to that column, so any column
    # mapped to it automatically gives it up, whichever comes first.
    multi = {"social", "comment", "keywords", "track", "ignored"}
    seen: dict[str, ColumnMap] = {}
    for c in columns:
        if c.source == "user" and c.field and c.field not in multi and c.field not in seen:
            seen[c.field] = c
    for c in columns:
        if c.field and c.field not in multi and seen.get(c.field) is not c:
            if c.field in seen:
                which = "your choice" if seen[c.field].source == "user" else "the first"
                notes.append(f'Columns "{seen[c.field].header}" and "{c.header}" both look like {c.field}; using {which}.')
                c.field, c.source = None, "duplicate"
            else:
                seen[c.field] = c
    return columns, tpl, notes


# ---------------------------------------------------------------------------
# File column
# ---------------------------------------------------------------------------


def file_key(name: str) -> str:
    """File name key: no folders, no extension, case and spacing ignored."""
    base = re.split(r"[\\/]", name.strip())[-1]
    stem, ext = os.path.splitext(base)
    if ext.lower() not in POSTER_EXTENSIONS + (".pptx", ".ppt", ".zip", ".tif", ".tiff", ".webp"):
        stem = base
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode().lower())


def file_index(file_names: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    {file key: file name} for keys only one upload has, and {key: names} for
    keys two or more uploads share ("Smith Poster.pdf" and "smith_poster.pdf").
    A shared key names no file for sure, so a sheet cell links to one of those
    files only by its exact base name.
    """
    groups: dict[str, list[str]] = {}
    for name in file_names:
        groups.setdefault(file_key(name), []).append(name)
    unique = {k: v[0] for k, v in groups.items() if len(v) == 1}
    shared = {k: v for k, v in groups.items() if len(v) > 1}
    return unique, shared


def _base(name: str) -> str:
    return re.split(r"[\\/]", name.strip())[-1]


# Fields a file column may have been mapped to, preferred first.
FILE_COLUMN_FIELDS = ("file_name", "poster_number", None)


def find_file_column(table: Table, columns: list[ColumnMap], file_names: list[str]) -> tuple[ColumnMap | None, dict[str, str]]:
    """
    The file column is decided by its values, not its header: the column whose
    cells match the most uploaded file names. Without an upload list, fall back
    to a column named like a file column whose values look like file names.
    """
    by_key, collisions = file_index(file_names)

    best: tuple[float, int, ColumnMap] | None = None
    for c in columns:
        # Only a column that can name a file: one named for files, a poster
        # number, or one we could not place. "Track #" (ignored) or a title
        # column never links rows, even when its values look like file names.
        if c.field not in FILE_COLUMN_FIELDS:
            continue
        values = [cells[c.index] for _, cells in table.rows if cells[c.index].strip()]
        if not values:
            continue
        if by_key:
            # Each file counts once, so a column that repeats a value ("1" in
            # every row) cannot pass as a file column.
            hits = len({file_key(v) for v in values} & (set(by_key) | set(collisions)))
            score = hits / len(values)
        else:
            looks = sum(1 for v in values if v.lower().endswith(POSTER_EXTENSIONS))
            score = looks / len(values) if c.field == "file_name" else 0.0
        rank = FILE_COLUMN_FIELDS.index(c.field)
        if score >= 0.5 and (best is None or (score, -rank) > (best[0], -best[1])):
            best = (score, rank, c)
    if best is None and by_key:
        # Few exact hits (typos, renamed files): still use a column headed like
        # a file column whose values look like file names. Its cells then link
        # by close base names, for the organizer to check.
        for c in columns:
            values = [cells[c.index] for _, cells in table.rows if cells[c.index].strip()]
            if c.field == "file_name" and values:
                looks = sum(1 for v in values if v.lower().endswith(POSTER_EXTENSIONS))
                if looks / len(values) >= 0.5:
                    best = (0.0, 0, c)
                    break
    if best is None:
        return None, by_key
    chosen = best[2]
    for c in columns:
        if c is not chosen and c.field == "file_name":
            c.field, c.source = None, "none"
    # A column only counts as named for files when the template or the user
    # mapped it to file_name. One picked because its values match the uploads
    # (a Curious position column next to files named 1.pdf, 2.pdf) is a good
    # guess, so its links are checked.
    named = chosen.field == "file_name" and chosen.source in ("template", "user")
    chosen.renamed = chosen.field != "file_name"
    chosen.field, chosen.source = "file_name", (chosen.source if named else "values")
    return chosen, by_key


FILE_FUZZ = 0.85  # base names this close count as a typo of each other
FILE_FUZZ_MARGIN = 0.05


def link_files_fuzzy(values: dict[int, str], file_names: list[str]) -> dict[int, str]:
    """
    Rows whose file cell names no uploaded file exactly: link a cell to the one
    file whose base name is at least 0.85 alike and leads the next by 0.05
    ("smith_poster.pdf" for "smith-poster-v2.pdf"). One file per row.
    """
    keys = {file_key(n): n for n in file_names}
    links: dict[int, str] = {}
    taken: set[str] = set()
    for i, raw in values.items():
        k = file_key(raw)
        if not k or not keys:
            continue
        ranked = sorted(((difflib.SequenceMatcher(None, k, fk).ratio(), fk) for fk in keys), reverse=True)
        best, fk = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if best >= FILE_FUZZ and best - second >= FILE_FUZZ_MARGIN and fk not in taken:
            links[i] = keys[fk]
            taken.add(fk)
    return links


def _stem(name: str) -> str:
    return os.path.splitext(re.split(r"[\\/]", name.strip())[-1])[0].lower()


def file_number(name: str) -> str | None:
    """
    The submission number a file is named by, when the name says so plainly:
    it starts with the number ("17.pdf", "17_patel.pdf", "017-x.pdf") or
    carries it after "submission" ("bosc_submission_17.pdf"), and the number
    stands alone ("3D_imaging.pdf" is not 3). Only these pair a row on their
    own.
    """
    stem = _stem(name)
    m = re.search(r"submission[\s_-]*0*(\d+)(?![a-z0-9])", stem)
    if m:
        return m.group(1)
    # The number must end at a separator or the end of the name, so
    # "3D_imaging.pdf" or "16S_rRNA.pdf" is not submission 3 or 16.
    m = re.match(r"0*(\d+)(?![a-z0-9])", stem)
    if not m:
        return None
    # "2026_smith_poster.pdf" starts with a year, not a submission number.
    if re.fullmatch(r"(19|20)\d\d", m.group(0)) and stem[4:5] and not stem[4:5].isalnum():
        return None
    return m.group(1) or "0"


def file_numbers(name: str) -> list[str]:
    """
    Every number a file name may carry, best first: the plain one from
    file_number(), then any other run of digits, and a number run together
    with a year ("ISMB20262153Patel.pdf") -> 2153 as well. Numbers found this
    way can be anything (a version, a browser's "(2)" suffix), so a pair made
    from them is only suggested for the organizer to check.
    """
    stem = _stem(name)
    if re.search(r"submission[\s_-]*\d+(?![a-z0-9])", stem):
        return [file_number(name)]
    out: list[str] = []
    plain = file_number(name)
    if plain:
        out.append(plain)
    for run in re.findall(r"\d+", stem):
        out.append(run.lstrip("0") or "0")
        if len(run) > 4 and run[:2] in ("19", "20"):
            out.append(run[4:].lstrip("0") or "0")
    return list(dict.fromkeys(out))


def link_files_by_number(
    table: Table, columns: list[ColumnMap], file_names: list[str]
) -> tuple[ColumnMap | None, dict[int, str], dict[int, str]]:
    """
    Link rows to files through the row's submission or poster number, when the
    files are named by it. Returns (column, links, suggestions):

    - links: the file name starts with the number or says "submission_17".
    - suggestions: the number only appears somewhere else in the name
      ("Smith_poster (2).pdf", "ISMB20262153Patel.pdf"). Not applied; the row
      waits for title matching, with this file as a suggestion to check.

    A file that could belong to two rows is left alone (it falls through to
    title matching), and so is a number two rows share. Rows a file column
    already linked never get here.
    """
    if not file_names:
        return None, {}, {}
    for c in columns:
        if c.field != "poster_number":
            continue
        ids: dict[str, int] = {}
        repeated: set[str] = set()
        values = 0
        for i, (_, cells) in enumerate(table.rows):
            v = cells[c.index].strip()
            if not v:
                continue
            values += 1
            if re.fullmatch(r"\d+", v):
                key = v.lstrip("0") or "0"
                if key in ids:
                    repeated.add(key)
                ids.setdefault(key, i)
        if not values or len(ids) / values < 0.8:
            continue
        # A number two rows share (positions that restart in each track) names
        # neither row for sure, so it links nothing; those rows wait for titles.
        for key in repeated:
            del ids[key]
        links: dict[int, str] = {}
        suggestions: dict[int, str] = {}
        for name in file_names:
            plain = file_number(name)
            if plain is not None and plain in ids:
                links.setdefault(ids[plain], name)
                continue
            hits = {ids[n] for n in file_numbers(name) if n in ids}
            if len(hits) == 1:
                suggestions.setdefault(hits.pop(), name)
        suggestions = {r: n for r, n in suggestions.items() if r not in links}
        # A file named plainly by a row's number ("17.pdf", "submission_17")
        # links even when most files are named some other way; every link is
        # title checked after extraction anyway. Weaker suggestions (a number
        # elsewhere in the name) count only when the column explains at least
        # half of the uploads, so a column of small numbers that happen to
        # appear in file names suggests nothing.
        if (len(links) + len(suggestions)) / len(file_names) < 0.5:
            suggestions = {}
        if links or suggestions:
            return c, links, suggestions
    return None, {}, {}


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

_SUPERSCRIPT_DIGITS = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
_SUPERSCRIPTS = str.maketrans(_SUPERSCRIPT_DIGITS, "0123456789")
_POSITION_PREFIX = re.compile(
    # The separator must not run into another digit, so a range or decimal
    # ("2019-2023 trends", "3.5 T MRI") is never read as an index.
    r"^\s*(?:(?:poster|abstract|no\.?|nr\.?|p|#)\s*)?(\d{1,5})\s*[\u2010-\u2015\-:.)|](?!\s*\d)\s*", re.I
)


def clean(value: str) -> str:
    value = re.sub(r"[ \t]+", " ", (value or "").strip())
    return "" if value.lower() in PLACEHOLDERS else value


def strip_position_prefixes(titles: list[str], numbers: list[str] | None = None) -> list[str]:
    """
    Drop the poster's index from the start of its title ("12 - ", "P12: ",
    "#12 |", "Poster 12."), as Curious does. A prefix is dropped when it is the
    row's own poster number, or when most titles in the column have one.
    """
    numbers = numbers or [""] * len(titles)
    filled = [t for t in titles if t]
    most = bool(filled) and sum(1 for t in filled if _POSITION_PREFIX.match(t)) / len(filled) >= 0.8
    out = []
    for t, n in zip(titles, numbers):
        m = _POSITION_PREFIX.match(t) if t else None
        if m and t[m.end():].strip():
            index = m.group(1).lstrip("0")
            year = bool(re.fullmatch(r"(19|20)\d\d", m.group(1)))
            if n:
                # The row has its own number: cut only when the title starts
                # with it, and never a year ("2026: A review" stays whole even
                # for poster 2026).
                cut = index == n.lstrip("0") and not year
            else:
                # No number for the row: cut when most titles carry an index,
                # but never a year ("2026: A review").
                cut = most and not year
            if cut:
                t = t[m.end():]
        out.append(t)
    return out


def _family_given_list(text: str) -> list[dict] | None:
    """
    Authors written "Family, Given" and separated by semicolons ("Smith,
    John; Doe, Jane", the usual citation form). None when the cell is not in
    that form, so the usual comma split applies. Only semicolons count: a line
    break or "and" between "John Do, Ann Lee" pairs is just as likely to
    separate two lists of "Given Family" names.
    """
    if ";" not in text:
        return None
    groups = [g.strip() for g in re.split(r"\s*;\s*", text) if g and g.strip()]
    if len(groups) < 2:
        return None
    out = []
    for g in groups:
        m = re.match(r"^(.*?[^\d\s,])\s*((?:\d{1,2}\s*,?\s*)*)$", g)
        if not m or m.group(1).count(",") != 1:
            return None
        family, given = (p.strip() for p in m.group(1).split(","))
        if not family or not given or len(family.split()) > 4 or len(given.split()) > 4 or re.search(r"\d", family + given):
            return None
        out.append({"name": f"{family}, {given}", "numbers": [int(n) for n in re.findall(r"\d{1,2}", m.group(2))]})
    return out


_NAME_PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "da", "das", "dos", "du", "di", "le", "la", "al", "el", "bin", "ter", "ten", "zu"}


def _single_inverted(text: str) -> dict | None:
    """
    One author written "Family, Given" ("Smith, John", "van Dijk, J. R.",
    "Lee, Ann1"). Read only when the cell has a single comma and nothing
    else that separates authors, the part before the comma is one word (or a
    family name with a particle like "van Dijk"), and the part after it is one
    to three given names or initials. "John Smith, Jane Doe" stays two authors.
    """
    if re.search(r"[;\n&]|\sand\s", text) or text.count(",") != 1:
        return None
    m = re.match(r"^(.*?[^\d\s,])\s*((?:\d{1,2}\s*,?\s*)*)$", text.strip())
    if not m or m.group(1).count(",") != 1:
        return None
    family, given = (p.strip() for p in m.group(1).split(","))
    fw, gw = family.split(), given.split()
    if not fw or not gw or len(gw) > 3 or re.search(r"\d", family + given):
        return None
    if len(fw) > 1 and not all(w.lower() in _NAME_PARTICLES for w in fw[:-1]):
        return None
    if len(gw) == 2 and not any(re.fullmatch(r"[A-Z]\.?(?:-?[A-Z]\.?)*", w) for w in gw):
        # "Smith, Jane Doe" reads as two authors, "Smith, J. R." as one.
        return None
    return {"name": f"{family}, {given}", "numbers": [int(n) for n in re.findall(r"\d{1,2}", m.group(2))]}


def parse_authors(cell: str) -> list[dict]:
    """
    "John Do1, Ann Lee1,2, Bo Chen\u00b3" -> [{name, numbers}]. Authors are split
    on commas, semicolons, " and " and newlines; trailing digits are affiliation
    numbers. A bare number after a comma ("Lee1, 2") belongs to the author before.
    "Smith, John; Doe, Jane" (every author as "Family, Given", split by
    semicolons, lines or "and") is two authors, kept in that form.
    """
    text = (cell or "").translate(_SUPERSCRIPTS).replace("*", "")
    inverted = _family_given_list(text)
    if inverted is not None:
        return inverted
    single = _single_inverted(text)
    if single is not None:
        return [single]
    parts = [p.strip() for p in re.split(r"\s*[,;\n]\s*|\s+and\s+|\s*&\s*", text) if p and p.strip()]
    authors: list[dict] = []
    for part in parts:
        if re.fullmatch(r"\d{1,2}", part) and authors:
            authors[-1]["numbers"].append(int(part))
            continue
        m = re.match(r"^(.*?[^\d\s])\s*((?:\d{1,2}\s*,?\s*)*)$", part)
        name, nums = (m.group(1), m.group(2)) if m else (part, "")
        name = re.sub(r"\s+", " ", name).strip()
        if not name:
            continue
        authors.append({"name": name, "numbers": [int(n) for n in re.findall(r"\d{1,2}", nums)]})
    return authors


def resolve_numbers(numbers: list[int], known: set[int]) -> list[int]:
    """
    Run-together marks ("Lee12" or superscript "\u00b9\u00b2") read as 12. When 12
    is not an affiliation number but 1 and 2 both are, it means 1 and 2.
    """
    out: list[int] = []
    for n in numbers:
        digits = [int(d) for d in str(n)]
        if n not in known and len(digits) > 1 and all(d in known for d in digits):
            out += digits
        else:
            out.append(n)
    return out


def split_name(name: str) -> tuple[str, str]:
    """"Family, Given" or "Given Middle Family" -> (given, family)."""
    if "," in name:
        family, given = [p.strip() for p in name.split(",", 1)]
        return ("", given) if given and not family else (given, family)
    parts = name.split()
    if len(parts) == 1:
        return "", parts[0]
    particles = {"van", "von", "de", "der", "den", "del", "da", "di", "du", "la", "le", "dos", "das", "ter", "ten"}
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in particles:
        i -= 1
    return " ".join(parts[:i]), " ".join(parts[i:])


def parse_affiliations(cell: str) -> tuple[dict[int, str], list[str]]:
    """
    Numbered lists ("1 Dept A\\n2 Dept B", "1. A; 2. B", "\u00b9A \u00b2B") become
    {1: "Dept A", 2: "Dept B"}. Anything unnumbered is returned as a plain list.
    """
    raw = (cell or "").strip()
    if raw[:1] in _SUPERSCRIPT_DIGITS:
        # Inline superscript marks ("¹Dept A ²Dept B"): each mark starts a
        # new affiliation, so put it on its own line before reading the list.
        raw = re.sub(f"[{_SUPERSCRIPT_DIGITS}]+", lambda m: "\n" + m.group(0).translate(_SUPERSCRIPTS) + " ", raw)
    text = raw.translate(_SUPERSCRIPTS).strip()
    if not text:
        return {}, []
    numbered = list(re.finditer(r"(?:^|[\n;])\s*(\d{1,2})[\s.):]+(?=\D)", text))
    if numbered:
        out: dict[int, str] = {}
        for k, m in enumerate(numbered):
            end = numbered[k + 1].start() if k + 1 < len(numbered) else len(text)
            value = text[m.end() : end].strip(" \n;,")
            if value:
                out[int(m.group(1))] = re.sub(r"\s+", " ", value)
        return out, []
    lines = [re.sub(r"\s+", " ", ln).strip(" ;") for ln in re.split(r"\n|;", text)]
    return {}, [ln for ln in lines if ln]


def split_keywords(cell: str) -> list[str]:
    """
    One keyword per line when the cell has line breaks (EasyChair: keywords may
    contain commas), else split on semicolons and commas. A Python-style list
    ("['a', 'b']") is read as its items.
    """
    cell = (cell or "").strip()
    if re.fullmatch(r"\[.*\]", cell, re.S):
        items = re.findall(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"", cell)
        if items:
            return [(a or b).strip() for a, b in items if (a or b).strip()]
    sep = r"\n" if "\n" in cell else r"[;,]"
    return [k.strip() for k in re.split(sep, cell) if k.strip()]


def orcid_url(value: str) -> str:
    """An ORCID iD from an ORCID or web page cell, as https://orcid.org/... ; else empty."""
    value = value or ""
    m = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])", value) or re.fullmatch(
        r"\s*(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])\s*", value
    )
    return f"https://orcid.org/{m.group(1).upper()}" if m else ""


def find_doi(value: str) -> str:
    m = re.search(r"10\.\d{4,9}/\S+", value or "")
    return m.group(0).rstrip(".,;)]") if m else ""


def parse_form_fields(cell: str) -> dict[str, str]:
    """
    EasyChair "Form fields": one "(name) value" per line; a line without a
    "(name)" continues the value above. Returns {normalized name: value}.
    """
    out: dict[str, str] = {}
    last = None
    for line in (cell or "").splitlines():
        m = re.match(r"^\s*\(([^)]+)\)\s*(.*)$", line)
        if m:
            last = _norm_header(m.group(1))
            out[last] = m.group(2).strip()
        elif last and line.strip():
            out[last] = (out[last] + "\n" + line.strip()).strip()
    return out


def read_form_fields(cell: str) -> dict:
    """
    Keep only what belongs on a poster record. Private questions (fees,
    diversity, income level, conflicts of interest, comments for the chairs)
    and consents are never read past this point.
      - submission type: TALK ONLY rows are not posters
      - previously published abstract: its DOI (IsDerivedFrom)
      - source code or content link: a URL (IsSupplementedBy)
      - license: the project's license, shown as a hint, never the poster's
    """
    keep: dict = {}
    for name, value in parse_form_fields(cell).items():
        if not value:
            continue
        if "submission type" in name:
            keep["submission_type"] = value.strip().upper()
        elif "previous" in name and ("abstract" in name or "published" in name) or name == "doi":
            doi = find_doi(value)
            if doi:
                keep["derived_from_doi"] = doi
        elif name == "source" or "source code" in name or ("link" in name and "source" in name):
            url = value.strip().split()[0]
            if re.match(r"^https?://\S+\.\S+", url, re.I):
                keep["source_url"] = url
        elif name == "license" or "license" in name:
            keep["project_license"] = value.strip()
    return keep


def as_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I):
        value = "https://" + value.lstrip("/")
    return value


# ---------------------------------------------------------------------------
# Rows in our format
# ---------------------------------------------------------------------------




def _creator_rows(link: dict, authors_cell: str, affiliations_cell: str, warn) -> list[dict]:
    label = link["file_name"] or link["row_key"]
    authors = parse_authors(authors_cell)
    numbered, plain = parse_affiliations(affiliations_cell)
    rows = []
    unassigned_numbered = False
    for a in authors:
        given, family = split_name(a["name"])
        row = {**link, "given_name": given, "family_name": family, "name_type": "Personal"}
        if numbered and a["numbers"]:
            a["numbers"] = resolve_numbers(a["numbers"], set(numbered))
            names = [numbered[n] for n in a["numbers"] if n in numbered]
            missing = [n for n in a["numbers"] if n not in numbered]
            if missing:
                warn("affiliation_number_missing", f'{label}: {a["name"]} points to affiliation {missing}, which is not in the list.')
        elif numbered and not a["numbers"] and len(numbered) == 1:
            names = list(numbered.values())
        elif numbered and not a["numbers"]:
            # Numbered affiliations but no numbers on this author: keep them all
            # for review rather than dropping them.
            names = list(numbered.values())
            unassigned_numbered = True
        elif plain and len(plain) == 1:
            names = plain
        elif plain:
            names = plain
            if len(authors) > 1:
                warn("affiliations_unassigned", f"{label}: affiliations are not numbered, so every author gets all of them. Check the Creators tab.")
        else:
            names = []
        for slot, name in enumerate(names[:30], start=1):
            row[f"contributor_affiliation_{slot}_name"] = name
        rows.append(row)
    if unassigned_numbered:
        warn(
            "affiliations_unassigned",
            f"{label}: affiliations are numbered but the authors have no numbers, so authors without one get all of "
            "them. Check the Creators tab.",
        )
    return rows


def _author_table_index(authors: Table) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """
    The poster's authors in row order, keyed two ways: by submission or poster
    number, and by file name key, whichever the authors table names posters by.
    """
    cols = _author_columns(authors.headers) or {}

    def get(cells: list[str], role: str) -> str:
        return clean(cells[cols[role]]) if role in cols else ""

    by_number: dict[str, list[dict]] = {}
    by_file: dict[str, list[dict]] = {}
    for _, cells in authors.rows:
        given, family = get(cells, "given"), get(cells, "family")
        if not family and get(cells, "full"):
            given, family = split_name(get(cells, "full"))
        if given and not family:
            # A lone name (EasyChair allows a blank first or last name) is the
            # family name, which Zenodo requires for a person.
            given, family = "", given
        if not (given or family):
            continue
        person = {
            "given": given,
            "family": family,
            "affiliation": get(cells, "affiliation"),
            "orcid": orcid_url(cells[cols["orcid"]]) if "orcid" in cols else "",
        }
        number = get(cells, "number").lstrip("0")
        if number:
            by_number.setdefault(number, []).append(person)
        file_cell = get(cells, "file")
        if file_cell:
            by_file.setdefault(file_key(file_cell), []).append(person)
    return by_number, by_file


def _creator_rows_from_table(link: dict, people: list[dict]) -> list[dict]:
    rows = []
    for p in people:
        if not (p["given"] or p["family"]):
            continue
        row = {**link, "given_name": p["given"], "family_name": p["family"], "name_type": "Personal"}
        if p["orcid"]:
            row["name_identifier"] = p["orcid"]
        if p["affiliation"]:
            row["contributor_affiliation_1_name"] = p["affiliation"]
        rows.append(row)
    return rows


# Cell values that mean "no" in a yes/no column (Excel stores FALSE, a CSV
# export may write 0 or "no").
FALSE_VALUES = {"", "no", "n", "false", "0", "none", "-", "--", "n/a", "na", "null", "nil"}


def _flag_set(value: str) -> bool:
    """A Deleted or Withdrawn cell that says yes (yes, TRUE, 1, a date, "withdrawn")."""
    return (value or "").strip().lower() not in FALSE_VALUES


_DECISION_NEGATIVE = re.compile(r"\b(?:not|non|un)\b[\s\w-]{0,15}?accept|\b(?:non|un)-?accept|reject|declin|withdr|cancel")
_DECISION_PENDING = re.compile(r"pending|undecided|to be decided|\btbd\b|under review|in review|waitlist|wait list|on hold|tentative|\?")


def _rejected(decision: str) -> bool:
    """A final no: rejected, declined, withdrawn, cancelled, not accepted."""
    d = re.sub(r"\s+", " ", decision.lower())
    return bool(re.search(r"reject|declin|withdr|cancel", d)) or (
        bool(re.search(r"\b(?:not|non|un)\b[\s\w-]{0,15}?accept|\b(?:non|un)-?accept", d))
        and not _DECISION_PENDING.search(d) and "yet" not in d
    )


def _accepted(decision: str) -> bool:
    """A final yes. "Not yet accepted", "pending" or "accept?" are not."""
    d = re.sub(r"\s+", " ", decision.lower())
    if _DECISION_NEGATIVE.search(d) or _DECISION_PENDING.search(d):
        return False
    return "accept" in d


def map_sheet(
    data: bytes,
    filename: str,
    file_names: list[str],
    overrides: dict[str, str | None] | None = None,
    extra_files: list[tuple[bytes, str]] | None = None,
    expected_format: str | None = None,
) -> dict:
    warnings: list[dict] = []

    def warn(code: str, message: str, severity: str = "warning"):
        warnings.append({"code": code, "message": message, "severity": severity})

    if expected_format is not None and expected_format not in FORMATS:
        raise ValueError(f"Unknown sheet format {expected_format!r}; expected one of {', '.join(FORMATS)}.")
    check_files([filename] + [n for _, n in (extra_files or [])], expected_format)
    if expected_format:
        check_tables([(data, filename)] + list(extra_files or []), expected_format)
    table, author_table, topics_by_number = read_tables([(data, filename)] + list(extra_files or []))
    if expected_format:
        check_format(table.headers, expected_format, table)
    columns, tpl, notes = map_columns(table, overrides)
    for n in notes:
        warn("mapping_note", n)
    file_col, by_key = find_file_column(table, columns, file_names)
    _, shared_keys = file_index(file_names)
    uploaded = set(file_names)

    # Form answers include private questions (fees, diversity, comments for the
    # chairs). Show only which questions were asked, never the answers.
    for c in columns:
        if c.field == "form_fields":
            asked = {name for _, cells in table.rows for name in parse_form_fields(cells[c.index])}
            c.samples = [", ".join(sorted(asked))[:160] + " (answers not shown)"] if asked else []

    by_field: dict[str, list[ColumnMap]] = {}
    for c in columns:
        if c.field:
            by_field.setdefault(c.field, []).append(c)

    def one(fld: str, cells: list[str]) -> str:
        cols = by_field.get(fld) or []
        return clean(cells[cols[0].index]) if cols else ""

    # Skip rows that are empty apart from a numbering column (template filler).
    id_cols = {c.index for c in by_field.get("poster_number", [])}
    data_rows = [(n, cells) for n, cells in table.rows if any(clean(v) for i, v in enumerate(cells) if i not in id_cols)]

    # Rows that are not posters: deleted, not accepted, talk only.
    skipped = {"deleted": 0, "not_accepted": 0, "talk_only": 0}
    has_decision = bool(by_field.get("decision"))
    any_accepted = has_decision and any(_accepted(one("decision", cells)) for _, cells in data_rows)
    if has_decision and not any_accepted:
        warn("no_decisions", "No row is marked accepted yet, so every row is kept except rejected or withdrawn ones.", "info")
    kept: list[tuple[int, list[str], dict]] = []
    for n, cells in data_rows:
        deleted_cols = by_field.get("deleted") or []
        if deleted_cols and _flag_set(cells[deleted_cols[0].index]):
            skipped["deleted"] += 1
            continue
        if has_decision and (
            not _accepted(one("decision", cells)) if any_accepted else _rejected(one("decision", cells))
        ):
            skipped["not_accepted"] += 1
            continue
        form = read_form_fields(one("form_fields", cells)) if by_field.get("form_fields") else {}
        if form.get("submission_type", "").startswith("TALK ONLY"):
            skipped["talk_only"] += 1
            continue
        kept.append((n, cells, form))
    reasons = {"deleted": "deleted", "not_accepted": "not accepted", "talk_only": "talk only"}
    if any(skipped.values()):
        parts = [f"{v} {reasons[k]}" for k, v in skipped.items() if v]
        warn("rows_skipped", "Skipped rows that are not accepted posters: " + ", ".join(parts) + ".", "info")

    # A sheet with no poster rows left would fill nothing: refuse it.
    if not kept:
        where = {
            "file": table.file_name or None,
            "sheet": None if (table.file_name or "").lower().endswith(TEXT_EXTENSIONS) else table.sheet_name,
        }
        if not data_rows:
            message = "The worksheet is blank apart from the header row."
            issue = message
        else:
            parts = [f"{v} {reasons[k]}" for k, v in skipped.items() if v]
            message = "The sheet has no accepted posters: every row is skipped (" + ", ".join(parts) + ")."
            issue = (
                f"All {len(data_rows)} row{'s' if len(data_rows) != 1 else ''} below the header are skipped ("
                + ", ".join(parts)
                + "). Mark the accepted posters, or export the sheet again after decisions are made."
            )
        raise SheetFormatError(message, [], None, [sheet_issue(issue, **where)])

    # A poster number two rows share (positions that restart per track) names
    # neither row for sure: no file, authors table or topic joins by it.
    counts: dict[str, int] = {}
    for _, cells, _ in kept:
        key = one("poster_number", cells).lstrip("0")
        if key:
            counts[key] = counts.get(key, 0) + 1
    repeated_numbers = {k for k, v in counts.items() if v > 1}
    if repeated_numbers:
        shown = ", ".join(sorted(repeated_numbers, key=lambda k: (len(k), k))[:10])
        warn(
            "duplicate_numbers",
            f"{len(repeated_numbers)} poster number(s) appear on more than one row ({shown}), so those rows are not "
            "linked by number. They are matched by title after extraction.",
        )

    # Link rows to files. Each tier only sees rows and files still unlinked.
    links: dict[int, str] = {}
    link_method: dict[int, str] = {}
    file_cells: dict[int, str] = {}
    ambiguous_cells: list[int] = []
    # A column picked because its values match the uploads (a number column
    # next to files named 7.pdf) links nothing for a value two rows share, and
    # that value joins no authors or topics either, like a repeated number.
    repeated_cells: set[str] = set()
    if file_col is not None and file_col.renamed:
        seen_cells: dict[str, int] = {}
        for _, cells, _ in kept:
            k = file_key(cells[file_col.index]) if cells[file_col.index].strip() else ""
            if k:
                seen_cells[k] = seen_cells.get(k, 0) + 1
        repeated_cells = {k for k, v in seen_cells.items() if v > 1}
        repeated_numbers |= {k.lstrip("0") or "0" for k in repeated_cells}
    if file_col is not None:
        for i, (row_no, cells, _) in enumerate(kept):
            raw_file = cells[file_col.index].strip()
            if not raw_file or file_key(raw_file) in repeated_cells:
                continue
            file_cells[i] = raw_file
            if not file_names:
                # Nothing uploaded yet (the sheet came first): keep the name
                # the sheet gives, so the row re-links once that file is
                # stored. It is not a verified link until then.
                links[i], link_method[i] = _base(raw_file), "column_unverified"
                continue
            key = file_key(raw_file)
            if _base(raw_file) in uploaded:
                name = _base(raw_file)
            elif key in shared_keys:
                # Two uploads share this key and neither is named exactly.
                ambiguous_cells.append(row_no)
                continue
            else:
                name = by_key.get(key)
            if name:
                links[i], link_method[i] = name, "column"
    if ambiguous_cells:
        warn(
            "ambiguous_file",
            f"{len(ambiguous_cells)} row(s) name a file that matches two or more uploads that differ only in case, "
            "spacing or punctuation (rows " + ", ".join(str(r) for r in ambiguous_cells[:10]) + "). They are matched "
            "by title after extraction.",
        )
    if by_key and file_cells:
        free = [n for n in file_names if n not in set(links.values())]
        loose = link_files_fuzzy(
            {i: v for i, v in file_cells.items() if i not in links and kept[i][0] not in ambiguous_cells}, free
        )
        for i, name in loose.items():
            links[i], link_method[i] = name, "column_fuzzy"
    num_col = None
    suggested: dict[int, str] = {}
    # A number two kept rows share links neither row, even when the file
    # column already linked one of them.
    open_rows = [
        i for i in range(len(kept))
        if i not in links and one("poster_number", kept[i][1]).lstrip("0") not in repeated_numbers
    ]
    free = [n for n in file_names if n not in set(links.values())]
    if open_rows and free:
        num_col, num_links, num_suggested = link_files_by_number(
            Table(table.sheet_name, table.header_row, table.headers, [(kept[i][0], kept[i][1]) for i in open_rows]),
            columns,
            free,
        )
        for k, name in num_links.items():
            links[open_rows[k]], link_method[open_rows[k]] = name, "number"
        suggested = {open_rows[k]: name for k, name in num_suggested.items()}

    people, people_by_file = _author_table_index(author_table) if author_table else ({}, {})
    titles = strip_position_prefixes(
        [one("title", cells) for _, cells, _ in kept], [one("poster_number", cells) for _, cells, _ in kept]
    )

    sheets: dict[str, list[dict]] = {SHEET_ABOUT: [], SHEET_CREATORS: [], SHEET_ADDITIONAL: [], SHEET_RELATED: []}
    unmatched_rows: list[dict] = []
    match_rows: list[dict] = []
    pairs: list[dict] = []
    hints: dict[str, dict] = {}
    used: dict[str, int] = {}

    for i, ((row_no, cells, form), title) in enumerate(zip(kept, titles)):
        row_key = f"row-{row_no}"
        file_name = links.get(i, "")
        if file_col is not None and not file_name:
            unmatched_rows.append({"row": row_no, "value": file_cells.get(i, ""), "title": title})
        if file_name and file_name in used:
            warn("duplicate_file", f'Rows {used[file_name]} and {row_no} both point to "{file_name}"; using row {used[file_name]}.')
            continue
        if file_name:
            used[file_name] = row_no
        link = {"row_key": row_key, "file_name": file_name}

        number = one("poster_number", cells).lstrip("0")
        join_number = "" if number in repeated_numbers else number
        table_people = people.get(join_number, []) if join_number else []
        if not table_people and file_name:
            # Authors table keyed by what the rows link files with ("7" for 7.pdf).
            # A number two rows share names neither, so it is not looked up.
            key = file_key(file_name)
            stem_number = key.lstrip("0") or "0"
            table_people = people_by_file.get(key) or (
                people.get(stem_number, []) if stem_number not in repeated_numbers else []
            )
        names = [f'{p["given"]} {p["family"]}'.strip() for p in table_people] or [
            a["name"] for a in parse_authors(one("authors", cells))
        ]
        if file_name:
            # Checked against the extracted title after extraction, however it paired.
            pairs.append({"row_key": row_key, "row": row_no, "file_name": file_name, "method": link_method[i],
                          "title": title, "authors": names})
        else:
            entry = {"row_key": row_key, "row": row_no, "title": title, "authors": names, "number": number}
            if i in suggested and suggested[i] not in used:
                entry["suggested_file"] = suggested[i]
            match_rows.append(entry)

        keywords = []
        for c in by_field.get("keywords", []):
            keywords += split_keywords(clean(cells[c.index]))
        for c in by_field.get("track", []):
            if clean(cells[c.index]):
                keywords.append(clean(cells[c.index]))
        # EasyChair topics (submission_topic.csv) are the conference's own
        # categories, so they are keywords too.
        topic_key = join_number or (file_key(file_name).lstrip("0") if file_name else "")
        if topic_key in repeated_numbers:
            topic_key = ""
        keywords += topics_by_number.get(topic_key, [])
        seen: set[str] = set()
        keywords = [k for k in keywords if not (k.lower() in seen or seen.add(k.lower()))]

        about = dict(link)
        if title:
            about["title"] = title
        if keywords:
            about["keywords"] = "; ".join(keywords)
        sheets[SHEET_ABOUT].append(about)

        extra = dict(link)
        for fld, key in (("abstract", "submission_abstract"), ("language", "language"), ("domain", "domain")):
            if one(fld, cells):
                extra[key] = one(fld, cells)
        if len(extra) > 2:
            sheets[SHEET_ADDITIONAL].append(extra)

        if table_people:
            sheets[SHEET_CREATORS] += _creator_rows_from_table(link, table_people)
        elif one("authors", cells):
            sheets[SHEET_CREATORS] += _creator_rows(link, one("authors", cells), one("affiliations", cells), warn)
        elif one("affiliations", cells):
            warn("affiliations_without_authors", f"{file_name or row_key}: affiliations given but no authors; extraction will fill authors.")

        if one("website", cells):
            sheets[SHEET_RELATED].append(
                {**link, "related_identifier": as_url(one("website", cells)),
                 "related_identifier_type": "URL", "relation_type": "References", "resource_type_general": "Other"}
            )
        # A DOI column is the poster's own DOI (already registered), so the
        # poster keeps it instead of getting a new one. The grid has no DOI
        # column, so it travels as a per-row hint for the app to apply.
        doi = find_doi(one("doi", cells)) if one("doi", cells) else ""
        if doi:
            hints.setdefault(row_key, {})["doi"] = doi
        if form.get("derived_from_doi"):
            sheets[SHEET_RELATED].append(
                {**link, "related_identifier": form["derived_from_doi"], "related_identifier_type": "DOI",
                 "relation_type": "IsDerivedFrom"}
            )
        if form.get("source_url"):
            sheets[SHEET_RELATED].append(
                {**link, "related_identifier": form["source_url"], "related_identifier_type": "URL",
                 "relation_type": "IsSupplementedBy", "resource_type_general": "Other"}
            )
        if form.get("project_license"):
            hints.setdefault(row_key, {})["project_license"] = form["project_license"]

    unverified = [kept[i][0] for i, m in link_method.items() if m == "column_unverified"]
    if unverified:
        warn(
            "files_not_uploaded",
            f"No poster files were uploaded yet, so {len(unverified)} row(s) keep the file name the sheet gives. Each "
            "re-links once a file with that name is uploaded, and its title is checked after extraction.",
        )
    fuzzy_rows = [kept[i][0] for i, m in link_method.items() if m == "column_fuzzy"]
    if fuzzy_rows:
        warn(
            "files_linked_fuzzy",
            f"{len(fuzzy_rows)} row(s) name a file that is not uploaded but is close to one that is (rows "
            + ", ".join(str(r) for r in fuzzy_rows[:10])
            + "); linked to it for you to check.",
        )
    if num_col is not None:
        n_num = sum(1 for m in link_method.values() if m == "number")
        if n_num:
            warn(
                "files_linked_by_number",
                f'Linked {n_num} row(s) to poster files by the number in column "{num_col.header}" found in the file names.',
                "info",
            )
    n_suggested = sum(1 for r in match_rows if r.get("suggested_file"))
    if n_suggested:
        warn(
            "files_suggested_by_number",
            f"{n_suggested} file name(s) carry a row's number but do not start with it (for example a download copy "
            "named \"poster (2).pdf\"). They are not linked yet; after extraction each is checked by its title, and you "
            "confirm it.",
        )
    if file_col is None and num_col is None and kept:
        warn(
            "file_column_missing",
            "No column holds the poster file names. After extraction, each row is matched to a poster by its title, "
            "with the authors as a tiebreaker, and you check the matches. A file name column (for example 12_smith.pdf) "
            "links rows exactly.",
        )
    elif match_rows:
        warn(
            "rows_without_file",
            f"{len(match_rows)} row(s) are not linked to a poster file yet; they will be matched by title after extraction.",
        )
    unmatched_files = sorted(n for n in file_names if n not in used)
    if used and unmatched_files:
        warn("files_without_row", f"{len(unmatched_files)} uploaded poster(s) have no linked row yet.", "info")
    n_license = sum(1 for h in hints.values() if "project_license" in h)
    n_doi = sum(1 for h in hints.values() if "doi" in h)
    if n_doi:
        warn(
            "existing_doi",
            f"{n_doi} row(s) give a DOI the poster already has. Those posters keep that DOI instead of getting a new one.",
            "info",
        )
    if n_license:
        warn(
            "project_license",
            f"{n_license} row(s) give a license in the form fields. That is the license of the work the poster presents, "
            "not of the poster, so it is not applied.",
            "info",
        )
    if author_table is not None:
        warn("authors_table_joined", f'Authors read from the "{author_table.sheet_name}" table, one row per author.', "info")

    dropped = sorted({c.header for c in columns if c.field in NOT_STORED})
    if dropped:
        warn("columns_not_stored", "Read but not stored (no matching poster field): " + ", ".join(dropped) + ".", "info")
    unknown = [c.header for c in columns if c.field is None and c.header and c.samples]
    if unknown:
        warn("columns_unmapped", "Not matched to a field: " + ", ".join(unknown) + ".")

    known = tpl is not None and not any(c.source in ("llm", "fuzzy") for c in columns if c.field)
    link_column = file_col or num_col
    # A file link found from values (not a column named for files, not EasyChair
    # numbering) is a good guess, so the organizer checks it.
    number_checked = num_col is not None and not (tpl or {}).get("file_number")
    fuzzy_checked = bool(fuzzy_rows)
    values_checked = file_col is not None and file_col.source == "values"
    return {
        "template": {
            "id": tpl["id"] if tpl else None,
            "name": tpl["name"] if tpl else None,
            "known": known,
            "fingerprint": header_fingerprint(table.headers),
        },
        "sheet_name": table.sheet_name,
        "header_row": table.header_row,
        "authors_table": author_table.sheet_name if author_table else None,
        "mapping": [
            {"column": c.header, "field": c.field, "source": c.source, "stored": bool(c.field) and c.field not in NOT_STORED, "samples": c.samples}
            for c in columns
            if c.header
        ],
        "file_column": (
            {
                "column": link_column.header,
                "method": "column" if file_col else "number",
                "methods": {m: sum(1 for v in link_method.values() if v == m) for m in sorted(set(link_method.values()))},
                "matched_rows": len(used),
                "rows": len(kept),
            }
            if link_column
            else None
        ),
        "sheets": {k: v for k, v in sheets.items() if v},
        "unmatched_rows": unmatched_rows,
        "unmatched_files": unmatched_files,
        "match_rows": match_rows,
        "pairs": pairs,
        "hints": hints,
        "skipped": skipped,
        "warnings": warnings,
        "needs_confirmation": (not known)
        or bool(unmatched_rows)
        or bool(match_rows)
        or number_checked
        or fuzzy_checked
        or values_checked
        or bool(unverified)
        or any(w["severity"] == "error" for w in warnings),
        "ok": bool(used) or bool(match_rows),
        "fields": FIELDS,
    }
