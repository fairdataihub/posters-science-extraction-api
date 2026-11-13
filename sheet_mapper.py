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
    },
]


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


def _check_xlsx_size(data: bytes) -> None:
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            unpacked = sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile:
        raise ValueError("This .xlsx file could not be opened. Save it again from Excel and retry.") from None
    if unpacked > MAX_XLSX_UNPACKED:
        raise ValueError(
            f"This workbook unpacks to {unpacked // (1024 * 1024)} MB, over the {MAX_XLSX_UNPACKED // (1024 * 1024)} MB "
            "limit. Remove sheets or images that are not the poster list and retry."
        )


def _capped(rows, sheet_name: str) -> list:
    out = []
    for r in rows:
        if len(out) == MAX_ROWS:
            raise ValueError(f'Sheet "{sheet_name}" has more than {MAX_ROWS:,} rows. Split it or remove rows that are not posters.')
        out.append(r)
    return out


def _read_rows(data: bytes, filename: str) -> list[tuple[str, list[list[str]]]]:
    """Return [(sheet name, rows of cell text)] for every sheet in the file."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in (".csv", ".tsv", ".txt"):
        text = data.decode("utf-8-sig", errors="replace")
        if ext == ".tsv":
            delimiter = "\t"
        else:
            try:
                delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t").delimiter
            except csv.Error:
                delimiter = ","
        name = os.path.splitext(os.path.basename(filename))[0]
        rows = _capped(([_cell_text(c) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)), name)
        return [(name, rows)]
    if ext in (".xlsx", ".xlsm"):
        import openpyxl

        _check_xlsx_size(data)
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        return [
            (ws.title, _capped(([_cell_text(c) for c in r] for r in ws.iter_rows(values_only=True)), ws.title))
            for ws in wb.worksheets
        ]
    if ext == ".xls":
        import xlrd

        book = xlrd.open_workbook(file_contents=data)
        return [
            (s.name, _capped(([_cell_text(s.cell_value(i, j)) for j in range(s.ncols)] for i in range(s.nrows)), s.name))
            for s in book.sheets()
        ]
    raise ValueError(f"Unsupported sheet type: {ext or filename}. Use .xlsx, .xls, .csv or .tsv.")


def _header_score(cells: list[str]) -> int:
    return sum(1 for c in cells if _norm_header(c) in _ALIAS_INDEX)


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
    if "title" in idx:
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


def read_tables(files: list[tuple[bytes, str]]) -> tuple[Table, Table | None]:
    """
    Pick the poster table (the sheet and header row with the most recognizable
    headers) and, when one is there, a per-author table to join to it. Several
    files are read as one workbook, so an EasyChair export can come as one
    .xlsx or as submission.csv plus author.csv.
    """
    best = None
    authors = None
    for data, filename in files:
        for sheet_name, rows in _read_rows(data, filename):
            for i, cells in enumerate(rows[:15]):
                if is_author_table(cells):
                    if authors is None:
                        authors = _table_at(sheet_name, rows, i)
                    break
                score = _header_score(cells)
                if best is None or score > best[0]:
                    best = (score, sheet_name, i, rows)
    if best is None or best[0] == 0:
        raise ValueError("Could not find a header row. The first rows should name the columns.")
    _, sheet_name, idx, rows = best
    return _table_at(sheet_name, rows, idx), authors


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
    multi = {"social", "comment", "keywords", "track", "ignored"}
    seen: dict[str, ColumnMap] = {}
    for c in columns:
        if c.field and c.field not in multi:
            if c.field in seen:
                notes.append(f'Columns "{seen[c.field].header}" and "{c.header}" both look like {c.field}; using the first.')
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


def find_file_column(table: Table, columns: list[ColumnMap], file_names: list[str]) -> tuple[ColumnMap | None, dict[str, str]]:
    """
    The file column is decided by its values, not its header: the column whose
    cells match the most uploaded file names. Without an upload list, fall back
    to a column named like a file column whose values look like file names.
    """
    by_key: dict[str, str] = {}
    for name in file_names:
        by_key.setdefault(file_key(name), name)

    best: tuple[float, ColumnMap] | None = None
    for c in columns:
        values = [cells[c.index] for _, cells in table.rows if cells[c.index].strip()]
        if not values:
            continue
        if by_key:
            hits = sum(1 for v in values if file_key(v) in by_key)
            score = hits / len(values)
        else:
            looks = sum(1 for v in values if v.lower().endswith(POSTER_EXTENSIONS))
            score = looks / len(values) if c.field == "file_name" else 0.0
        if score >= 0.5 and (best is None or score > best[0]):
            best = (score, c)
    if best is None and by_key:
        # Few exact hits (typos, renamed files): still use a column headed like
        # a file column whose values look like file names. Its cells then link
        # by close base names, for the organizer to check.
        for c in columns:
            values = [cells[c.index] for _, cells in table.rows if cells[c.index].strip()]
            if c.field == "file_name" and values:
                looks = sum(1 for v in values if v.lower().endswith(POSTER_EXTENSIONS))
                if looks / len(values) >= 0.5:
                    best = (0.0, c)
                    break
    if best is None:
        return None, by_key
    chosen = best[1]
    for c in columns:
        if c is not chosen and c.field == "file_name":
            c.field, c.source = None, "none"
    chosen.field, chosen.source = "file_name", ("values" if chosen.source not in ("template", "user") else chosen.source)
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
    carries it after "submission" ("bosc_submission_17.pdf"). Only these pair
    a row on their own.
    """
    stem = _stem(name)
    m = re.search(r"submission[\s_-]*0*(\d+)", stem)
    if m:
        return m.group(1)
    m = re.match(r"0*(\d+)(?![\d])", stem)
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
    if re.search(r"submission[\s_-]*\d", stem):
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
    title matching). Rows a file column already linked never get here.
    """
    if not file_names:
        return None, {}, {}
    for c in columns:
        if c.field != "poster_number":
            continue
        ids: dict[str, int] = {}
        values = 0
        for i, (_, cells) in enumerate(table.rows):
            v = cells[c.index].strip()
            if not v:
                continue
            values += 1
            if re.fullmatch(r"\d+", v):
                ids.setdefault(v.lstrip("0") or "0", i)
        if not values or len(ids) / values < 0.8:
            continue
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
        if (links or suggestions) and (len(links) + len(suggestions)) / len(file_names) >= 0.5:
            return c, links, suggestions
    return None, {}, {}


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")
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
            if n:
                # The row has its own number: cut only when the title starts with it.
                cut = index == n.lstrip("0")
            else:
                # No number for the row: cut when most titles carry an index,
                # but never a year ("2026: A review").
                cut = most and not re.fullmatch(r"(19|20)\d\d", m.group(1))
            if cut:
                t = t[m.end():]
        out.append(t)
    return out


def parse_authors(cell: str) -> list[dict]:
    """
    "John Do1, Ann Lee1,2, Bo Chen\u00b3" -> [{name, numbers}]. Authors are split
    on commas, semicolons, " and " and newlines; trailing digits are affiliation
    numbers. A bare number after a comma ("Lee1, 2") belongs to the author before.
    """
    text = (cell or "").translate(_SUPERSCRIPTS).replace("*", "")
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
    text = (cell or "").translate(_SUPERSCRIPTS).strip()
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


def _accepted(decision: str) -> bool:
    d = decision.lower()
    return "accept" in d and "reject" not in d


def map_sheet(
    data: bytes,
    filename: str,
    file_names: list[str],
    overrides: dict[str, str | None] | None = None,
    extra_files: list[tuple[bytes, str]] | None = None,
) -> dict:
    warnings: list[dict] = []

    def warn(code: str, message: str, severity: str = "warning"):
        warnings.append({"code": code, "message": message, "severity": severity})

    table, author_table = read_tables([(data, filename)] + list(extra_files or []))
    columns, tpl, notes = map_columns(table, overrides)
    for n in notes:
        warn("mapping_note", n)
    file_col, by_key = find_file_column(table, columns, file_names)

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
        warn("no_decisions", "No row is marked accepted yet, so every row is kept.", "info")
    kept: list[tuple[int, list[str], dict]] = []
    for n, cells in data_rows:
        if one("deleted", cells):
            skipped["deleted"] += 1
            continue
        if any_accepted and not _accepted(one("decision", cells)):
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

    # Link rows to files. Each tier only sees rows and files still unlinked.
    links: dict[int, str] = {}
    link_method: dict[int, str] = {}
    file_cells: dict[int, str] = {}
    if file_col is not None:
        for i, (_, cells, _) in enumerate(kept):
            raw_file = cells[file_col.index].strip()
            if not raw_file:
                continue
            file_cells[i] = raw_file
            name = by_key.get(file_key(raw_file)) if by_key else re.split(r"[\\/]", raw_file)[-1]
            if name:
                links[i], link_method[i] = name, "column"
    if by_key and file_cells:
        free = [n for n in file_names if n not in set(links.values())]
        loose = link_files_fuzzy({i: v for i, v in file_cells.items() if i not in links}, free)
        for i, name in loose.items():
            links[i], link_method[i] = name, "column_fuzzy"
    num_col = None
    suggested: dict[int, str] = {}
    open_rows = [i for i in range(len(kept)) if i not in links]
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
        table_people = people.get(number, []) if number else []
        if not table_people and file_name:
            # Authors table keyed by what the rows link files with ("7" for 7.pdf).
            key = file_key(file_name)
            table_people = people_by_file.get(key) or people.get(key.lstrip("0") or "0", [])
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
        keywords = list(dict.fromkeys(keywords))

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
        or any(w["severity"] == "error" for w in warnings),
        "ok": bool(used) or bool(match_rows),
        "fields": FIELDS,
    }
