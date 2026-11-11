"""
Map an organizer's poster spreadsheet (Curious, EasyChair, our own, ...) onto
the posters.science bulk metadata grid.

The pipeline only ever works on our own format. Everything else is converted
here, before extraction starts:

1. Read the sheet (.xlsx, .xls, .csv, .tsv) and find the header row.
2. Map each column to one of our fields, in three tiers:
   a. known template: the header set matches a saved template (exact, no guessing)
   b. header aliases: normalized and fuzzy matching against a synonym list
   c. small LLM, only for columns still unmapped, and only when
      SHEET_MAPPER_LLM_URL is set. Output is constrained to our field names.
3. Find the file column by its values: the column whose cells match the
   uploaded poster file names. It is required; rows are never matched by title.
4. Convert values with deterministic parsers (authors with superscript
   affiliation numbers, numbered affiliation lists, position prefixes on
   titles, "no" placeholders).

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
    {
        "id": "posters-science-bulk",
        "name": "Posters.science bulk submission template (Posters sheet)",
        "headers": {
            "posterId": "poster_number",
            "fileName": "file_name",
            "title": "title",
            "abstract": "abstract",
            "keywords": "keywords",
            "researchField": "domain",
            "language": "language",
            "existingDoi": "doi",
        },
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
        rows = [[_cell_text(c) for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)]
        return [(os.path.splitext(os.path.basename(filename))[0], rows)]
    if ext in (".xlsx", ".xlsm"):
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        return [
            (ws.title, [[_cell_text(c) for c in r] for r in ws.iter_rows(values_only=True)])
            for ws in wb.worksheets
        ]
    if ext == ".xls":
        import xlrd

        book = xlrd.open_workbook(file_contents=data)
        return [
            (s.name, [[_cell_text(s.cell_value(i, j)) for j in range(s.ncols)] for i in range(s.nrows)])
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


# A per-author table (EasyChair Conference data download "Authors" sheet, or
# its author.csv): one row per author per submission.
AUTHOR_TABLE_NEEDS = {"submission number", "first name", "last name"}


def is_author_table(headers: list[str]) -> bool:
    return AUTHOR_TABLE_NEEDS <= {_norm_header(h) for h in headers}


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
    if best is None:
        return None, by_key
    chosen = best[1]
    for c in columns:
        if c is not chosen and c.field == "file_name":
            c.field, c.source = None, "none"
    chosen.field, chosen.source = "file_name", ("values" if chosen.source not in ("template", "user") else chosen.source)
    return chosen, by_key


def file_numbers(name: str) -> list[str]:
    """
    Submission numbers a poster file name may carry, best first:
    "bosc_submission_17.pdf" -> 17, "17_smith.pdf" -> 17, and a number run
    together with a year ("ISMB20262153Patel.pdf") -> 2153 as well.
    """
    stem = os.path.splitext(re.split(r"[\\/]", name.strip())[-1])[0].lower()
    m = re.search(r"submission[\s_-]*0*(\d+)", stem)
    if m:
        return [m.group(1)]
    out: list[str] = []
    for run in re.findall(r"\d+", stem):
        out.append(run.lstrip("0") or "0")
        if len(run) > 4 and run[:2] in ("19", "20"):
            out.append(run[4:].lstrip("0") or "0")
    return list(dict.fromkeys(out))


def link_files_by_number(table: Table, columns: list[ColumnMap], file_names: list[str]) -> tuple[ColumnMap | None, dict[int, str]]:
    """
    Link rows to files through the row's submission or poster number, when the
    files are named by it. A file that could belong to two rows is left alone
    (it falls through to title matching). Used only when no column holds the
    file names.
    """
    if not file_names:
        return None, {}
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
        for name in file_names:
            hits = {ids[n] for n in file_numbers(name) if n in ids}
            if len(hits) == 1:
                row = hits.pop()
                if row not in links:
                    links[row] = name
        if links and len(links) / len(file_names) >= 0.5:
            return c, links
    return None, {}


# ---------------------------------------------------------------------------
# Value parsers
# ---------------------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")
_POSITION_PREFIX = re.compile(r"^\s*\d{1,5}\s*[\u2010-\u2015\-:.)]\s*")


def clean(value: str) -> str:
    value = re.sub(r"[ \t]+", " ", (value or "").strip())
    return "" if value.lower() in PLACEHOLDERS else value


def strip_position_prefixes(titles: list[str]) -> list[str]:
    """Drop "12 - " style prefixes, but only when most titles in the column have one."""
    filled = [t for t in titles if t]
    if not filled or sum(1 for t in filled if _POSITION_PREFIX.match(t)) / len(filled) < 0.8:
        return titles
    return [_POSITION_PREFIX.sub("", t, count=1) if t else t for t in titles]


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
        return given, family
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
    """An ORCID iD from a web page cell, as https://orcid.org/... ; else empty."""
    m = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])", value or "")
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


def _author_table_index(authors: Table) -> dict[str, list[dict]]:
    """Submission number -> its authors in row order (EasyChair Authors sheet)."""
    idx = {_norm_header(h): i for i, h in enumerate(authors.headers)}

    def get(cells: list[str], key: str) -> str:
        return clean(cells[idx[key]]) if key in idx else ""

    out: dict[str, list[dict]] = {}
    for _, cells in authors.rows:
        sub = get(cells, "submission number").lstrip("0")
        if not sub:
            continue
        out.setdefault(sub, []).append(
            {
                "given": get(cells, "first name"),
                "family": get(cells, "last name"),
                "affiliation": get(cells, "affiliation"),
                "orcid": orcid_url(cells[idx["web page"]]) if "web page" in idx else "",
            }
        )
    return out


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

    # No file column: try submission numbers in the file names.
    num_col, num_links = (None, {})
    if file_col is None:
        num_col, num_links = link_files_by_number(
            Table(table.sheet_name, table.header_row, table.headers, [(n, c) for n, c, _ in kept]), columns, file_names
        )

    people = _author_table_index(author_table) if author_table else {}
    titles = strip_position_prefixes([one("title", cells) for _, cells, _ in kept])

    sheets: dict[str, list[dict]] = {SHEET_ABOUT: [], SHEET_CREATORS: [], SHEET_ADDITIONAL: [], SHEET_RELATED: []}
    unmatched_rows: list[dict] = []
    match_rows: list[dict] = []
    hints: dict[str, dict] = {}
    used: dict[str, int] = {}

    for i, ((row_no, cells, form), title) in enumerate(zip(kept, titles)):
        row_key = f"row-{row_no}"
        file_name = ""
        if file_col is not None:
            raw_file = cells[file_col.index].strip()
            if raw_file:
                file_name = (by_key.get(file_key(raw_file)) if by_key else re.split(r"[\\/]", raw_file)[-1]) or ""
                if not file_name:
                    unmatched_rows.append({"row": row_no, "value": raw_file, "title": title})
            else:
                unmatched_rows.append({"row": row_no, "value": "", "title": title})
        else:
            file_name = num_links.get(i, "")
        if file_name and file_name in used:
            warn("duplicate_file", f'Rows {used[file_name]} and {row_no} both point to "{file_name}"; using row {used[file_name]}.')
            continue
        if file_name:
            used[file_name] = row_no
        link = {"row_key": row_key, "file_name": file_name}

        number = one("poster_number", cells).lstrip("0")
        table_people = people.get(number, []) if number else []
        if not file_name:
            names = [f'{p["given"]} {p["family"]}'.strip() for p in table_people] or [
                a["name"] for a in parse_authors(one("authors", cells))
            ]
            match_rows.append({"row_key": row_key, "row": row_no, "title": title, "authors": names, "number": number})

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
        if one("doi", cells):
            doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", one("doi", cells), flags=re.I)
            sheets[SHEET_RELATED].append(
                {**link, "related_identifier": doi, "related_identifier_type": "DOI", "relation_type": "IsIdenticalTo"}
            )
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
            hints[row_key] = {"project_license": form["project_license"]}

    if num_col is not None:
        warn(
            "files_linked_by_number",
            f'Linked {len(used)} row(s) to poster files by the number in column "{num_col.header}" found in the file names.',
            "info",
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
    if hints:
        warn(
            "project_license",
            f"{len(hints)} row(s) give a license in the form fields. That is the license of the work the poster presents, "
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
        "hints": hints,
        "skipped": skipped,
        "warnings": warnings,
        "needs_confirmation": (not known)
        or bool(unmatched_rows)
        or bool(match_rows)
        or number_checked
        or values_checked
        or any(w["severity"] == "error" for w in warnings),
        "ok": bool(used) or bool(match_rows),
        "fields": FIELDS,
    }
