# Poster sheet mapping

Organizers often already have a spreadsheet of their posters (Curious, EasyChair, their own). `POST /sheets/map` converts that sheet into the bulk metadata grid format the posters.science app already saves, so the rest of the pipeline only ever sees our format.

It touches no database, no storage and no GPU model, so it can run while extraction is busy.

## Request

`POST /sheets/map`, `multipart/form-data`:

| Field | Required | What it is |
|-------|----------|------------|
| `sheet` | yes | The spreadsheet: `.xlsx`, `.xls`, `.csv` or `.tsv`, up to 10 MB. Send it more than once for an export split over files (EasyChair `submission.csv` plus `author.csv`) |
| `file_names` | yes in practice | JSON array of the poster file names uploaded to the batch |
| `mapping` | no | JSON object `{column header: field or null}`, the user's confirmed choices from the confirm screen |

## Response

```json
{
  "template": {"id": "curious-2027", "name": "...", "known": true, "fingerprint": "a09719cad5552101"},
  "sheet_name": "Posters",
  "header_row": 1,
  "mapping": [{"column": "Title", "field": "title", "source": "template", "stored": true, "samples": ["..."]}],
  "authors_table": null,
  "file_column": {"column": "File name", "method": "column", "matched_rows": 48, "rows": 50},
  "sheets": {
    "About the Poster": [{"row_key": "row-13", "file_name": "12_smith.pdf", "title": "...", "keywords": "Human Health"}],
    "Creators": [{"row_key": "row-13", "file_name": "12_smith.pdf", "given_name": "Jane", "family_name": "Do", "name_type": "Personal", "contributor_affiliation_1_name": "..."}],
    "Additional information": [{"row_key": "row-13", "file_name": "12_smith.pdf", "submission_abstract": "..."}],
    "Related resources": [{"row_key": "row-13", "file_name": "12_smith.pdf", "related_identifier": "https://...", "related_identifier_type": "URL", "relation_type": "References"}]
  },
  "unmatched_rows": [{"row": 14, "value": "13_lee.pdf", "title": "..."}],
  "unmatched_files": ["40_park.pdf"],
  "match_rows": [{"row_key": "row-14", "row": 14, "title": "...", "authors": ["Ann Lee", "Bo Chen"], "number": "13"}],
  "hints": {"row-20": {"project_license": "MIT"}},
  "skipped": {"deleted": 0, "not_accepted": 12, "talk_only": 3},
  "warnings": [{"code": "rows_without_file", "message": "...", "severity": "warning"}],
  "needs_confirmation": true,
  "ok": true,
  "fields": {"title": "poster title", "...": "..."}
}
```

`sheets` uses the grid's sheet names and fillKeys. Every row carries `row_key` (its row in the file) and `file_name`. Rows linked to a file have its name; the app swaps it for the poster id before saving. Rows not linked yet have an empty `file_name` and are listed in `match_rows`, to be matched by title once posters extract (see below).

Errors: `400` for a missing or wrong file type, `413` over 10 MB, `422` when the file can't be read or has no header row.

## How columns are mapped

1. **Known template.** If every header of a saved template is present (extra columns are fine, so an added file name column still matches), its mapping is used as is. Templates live in `TEMPLATES` in `sheet_mapper.py`. Add one per confirmed organizer format.
2. **Header aliases.** Normalized headers are looked up in `ALIASES`, then fuzzy matched (difflib, cutoff 0.86).
3. **Small LLM, optional.** Only for columns still unmapped, and only when `SHEET_MAPPER_LLM_URL` is set. The model sees each header with three sample values and answers through a JSON schema that only allows our field names or null. Anything else is dropped. A failed call is a warning, never an error.
4. **User choices** sent back in `mapping` win over all of the above.

`template.known` is true only when a template matched and nothing came from fuzzy matching or the LLM. `needs_confirmation` tells the app to show the confirm screen: unknown template, rows without a file, a file link that was inferred (from values, or from numbers outside EasyChair), or an error.

Known templates: Curious (`curious-2027`), EasyChair (`easychair`) and our own bulk template (`posters-science-bulk`).

## Linking rows to poster files

In order:

1. **File name column.** Found by its values, not its header: the column whose cells match the most uploaded file names (case, spacing, folders and extension ignored), at least half of them. This is the exact link and the one to ask organizers for.
2. **Submission number in the file name.** With no file column, a numeric number column (EasyChair `#`, a poster number) links rows to files named by it: `17.pdf`, `17_smith.pdf`, `bosc2026_submission_17.pdf`, or a number run on from a year (`ISMB20262153Patel.pdf` also reads as 2153). A file that could belong to two rows is left alone. For EasyChair this is the documented naming, so it needs no check; for other sheets the organizer confirms it.
3. **Title, after extraction.** Rows still without a file go to `match_rows`, with a `file_column_missing` (or `rows_without_file`) warning. Once posters extract, the app sends those rows and the extracted titles and authors to `POST /sheets/match`.

## POST /sheets/match

JSON body: `rows` (the `match_rows` from `/sheets/map`) and `posters` (`[{file_name, title, authors: [names]}]` as extracted). Up to 2000 of each.

Tiers, strongest first:

1. **Title.** ROUGE-L F1 between the two titles (words, case and accents ignored). Automatic (`confidence: "high"`) when the score is at least 0.95 and no other row or poster is within 0.10 of it.
2. **Title and authors.** When the title alone isn't decisive (two posters share a title, or the poster words it a little differently, 0.60 or more), the author family names break the tie: at least half in common (spelling and OCR slips allowed), and the combined score (0.75 title, 0.25 authors) at least 0.10 ahead of the next candidate. These come back as `confidence: "review"` for the organizer to check.
3. **No match.** The row is listed in `unmatched_rows` with its closest poster as a suggestion, and the file in `unmatched_files`.

Matching is one to one, best score first. Two identical titles with no authors to tell them apart are never guessed.

```json
{
  "matches": [{"row_key": "row-14", "file_name": "lee.pdf", "method": "title", "confidence": "high",
               "title_score": 0.981, "author_score": 1.0, "score": 0.986, "row_title": "...", "poster_title": "..."}],
  "unmatched_rows": [{"row_key": "row-20", "title": "...", "closest": {"file_name": "x.pdf", "title_score": 0.71, "author_score": null, "score": 0.71}}],
  "unmatched_files": ["x.pdf"],
  "thresholds": {"title_auto": 0.95, "title_min": 0.6, "margin": 0.1, "authors_min": 0.5}
}
```

## EasyChair

Built from the column headings of a real BOSC 2026 basic submission download and the BOSC submission form.

- Accepts the basic submission download (one sheet), the Conference data download (a workbook with Submissions and Authors sheets) and the same as CSV files (`submission.csv` plus `author.csv`). Headers are matched case-insensitively.
- With an Authors table, authors come from it in row order: first and last name, affiliation, and an ORCID when Web page is an orcid.org link. Email, country and the corresponding flag are not stored. Without it, the one-string Authors column (`A, B and C`) is split.
- Rows are skipped when Deleted is set, when Decision does not say accept (only once at least one row is accepted), or when the Submission Type form field is TALK ONLY.
- Keywords are one per line (a keyword may contain commas). Track name goes into keywords.
- Form fields (`(name) value` lines): a previously published abstract's DOI becomes a related resource (IsDerivedFrom), and a source code link a related resource (URL, IsSupplementedBy). The license question is the license of the work the poster presents, so it comes back as a hint in `hints`, never as the poster's license. All other questions, including fees, diversity, income level, conflicts of interest and comments for the chairs, are dropped as the file is read. The confirm screen shows the question names only, never the answers.
- Poster files: EasyChair's Download submissions zip names files by submission number, which step 2 links. At ISMB and BOSC the poster PDFs come from ISMB's poster platform instead, which has no EasyChair number, so those rows fall through to title matching.

## Value parsing (deterministic, no LLM)

- Placeholders such as `no`, `n/a`, `-` count as empty.
- Titles lose a leading position number (`12 - Title`), but only when most titles in the column have one.
- Authors in one cell are split on commas, semicolons, `and` and new lines. Trailing digits or superscripts are affiliation numbers (`John Do1`, `Ann Lee1,2`, `Bo Chen` with superscript 3). Run-together marks like `12` read as 1 and 2 when 12 is not in the affiliation list.
- Affiliations as a numbered list (`1 Dept A` on each line, or `1. A; 2. B`) are attached by number. A single unnumbered affiliation goes to every author. Several unnumbered ones go to every author with a warning.
- Names split as given and family name, keeping particles (`van der Berg`) with the family name. `Family, Given` is also read.
- Track goes into keywords. Website becomes a related resource (URL, References). An existing DOI becomes a related resource (DOI, IsIdenticalTo).
- Contact name and email, poster number, social links and comments are read and reported as not stored: the poster record has no place for them.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SHEET_MAPPER_LLM_URL` | Base URL of an OpenAI-compatible server (llama.cpp `llama-server`, Ollama, vLLM). Unset means no LLM tier | unset |
| `SHEET_MAPPER_LLM_MODEL` | Model name sent to that server | `qwen2.5-1.5b-instruct` |

A 1 to 2B instruct model at 4-bit (about 1 GB) runs this on CPU in a few seconds, for example:

```bash
llama-server -m qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8091 -c 4096
SHEET_MAPPER_LLM_URL=http://localhost:8091 python api.py
```

## Tests

```bash
python -m pytest tests/test_sheet_mapper.py tests/test_sheet_mapper_easychair.py tests/test_poster_matcher.py
```
