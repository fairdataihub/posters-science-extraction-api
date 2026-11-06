# Poster sheet mapping

Organizers often already have a spreadsheet of their posters (Curious, EasyChair, their own). `POST /sheets/map` converts that sheet into the bulk metadata grid format the posters.science app already saves, so the rest of the pipeline only ever sees our format.

It touches no database, no storage and no GPU model, so it can run while extraction is busy.

## Request

`POST /sheets/map`, `multipart/form-data`:

| Field | Required | What it is |
|-------|----------|------------|
| `sheet` | yes | The spreadsheet: `.xlsx`, `.xls`, `.csv` or `.tsv`, up to 10 MB |
| `file_names` | yes in practice | JSON array of the poster file names uploaded to the batch |
| `mapping` | no | JSON object `{column header: field or null}`, the user's confirmed choices from the confirm screen |

## Response

```json
{
  "template": {"id": "curious-2027", "name": "...", "known": true, "fingerprint": "a09719cad5552101"},
  "sheet_name": "Posters",
  "header_row": 1,
  "mapping": [{"column": "Title", "field": "title", "source": "template", "stored": true, "samples": ["..."]}],
  "file_column": {"column": "File name", "matched_rows": 48, "rows": 50},
  "sheets": {
    "About the Poster": [{"file_name": "12_smith.pdf", "title": "...", "keywords": "Human Health"}],
    "Creators": [{"file_name": "12_smith.pdf", "given_name": "Jane", "family_name": "Do", "name_type": "Personal", "contributor_affiliation_1_name": "..."}],
    "Additional information": [{"file_name": "12_smith.pdf", "submission_abstract": "..."}],
    "Related resources": [{"file_name": "12_smith.pdf", "related_identifier": "https://...", "related_identifier_type": "URL", "relation_type": "References"}]
  },
  "unmatched_rows": [{"row": 14, "value": "13_lee.pdf", "title": "..."}],
  "unmatched_files": ["40_park.pdf"],
  "warnings": [{"code": "rows_without_file", "message": "...", "severity": "warning"}],
  "needs_confirmation": true,
  "ok": true,
  "fields": {"title": "poster title", "...": "..."}
}
```

`sheets` uses the grid's sheet names and fillKeys. Rows carry `file_name`; the app swaps it for the poster id before saving.

Errors: `400` for a missing or wrong file type, `413` over 10 MB, `422` when the file can't be read or has no header row.

## How columns are mapped

1. **Known template.** If every header of a saved template is present (extra columns are fine, so an added file name column still matches), its mapping is used as is. Templates live in `TEMPLATES` in `sheet_mapper.py`. Add one per confirmed organizer format.
2. **Header aliases.** Normalized headers are looked up in `ALIASES`, then fuzzy matched (difflib, cutoff 0.86).
3. **Small LLM, optional.** Only for columns still unmapped, and only when `SHEET_MAPPER_LLM_URL` is set. The model sees each header with three sample values and answers through a JSON schema that only allows our field names or null. Anything else is dropped. A failed call is a warning, never an error.
4. **User choices** sent back in `mapping` win over all of the above.

`template.known` is true only when a template matched and nothing came from fuzzy matching or the LLM. `needs_confirmation` tells the app to show the confirm screen: unknown template, rows without a file, or an error.

## The file column is required

Each row has to say which uploaded poster it describes. The file column is found by its values, not its header: the column whose cells match the most uploaded file names (case, spacing, folders and extension ignored), at least half of them. Rows that don't match a file are listed in `unmatched_rows` and skipped. Rows are never matched by title.

If no column matches, the response has a `file_column_missing` error and no rows.

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
python -m pytest tests/test_sheet_mapper.py
```
