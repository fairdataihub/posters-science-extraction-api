# Poster sheet mapping

Organizers often already have a spreadsheet of their posters (Curious, EasyChair, their own). `POST /sheets/map` converts that sheet into the bulk metadata grid format the posters.science app already saves, so the rest of the pipeline only ever sees our format.

It touches no database, no storage and no GPU model, so it can run while extraction is busy.

## The sheet wins

Anything the organizer's sheet gives is the source of truth and overrides what we extract, with no merging. Extraction only fills fields the sheet does not have, or leaves empty. The only changes made to sheet values are cleaning: whitespace, placeholders such as `no` or `n/a`, the poster's index in front of the title (only the row's own number when the sheet has one; never a year or a number range like 2019-2023), affiliation numbers on author names, and splitting one cell into list items.

In the grid (posters.science app):

- Title, abstract, keywords, language: a filled sheet cell replaces the extracted value.
- Creators: when the sheet gives the author list for a poster, it replaces the extracted list. An empty cell on a sheet person (ORCID, affiliation, ROR) is filled from extraction when exactly one extracted person has the same family name and first initial.
- Related resources: when the sheet gives links for a poster, they replace the extracted links.
- A poster no sheet row reaches keeps its extracted metadata only.

## Request

`POST /sheets/map`, `multipart/form-data`:

| Field | Required | What it is |
|-------|----------|------------|
| `sheet` | yes | The spreadsheet: `.xlsx`, `.xls`, `.csv` or `.tsv`, up to 10 MB. An `.xlsx` must also unpack to 100 MB or less, and a sheet may have up to 20,000 rows. Send it more than once for an export split over files (EasyChair `submission.csv` plus `author.csv`, or a posters list plus an authors list) |
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
  "file_column": {"column": "File name", "method": "column", "methods": {"column": 46, "column_fuzzy": 1, "number": 1}, "matched_rows": 48, "rows": 50},
  "sheets": {
    "About the Poster": [{"row_key": "row-13", "file_name": "12_smith.pdf", "title": "...", "keywords": "Human Health"}],
    "Creators": [{"row_key": "row-13", "file_name": "12_smith.pdf", "given_name": "Jane", "family_name": "Do", "name_type": "Personal", "contributor_affiliation_1_name": "..."}],
    "Additional information": [{"row_key": "row-13", "file_name": "12_smith.pdf", "submission_abstract": "..."}],
    "Related resources": [{"row_key": "row-13", "file_name": "12_smith.pdf", "related_identifier": "https://...", "related_identifier_type": "URL", "relation_type": "References"}]
  },
  "unmatched_rows": [{"row": 14, "value": "13_lee.pdf", "title": "..."}],
  "unmatched_files": ["40_park.pdf"],
  "match_rows": [{"row_key": "row-14", "row": 14, "title": "...", "authors": ["Ann Lee", "Bo Chen"], "number": "13", "suggested_file": "Lee_poster (13).pdf"}],
  "pairs": [{"row_key": "row-13", "row": 13, "file_name": "12_smith.pdf", "method": "column", "title": "...", "authors": ["Jane Do"]}],
  "hints": {"row-20": {"project_license": "MIT", "doi": "10.5281/zenodo.123"}},
  "skipped": {"deleted": 0, "not_accepted": 12, "talk_only": 3},
  "warnings": [{"code": "rows_without_file", "message": "...", "severity": "warning"}],
  "needs_confirmation": true,
  "ok": true,
  "fields": {"title": "poster title", "...": "..."}
}
```

`sheets` uses the grid's sheet names and fillKeys. Every row carries `row_key` (its row in the file) and `file_name`. Rows linked to a file have its name; the app swaps it for the poster id before saving. Rows not linked yet have an empty `file_name` and are listed in `match_rows`, to be matched by title once posters extract (see below). Rows linked to a file are listed in `pairs`, so their titles can be checked against the extracted ones (see below).

`hints` holds per row values the grid has no column for: `doi`, a DOI the poster already has (from a DOI column), which the poster keeps instead of getting a new one; and `project_license` (EasyChair form fields).

Errors: `400` for a missing or wrong file type, `413` over 10 MB, `422` when the file can't be read, has no header row, unpacks past 100 MB, or has more than 20,000 rows.

## How columns are mapped

1. **Known template.** If every header of a saved template is present (extra columns are fine, so an added file name column still matches), its mapping is used as is. Templates live in `TEMPLATES` in `sheet_mapper.py`. Add one per confirmed organizer format.
2. **Header aliases.** Normalized headers are looked up in `ALIASES`, then fuzzy matched (difflib, cutoff 0.86).
3. **Small LLM, optional.** Only for columns still unmapped, and only when `SHEET_MAPPER_LLM_URL` is set. The model sees each header with three sample values and answers through a JSON schema that only allows our field names or null. Anything else is dropped. A failed call is a warning, never an error. Point it only at a model we host: the sample values of an unfamiliar column can be emails or private answers, so they must not leave our servers.
4. **User choices** sent back in `mapping` win over all of the above.

`template.known` is true only when a template matched and nothing came from fuzzy matching or the LLM. `needs_confirmation` tells the app to show the confirm screen: unknown template, rows without a file, a file link that was inferred (from values, from a close file name, or from numbers outside EasyChair), or an error.

Known templates: Curious (`curious-2027`) and EasyChair (`easychair`), the formats in the Poster.json Field Coverage workbook. Other organizations bring their own schemas; each one becomes a new entry in `TEMPLATES` plus a tab in the workbook.

## Linking rows to poster files

Each tier only sees the rows and files the tiers before it left unlinked. The goal is a row for every poster uploaded; a poster no tier can place keeps its extracted metadata only, and a row no tier can place is not used.

1. **File name column** (`column`). Found by its values, not its header: the column whose cells match the most uploaded file names by base name (folders, extension, case and spacing ignored), at least half of them. A column headed like a file column whose values look like file names is used even with fewer exact hits. This is the exact link and the one to ask organizers for.
2. **Close file name** (`column_fuzzy`). A file cell that names no uploaded file exactly links to the one unlinked file whose base name is at least 0.85 alike (difflib) and 0.05 ahead of the next (`smith_poster_v2.pdf` for `smith_poster_v3.pdf`). Flagged for the organizer to check.
3. **Submission number in the file name** (`number`). A numeric number column (EasyChair `#`, a poster number) links rows to files named by it, only when the name says so plainly: it starts with the number (`17.pdf`, `17_smith.pdf`, `017-x.pdf`) or has it after "submission" (`bosc2026_submission_17.pdf`). A name that starts with a year (`2026_smith.pdf`) does not count. A file that could belong to two rows is left alone. For EasyChair this is the documented naming, so it needs no check; for other sheets the organizer confirms it.
   A number found anywhere else in a name (`Smith_poster (2).pdf` is a browser's download copy, `ISMB20262153Patel.pdf` reads as 2153) is not linked. The row goes to `match_rows` with that file as `suggested_file`, and a `files_suggested_by_number` warning.
4. **Title, after extraction.** Rows still without a file go to `match_rows`, with a `file_column_missing` (or `rows_without_file`) warning. Each time the grid loads, the app sends those rows and the extracted titles and authors of the posters still without a row to `POST /sheets/match`.

`file_column.methods` counts the rows each tier linked.

Every linked row, however it was linked, is listed in `pairs` and gets a title check after extraction, so a wrong file is caught before its metadata goes out (a mistyped name in the organizer's own file column included).

## POST /sheets/match

JSON body: `rows` (the `match_rows` from `/sheets/map`), `posters` (`[{file_name, title, authors: [names]}]` as extracted) and, optionally, `pairs` (the `pairs` from `/sheets/map`). Up to 2000 of each.

**Title check on paired rows.** Each pair's sheet title is compared with its poster's extracted title using the bag of words score of tier 3 (spelling slips, a title cut short and an added subtitle allowed). The pair agrees at 0.50 or more, unless both sides list authors and share no family name. Each pair comes back in `pair_checks` with `status` `agree`, `disagree` (with `reason` `title` or `authors`) or `unchecked` (`not_extracted` or `no_title`); disagreements are also in `flagged`, for the organizer to check. The 0.50 was set on real titles from 53 conferences: a title against a noisy copy of itself (cut short, OCR slips, a subtitle added, words swapped) and against other posters from the same conference. At 0.50, 10 of 1,293 true pairs (0.8 percent) are flagged for a look and 1 of 1,291 different posters (0.1 percent) passes, a French and an English title of one project. Paired files are out of play for the loop below.

**Suggested files.** Before the loop, a row whose `suggested_file` has a title that agrees (the same check) takes that file, as method `number_suggested` with confidence `review`.

A loop of tiers, strictest first:

| Tier | `method` | Rule | `confidence` |
|------|----------|------|--------------|
| 1 | `title` | ROUGE-L F1 of the two titles (words, case and accents ignored) at least 0.95, and no other row or poster within 0.10 | `high`, applied as is |
| 2 | `title_authors` | Title at least 0.60 and author family names at least half in common (spelling and OCR slips allowed); the combined score (0.75 title, 0.25 authors) at least 0.10 ahead | `review` |
| 3 | `words_fuzzy` | Bag of words: word order ignored, each word may differ by a slip (spelling ratio 0.80 or more), and a title cut short or with a subtitle counts by containment. At least 0.70, or 0.55 when the authors agree, and 0.10 ahead | `review` |
| 4 | `best_remaining` | Among what is left, the row and the poster are each other's best candidate, the combined score is at least 0.50, and 0.05 ahead | `review` |

Each match takes its row and poster out of play, which can make another pair decisive, so after any match the loop starts again at tier 1 (`round` on each match). It stops when every poster has a row, or when no tier can place anything more. Matching is one to one. Two candidates that score the same are never guessed at any tier, so two identical titles with no authors to tell them apart stay unmatched. Rows left over are listed in `unmatched_rows` with the closest poster as a suggestion, and the files in `unmatched_files`.

On 2,000 synthetic titles with OCR slips in 15 percent of words, shuffled word order in a fifth and cut titles in a fifth, every poster was placed and none wrongly, in under a second.

```json
{
  "matches": [{"row_key": "row-14", "file_name": "lee.pdf", "method": "title", "confidence": "high", "round": 1,
               "title_score": 0.981, "words_score": 0.981, "author_score": 1.0, "score": 0.981,
               "row_title": "...", "poster_title": "..."}],
  "unmatched_rows": [{"row_key": "row-20", "title": "...", "closest": {"file_name": "x.pdf", "title_score": 0.31, "words_score": 0.4, "author_score": null, "score": 0.4}}],
  "unmatched_files": ["x.pdf"],
  "pair_checks": [{"row_key": "row-13", "file_name": "12_smith.pdf", "method": "column", "status": "disagree", "reason": "title",
                   "title_score": 0.12, "words_score": 0.2, "author_score": null, "row_title": "...", "poster_title": "..."}],
  "flagged": [{"row_key": "row-13", "...": "..."}],
  "rounds": 3,
  "thresholds": {"pair_agree": 0.5, "title_auto": 0.95, "title_min": 0.6, "authors_min": 0.5, "words_min": 0.7, "words_with_authors": 0.55, "floor": 0.5, "margin": 0.1, "final_margin": 0.05}
}
```

## EasyChair

Built from the column headings of a real BOSC 2026 basic submission download and the BOSC submission form.

- Accepts the basic submission download (one sheet), the Conference data download (a workbook with Submissions and Authors sheets) and the same as CSV files (`submission.csv` plus `author.csv`). Headers are matched case-insensitively.
- With an Authors table, authors come from it in row order: first and last name, affiliation, and an ORCID when Web page is an orcid.org link. Email, country and the corresponding flag are not stored. Without it, the one-string Authors column (`A, B and C`) is split.
- Any sheet can bring an authors table the same way, not only EasyChair: a sheet or file with no title column, a column naming the poster (a submission, paper or poster number or id, or a file name) and a person's name (first and last name, or one name column). Affiliation or institution, and an ORCID column (a bare iD or an orcid.org link), are read when present.
- Rows are skipped when Deleted is set, when Decision does not say accept (only once at least one row is accepted), or when the Submission Type form field is TALK ONLY.
- Keywords are one per line (a keyword may contain commas). Track name goes into keywords.
- Form fields (`(name) value` lines): a previously published abstract's DOI becomes a related resource (IsDerivedFrom), and a source code link a related resource (URL, IsSupplementedBy). The license question is the license of the work the poster presents, so it comes back as a hint in `hints`, never as the poster's license. All other questions, including fees, diversity, income level, conflicts of interest and comments for the chairs, are dropped as the file is read. The confirm screen shows the question names only, never the answers.
- Poster files: EasyChair's Download submissions zip names files by submission number, which tier 3 links. At ISMB and BOSC the poster PDFs come from ISMB's poster platform instead, whose file names carry ISMB's own number (`ISMB20262153PatelPoster.pdf`), not documented as the EasyChair one. Those numbers are only suggestions, so the rows are placed by title.

## Value parsing (deterministic, no LLM)

- Placeholders such as `no`, `n/a`, `-` count as empty.
- Titles lose the poster's index in front (`12 - Title` (Curious uses a long dash), `P12: Title`, `#12 | Title`, `Poster 12. Title`) when the index is the row's own poster number, or when most titles in the column have one. `3D printing` or `2024: a review` keep their numbers.
- Authors in one cell are split on commas, semicolons, `and` and new lines. Trailing digits or superscripts are affiliation numbers (`John Do1`, `Ann Lee1,2`, `Bo Chen` with superscript 3). Run-together marks like `12` read as 1 and 2 when 12 is not in the affiliation list.
- Affiliations as a numbered list (`1 Dept A` on each line, or `1. A; 2. B`) are attached by number. A single unnumbered affiliation goes to every author. Several unnumbered ones go to every author with a warning.
- Names split as given and family name, keeping particles (`van der Berg`) with the family name. `Family, Given` is also read.
- Track goes into keywords. Website becomes a related resource (URL, References). An existing DOI is the poster's own DOI and comes back in `hints` (the poster keeps it, no new DOI).
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
