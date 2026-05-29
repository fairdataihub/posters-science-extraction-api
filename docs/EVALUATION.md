# Evaluation

Validation methodology and results for poster2json.

## Metrics

The pipeline is validated using four complementary metrics:

| Metric | Description | Threshold | Rationale |
|--------|-------------|-----------|-----------|
| **Word Capture (w)** | Proportion of reference vocabulary in extracted text | ≥0.75 | Measures lexical completeness |
| **ROUGE-L (r)** | Longest common subsequence similarity | ≥0.75 | Captures sequential text preservation |
| **Number Capture (n)** | Proportion of numeric values preserved | ≥0.75 | Validates quantitative data integrity |
| **Field Proportion (f)** | Ratio of extracted to reference JSON elements | 0.50–1.50 | Accommodates layout variability |

### Pass Criteria

A poster passes validation if ALL conditions are met:
- Word Capture ≥ 0.75
- ROUGE-L ≥ 0.75
- Number Capture ≥ 0.75
- Field Proportion between 0.50 and 1.50

## Metric Implementation

### Word Capture

Measures vocabulary overlap between extracted and reference text:

```python
word_capture = len(extracted_words & reference_words) / len(reference_words)
```

- Tokenized to individual words
- Case-insensitive comparison
- Excludes common stopwords

### ROUGE-L (Section-Aware)

Uses longest common subsequence with section-aware matching:

```python
global_score = rouge_l(all_extracted_text, all_reference_text)
section_scores = [rouge_l(ext_section, ref_section) for each pair]
final_score = max(global_score, mean(section_scores))
```

This "forgiving ROUGE" approach accounts for structural reorganization in poster layouts.

### Number Capture

Evaluates preservation of quantitative data:

```python
# Extract all numbers from text
extracted_numbers = extract_numeric_values(extracted_text)
reference_numbers = extract_numeric_values(reference_text)

# Exclude DOIs and publication years from references
reference_numbers = filter_doi_components(reference_numbers)

number_capture = len(extracted_numbers & reference_numbers) / len(reference_numbers)
```

### Field Proportion

Measures structural completeness:

```python
extracted_fields = count_json_fields(extracted_json)
reference_fields = count_json_fields(reference_json)
field_proportion = extracted_fields / reference_fields
```

The range (0.50–1.50) accommodates:
- Nested vs flat section structures
- Variable poster layouts
- Optional metadata fields

## Text Normalization

Before comparison, text is normalized:

1. **Unicode normalization** (NFKD)
2. **Whitespace consolidation**
3. **Quote unification** (curly → straight)
4. **Dash normalization** (em/en dash → hyphen)
5. **Case normalization** (lowercase)

## Validation Results

### Current Performance

**Overall**: 19/20 (95%) passing

Extraction runs through the [poster2json](https://github.com/fairdataihub/poster2json)
library, validated against a 20-poster annotated corpus using `pdfplumber` (XY-cut
reading order, PyMuPDF fallback) for PDFs and Qwen2-VL for image posters.

### Aggregate Metrics

| Metric | Average Score |
|--------|---------------|
| Word Capture | 0.92 |
| ROUGE-L | 0.85 |
| Number Capture | 0.97 |
| Field Proportion | 0.88 |

The single failing poster (a dense table/flowchart layout) misses the ROUGE-L
threshold at 0.71; its text is fully captured, but the annotator's fine-grained
section segmentation differs from the merged sections the model produces.

Per-poster results and the full methodology live in the authoritative
[poster2json evaluation docs](https://github.com/fairdataihub/poster2json/blob/main/docs/evaluation.md).

## Test Set

The validation set includes 20 manually annotated scientific posters:

- **19 PDF posters**: Processed via pdfplumber (XY-cut reading order)
- **1 image poster**: Processed via Qwen2-VL

Posters cover diverse formats:
- Single and multi-column layouts
- Various font sizes and styles
- Tables, figures, and charts
- Multiple languages

## Running Validation

Validation is run from the
[poster2json-validation](https://github.com/fairdataihub/poster2json-validation)
repository, which scores generated JSON against the annotated corpus:

```bash
python validate_model.py --text-extractor pdfplumber
```

Output (under `outputs/<timestamp>/`):
- Individual `{poster_id}_extracted.json` files
- `results.json` with all metrics

## Reference Annotations

Ground truth annotations are stored in `manual_poster_annotation/`:

```
manual_poster_annotation/
├── {poster_id}/
│   ├── {poster_id}.pdf         # Source poster
│   ├── {poster_id}_sub-json.json  # Ground truth annotation
│   └── {poster_id}_raw.md      # Extracted raw text
```

## See Also

- [Architecture](ARCHITECTURE.md) - Technical details
- [API Reference](API.md) - REST API documentation

