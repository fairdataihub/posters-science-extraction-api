"""
Poster extraction validation

Validates LLM output against the extraction schema.
Reports warnings for any schema violations.

NOTE: Format normalization (caption structure, creator fields, affiliations)
is handled by postprocess_json() in poster_extraction.py. This module only
validates and reports - it does not modify the data (except adding metadata
and populating missing optional fields with defaults).
"""

import json
import copy
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import jsonschema
from jsonschema import Draft202012Validator

# Schema file path
SCHEMA_DIR = Path(__file__).parent
EXTRACTION_SCHEMA_PATH = SCHEMA_DIR / "poster_extraction_schema.json"
FULL_SCHEMA_PATH = SCHEMA_DIR / "poster_schema.json"

# Schema version for tracking
SCHEMA_VERSION = "extraction-v0.1"

# Cache the schema in memory after first load
_SCHEMA_CACHE: Optional[dict] = None


@dataclass
class ValidationWarning:
    """Represents a validation warning."""

    field: str
    issue: str
    message: str
    auto_fixed: bool = False  # Always False now - we don't auto-fix

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_extraction_schema() -> dict:
    """Load the extraction schema, using cached version if available."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        with open(EXTRACTION_SCHEMA_PATH) as f:
            _SCHEMA_CACHE = json.load(f)
    return _SCHEMA_CACHE


def get_empty_field_defaults() -> dict:
    """
    Return empty/default values for all fields in the full schema.

    Defaults to ensure the API always returns a complete, predictable structure.
    """
    return {
        "doi": "",
        "prefix": "",
        "suffix": "",
        "identifiers": [],
        "alternateIdentifiers": [],
        "creators": [],
        "titles": [],
        "subjects": [],
        "dates": [],
        "relatedIdentifiers": [],
        "sizes": [],
        "formats": [],
        "rightsList": [],
        "descriptions": [],
        "fundingReferences": [],
        "ethicsApprovals": [],
        "imageCaptions": [],
        "tableCaptions": [],
        "publisher": {},
        "types": {},
        "conference": {},
        "posterContent": {"sections": []},
        "publicationYear": None,
        "language": "en",
        "version": "",
        "domain": "",
    }


def populate_missing_fields(data: dict) -> Tuple[dict, List[str]]:
    """
    Fill in missing fields with empty/default values.

    This ensures the API response always contains all fields expected
    by the poster json schema.

    Args:
        data: The extraction result after validation

    Returns:
        Tuple of (complete_data, list_of_filled_field_names)
    """
    defaults = get_empty_field_defaults()
    filled_fields = []

    for field, default_value in defaults.items():
        if field not in data:
            data[field] = default_value
            filled_fields.append(field)

    return data, filled_fields


def validate_and_fix_extraction(
    data: dict, strict: bool = False
) -> Tuple[dict, List[Dict[str, Any]]]:
    """
    Validate LLM extraction output and report any schema violations.

    NOTE: This function no longer applies auto-fixes. Format normalization
    is handled upstream by postprocess_json() in poster_extraction.py.
    The 'strict' parameter is kept for API compatibility but has no effect.

    Args:
        data: The extracted JSON from the LLM (already post-processed)
        strict: Kept for API compatibility (no longer used)

    Returns:
        Tuple of (validated_data, warnings_list)
        - validated_data: The data with validation metadata added
        - warnings_list: List of warning dicts describing any issues found
    """
    # Don't modify original
    result = copy.deepcopy(data)
    warnings: List[ValidationWarning] = []

    # Handle error responses from extraction
    if "error" in result:
        return result, []

    # Schema validation
    schema = load_extraction_schema()
    validator = Draft202012Validator(schema)

    # Collect all schema errors as warnings
    schema_errors = list(validator.iter_errors(result))
    for error in schema_errors:
        warning = _create_warning_from_error(error)
        if warning:
            warnings.append(warning)

    # Add validation metadata
    result["_validation"] = {
        "schema_version": SCHEMA_VERSION,
        "valid": len(schema_errors) == 0,
        "warnings_count": len(warnings),
        "errors_count": len(schema_errors),
    }

    # Populate missing optional fields for complete API response
    result, filled_fields = populate_missing_fields(result)
    result["_validation"]["filled_fields"] = filled_fields

    # Convert warnings to dicts for JSON serialization
    warnings_list = [w.to_dict() for w in warnings]

    return result, warnings_list


def _create_warning_from_error(
    error: jsonschema.ValidationError,
) -> Optional[ValidationWarning]:
    """
    Create a warning from a schema validation error.

    Returns:
        ValidationWarning or None if error should be ignored
    """
    path = list(error.absolute_path)
    field_path = ".".join(str(p) for p in path) if path else "root"

    # Handle minItems errors for required arrays
    if error.validator == "minItems":
        if len(path) == 1 and path[0] in ["creators", "titles"]:
            return ValidationWarning(
                field=path[0],
                issue="empty_required_array",
                message=f"Required field '{path[0]}' is empty - extraction may have failed",
            )

    # Handle minLength errors for strings
    if error.validator == "minLength":
        return ValidationWarning(
            field=field_path,
            issue="empty_string",
            message=f"Field '{field_path}' has empty string value",
        )

    # Handle type errors
    if error.validator == "type":
        return ValidationWarning(
            field=field_path,
            issue="wrong_type",
            message=f"Field '{field_path}' has wrong type: expected {error.validator_value}",
        )

    # Handle required field errors
    if error.validator == "required":
        missing = [f for f in error.validator_value if f not in error.instance]
        if missing:
            return ValidationWarning(
                field=field_path,
                issue="missing_required",
                message=f"Missing required field(s) at '{field_path}': {', '.join(missing)}",
            )

    # Handle enum errors
    if error.validator == "enum":
        return ValidationWarning(
            field=field_path,
            issue="invalid_enum",
            message=f"Invalid value at '{field_path}': must be one of {error.validator_value}",
        )

    # Handle oneOf errors (e.g., affiliation can be string or object)
    if error.validator == "oneOf":
        return ValidationWarning(
            field=field_path,
            issue="invalid_format",
            message=f"Field '{field_path}' doesn't match expected format",
        )

    return None
