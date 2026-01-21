"""
Poster extraction validation

Validates LLM output against the extraction schema with auto-fix capabilities.
Uses "auto-fix and warn" strategy - attempts to repair common issues and
includes warnings in the response.
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
EXTRACTION_SCHEMA_PATH = SCHEMA_DIR / "poster_extraction_schema.json"           # What the model currently outputs
FULL_SCHEMA_PATH = SCHEMA_DIR / "poster_schema.json"

# Schema version for tracking
SCHEMA_VERSION = "extraction-v0.1"


@dataclass
class ValidationWarning:
    """Represents a validation warning with optional auto-fix information."""

    field: str
    issue: str
    message: str
    auto_fixed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_extraction_schema() -> dict:
    """Load the extraction schema from disk."""
    with open(EXTRACTION_SCHEMA_PATH) as f:
        return json.load(f)


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
        "imageCaption": [],
        "tableCaption": [],
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
    by the frontend's formSchema, even if the model didn't extract them.

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
    Validate LLM extraction output and apply auto-fixes where possible.

    Args:
        data: The extracted JSON from the LLM
        strict: If True, don't apply auto-fixes, just report errors

    Returns:
        Tuple of (fixed_data, warnings_list)
        - fixed_data: The data with auto-fixes applied
        - warnings_list: List of warning dicts describing issues found
    """
    # Don't modify original
    result = copy.deepcopy(data)
    warnings: List[ValidationWarning] = []

    # Handle error responses from extraction
    if "error" in result:
        return result, []

    # Phase 1: Structural auto-fixes (before schema validation)
    if not strict:
        result, pre_warnings = _apply_structural_fixes(result)
        warnings.extend(pre_warnings)

    # Phase 2: Schema validation
    schema = load_extraction_schema()
    validator = Draft202012Validator(schema)

    schema_errors = list(validator.iter_errors(result))

    # Phase 3: Report remaining schema errors as warnings
    for error in schema_errors:
        warning = _create_warning_from_error(error)
        if warning:
            warnings.append(warning)

    # Phase 4: Re-validate to get final count
    remaining_errors = list(validator.iter_errors(result))

    # Add validation metadata
    result["_validation"] = {
        "schema_version": SCHEMA_VERSION,
        "valid": len(remaining_errors) == 0,
        "warnings_count": len(warnings),
        "errors_count": len(remaining_errors),
    }

    # Phase 5: Populate missing fields for complete response
    result, filled_fields = populate_missing_fields(result)
    result["_validation"]["filled_fields"] = filled_fields

    # Convert warnings to dicts for JSON serialization
    warnings_list = [w.to_dict() for w in warnings]

    return result, warnings_list


def _apply_structural_fixes(data: dict) -> Tuple[dict, List[ValidationWarning]]:
    """
    Apply structural fixes before schema validation.

    These fixes handle common LLM output issues that would cause
    schema validation to fail.
    """
    warnings = []

    # Fix 1: Ensure creators exists and is an array
    if "creators" not in data:
        data["creators"] = []
        warnings.append(
            ValidationWarning(
                field="creators",
                issue="missing_required",
                message="Missing required field 'creators' - added empty array",
                auto_fixed=True,
            )
        )
    elif not isinstance(data["creators"], list):
        if isinstance(data["creators"], dict):
            data["creators"] = [data["creators"]]
            warnings.append(
                ValidationWarning(
                    field="creators",
                    issue="wrong_type",
                    message="Field 'creators' was object, wrapped in array",
                    auto_fixed=True,
                )
            )
        else:
            data["creators"] = []
            warnings.append(
                ValidationWarning(
                    field="creators",
                    issue="wrong_type",
                    message="Field 'creators' had invalid type, reset to empty array",
                    auto_fixed=True,
                )
            )

    # Fix 2: Ensure titles exists and is an array
    if "titles" not in data:
        data["titles"] = []
        warnings.append(
            ValidationWarning(
                field="titles",
                issue="missing_required",
                message="Missing required field 'titles' - added empty array",
                auto_fixed=True,
            )
        )
    elif not isinstance(data["titles"], list):
        if isinstance(data["titles"], dict):
            data["titles"] = [data["titles"]]
            warnings.append(
                ValidationWarning(
                    field="titles",
                    issue="wrong_type",
                    message="Field 'titles' was object, wrapped in array",
                    auto_fixed=True,
                )
            )
        else:
            data["titles"] = []
            warnings.append(
                ValidationWarning(
                    field="titles",
                    issue="wrong_type",
                    message="Field 'titles' had invalid type, reset to empty array",
                    auto_fixed=True,
                )
            )

    # Fix 3: Ensure posterContent exists
    if "posterContent" not in data:
        data["posterContent"] = {"sections": []}
        warnings.append(
            ValidationWarning(
                field="posterContent",
                issue="missing_required",
                message="Missing required field 'posterContent' - added empty structure",
                auto_fixed=True,
            )
        )
    elif not isinstance(data["posterContent"], dict):
        data["posterContent"] = {"sections": []}
        warnings.append(
            ValidationWarning(
                field="posterContent",
                issue="wrong_type",
                message="Field 'posterContent' had invalid type, reset to empty structure",
                auto_fixed=True,
            )
        )

    # Fix 4: Ensure imageCaption and tableCaption are arrays (optional fields)
    for field in ["imageCaption", "tableCaption"]:
        if field in data and not isinstance(data[field], list):
            if isinstance(data[field], dict):
                data[field] = [data[field]]
            else:
                data[field] = []

    # Fix 4b: Remove empty caption fields from imageCaption items
    if "imageCaption" in data and isinstance(data["imageCaption"], list):
        for i, caption_obj in enumerate(data["imageCaption"]):
            if isinstance(caption_obj, dict):
                # Find and remove empty string fields
                empty_fields = [
                    key for key, value in caption_obj.items()
                    if isinstance(value, str) and len(value.strip()) < 1
                ]
                for field in empty_fields:
                    del caption_obj[field]
                    warnings.append(
                        ValidationWarning(
                            field=f"imageCaption[{i}].{field}",
                            issue="empty_string",
                            message=f"Removed empty caption field '{field}' from imageCaption[{i}]",
                            auto_fixed=True,
                        )
                    )

    # Fix 5: Clean up creators - fix affiliations and add missing required fields
    # Process creators in-place, converting invalid items to placeholders
    for i in range(len(data.get("creators", []))):
        creator = data["creators"][i]

        # Convert non-dict items to placeholder
        if not isinstance(creator, dict):
            data["creators"][i] = {"name": ""}
            warnings.append(
                ValidationWarning(
                    field=f"creators[{i}]",
                    issue="invalid_item_converted",
                    message=f"Converted invalid creator at index {i} (was {type(creator).__name__}) to placeholder",
                    auto_fixed=True,
                )
            )
            continue

        # Add empty name if missing (preserve and fill strategy)
        if "name" not in creator:
            creator["name"] = ""
            warnings.append(
                ValidationWarning(
                    field=f"creators[{i}].name",
                    issue="missing_required_field",
                    message=f"Added empty 'name' field to creator at index {i}",
                    auto_fixed=True,
                )
            )
        elif not creator["name"]:
            # Name exists but is empty - warn but don't modify
            warnings.append(
                ValidationWarning(
                    field=f"creators[{i}].name",
                    issue="empty_required_field",
                    message=f"Creator at index {i} has empty 'name' field - requires human review",
                    auto_fixed=False,
                )
            )

        # Fix affiliation if it's a string instead of array
        if "affiliation" in creator:
            if isinstance(creator["affiliation"], str):
                creator["affiliation"] = [{"name": creator["affiliation"]}]
                warnings.append(
                    ValidationWarning(
                        field=f"creators[{i}].affiliation",
                        issue="string_not_array",
                        message=f"Converted string affiliation to array format for creator at index {i}",
                        auto_fixed=True,
                    )
                )
            elif isinstance(creator["affiliation"], list):
                # Fix individual affiliations that are strings or missing required 'name'
                for j, aff in enumerate(creator["affiliation"]):
                    if isinstance(aff, str):
                        creator["affiliation"][j] = {"name": aff}
                        warnings.append(
                            ValidationWarning(
                                field=f"creators[{i}].affiliation[{j}]",
                                issue="string_not_object",
                                message="Converted string affiliation to object format",
                                auto_fixed=True,
                            )
                        )
                    elif isinstance(aff, dict):
                        # Add missing 'name' field to affiliation objects
                        if "name" not in aff:
                            aff["name"] = ""
                            warnings.append(
                                ValidationWarning(
                                    field=f"creators[{i}].affiliation[{j}].name",
                                    issue="missing_required_field",
                                    message=f"Added empty 'name' to affiliation at creators[{i}].affiliation[{j}]",
                                    auto_fixed=True,
                                )
                            )

        # Fix nameIdentifiers - add missing 'nameIdentifier' field
        if "nameIdentifiers" in creator and isinstance(creator["nameIdentifiers"], list):
            for j, nid in enumerate(creator["nameIdentifiers"]):
                if isinstance(nid, dict) and "nameIdentifier" not in nid:
                    nid["nameIdentifier"] = ""
                    warnings.append(
                        ValidationWarning(
                            field=f"creators[{i}].nameIdentifiers[{j}].nameIdentifier",
                            issue="missing_required_field",
                            message=f"Added empty 'nameIdentifier' to creators[{i}].nameIdentifiers[{j}]",
                            auto_fixed=True,
                        )
                    )

        # Validate nameType enum if present
        if "nameType" in creator:
            if creator["nameType"] not in ["Personal", "Organizational"]:
                del creator["nameType"]
                warnings.append(
                    ValidationWarning(
                        field=f"creators[{i}].nameType",
                        issue="invalid_enum",
                        message=f"Removed invalid nameType value for creator at index {i}",
                        auto_fixed=True,
                    )
                )

    # Fix 6: Clean up titles - add missing required fields
    # Process titles, converting invalid items to placeholders
    for i in range(len(data.get("titles", []))):
        title_obj = data["titles"][i]

        # Convert non dict items to placeholder
        if not isinstance(title_obj, dict):
            data["titles"][i] = {"title": ""}
            warnings.append(
                ValidationWarning(
                    field=f"titles[{i}]",
                    issue="invalid_item_converted",
                    message=f"Converted invalid title at index {i} (was {type(title_obj).__name__}) to placeholder",
                    auto_fixed=True,
                )
            )
            continue

        # Add empty title if missing (preserve and fill strategy)
        if "title" not in title_obj:
            title_obj["title"] = ""
            warnings.append(
                ValidationWarning(
                    field=f"titles[{i}].title",
                    issue="missing_required_field",
                    message=f"Added empty 'title' field to title at index {i}",
                    auto_fixed=True,
                )
            )
        elif not title_obj["title"]:
            # Title exists but is empty - warn but don't modify
            warnings.append(
                ValidationWarning(
                    field=f"titles[{i}].title",
                    issue="empty_required_field",
                    message=f"Title at index {i} has empty 'title' field - requires human review",
                    auto_fixed=False,
                )
            )

        # Validate titleType enum if present
        if "titleType" in title_obj:
            valid_types = ["AlternativeTitle", "Subtitle", "TranslatedTitle", "Other"]
            if title_obj["titleType"] not in valid_types:
                del title_obj["titleType"]
                warnings.append(
                    ValidationWarning(
                        field=f"titles[{i}].titleType",
                        issue="invalid_enum",
                        message=f"Removed invalid titleType value for title at index {i}",
                        auto_fixed=True,
                    )
                )

    # Fix 7: Clean up posterContent sections
    if "posterContent" in data and isinstance(data["posterContent"], dict):
        if "sections" in data["posterContent"]:
            if not isinstance(data["posterContent"]["sections"], list):
                data["posterContent"]["sections"] = []
                warnings.append(
                    ValidationWarning(
                        field="posterContent.sections",
                        issue="wrong_type",
                        message="Field 'posterContent.sections' had invalid type, reset to empty array",
                        auto_fixed=True,
                    )
                )
            else:
                # Clean up invalid sections
                valid_sections = []
                for i, section in enumerate(data["posterContent"]["sections"]):
                    if not isinstance(section, dict):
                        warnings.append(
                            ValidationWarning(
                                field=f"posterContent.sections[{i}]",
                                issue="invalid_item",
                                message=f"Removed invalid section at index {i}: not an object",
                                auto_fixed=True,
                            )
                        )
                        continue
                    valid_sections.append(section)
                data["posterContent"]["sections"] = valid_sections

    # Fix 8: Add placeholder items for empty arrays with minItems: 1 requirement
    if len(data.get("creators", [])) == 0:
        data["creators"] = [{"name": ""}]
        warnings.append(
            ValidationWarning(
                field="creators",
                issue="empty_required_array",
                message="Added placeholder creator with empty 'name' - array requires at least 1 item",
                auto_fixed=True,
            )
        )

    if len(data.get("titles", [])) == 0:
        data["titles"] = [{"title": ""}]
        warnings.append(
            ValidationWarning(
                field="titles",
                issue="empty_required_array",
                message="Added placeholder title with empty 'title' - array requires at least 1 item",
                auto_fixed=True,
            )
        )

    return data, warnings


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
                auto_fixed=False,
            )

    # Handle minLength errors for strings
    if error.validator == "minLength":
        return ValidationWarning(
            field=field_path,
            issue="empty_string",
            message=f"Field '{field_path}' has empty string value",
            auto_fixed=False,
        )

    # Handle type errors
    if error.validator == "type":
        return ValidationWarning(
            field=field_path,
            issue="wrong_type",
            message=f"Field '{field_path}' has wrong type: expected {error.validator_value}",
            auto_fixed=False,
        )

    # Handle required field errors
    if error.validator == "required":
        missing = [f for f in error.validator_value if f not in error.instance]
        if missing:
            return ValidationWarning(
                field=field_path,
                issue="missing_required",
                message=f"Missing required field(s) at '{field_path}': {', '.join(missing)}",
                auto_fixed=False,
            )

    # Handle enum errors
    if error.validator == "enum":
        return ValidationWarning(
            field=field_path,
            issue="invalid_enum",
            message=f"Invalid value at '{field_path}': must be one of {error.validator_value}",
            auto_fixed=False,
        )

    return None
