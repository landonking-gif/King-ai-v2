"""
JSON Schema validation for subagent outputs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
from jsonschema import Draft7Validator, RefResolver


class SchemaValidator:
    """
    Validates subagent outputs against registered JSON schemas.

    Schemas are loaded from the schema registry directory and cached for performance.
    """

    def __init__(self, schema_registry_path: str) -> None:
        """
        Initialize schema validator.

        Args:
            schema_registry_path: Path to directory containing JSON schemas
        """
        self.schema_registry_path = Path(schema_registry_path)
        self.schema_cache: Dict[str, Dict[str, Any]] = {}
        self.resolver: Optional[RefResolver] = None
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load all schemas from the registry directory."""
        if not self.schema_registry_path.exists():
            # Create directory structure if it doesn't exist
            self.schema_registry_path.mkdir(parents=True, exist_ok=True)
            # Create default schemas
            self._create_default_schemas()

        # Load all JSON schema files
        for schema_file in self.schema_registry_path.glob("**/*.json"):
            schema_name = schema_file.stem
            try:
                with open(schema_file, "r") as f:
                    schema = json.load(f)
                    self.schema_cache[schema_name] = schema
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to load schema {schema_file}: {e}")

        # Set up resolver for $ref resolution
        if self.schema_cache:
            store = {
                f"file://{self.schema_registry_path}/{name}.json": schema
                for name, schema in self.schema_cache.items()
            }
            self.resolver = RefResolver.from_schema(
                next(iter(self.schema_cache.values())),
                store=store,
            )

    def _create_default_schemas(self) -> None:
        """Create default artifact schemas."""
        # Research snippet schema
        research_snippet_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "Research Snippet",
            "required": ["id", "text", "summary", "created_at"],
            "properties": {
                "id": {"type": "string", "description": "Unique identifier"},
                "source": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri"},
                        "doc_id": {"type": "string"},
                    },
                },
                "text": {"type": "string", "description": "Research content"},
                "summary": {"type": "string", "description": "Brief summary"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "provenance_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "safety_class": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential"],
                },
                "created_by": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
            },
        }

        # Claim verification schema
        claim_verification_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "Claim Verification",
            "required": ["id", "claim_text", "verdict", "confidence", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "claim_text": {"type": "string"},
                "verdict": {
                    "type": "string",
                    "enum": ["verified", "refuted", "inconclusive"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "disagreement_notes": {"type": "string"},
                "method": {"type": "string"},
                "verifier": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "action_suggestion": {"type": "string"},
            },
        }

        # Code patch schema
        code_patch_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": "Code Patch",
            "required": ["id", "files_changed", "patch_summary", "created_at"],
            "properties": {
                "id": {"type": "string"},
                "repo": {"type": "string"},
                "base_commit": {"type": "string"},
                "files_changed": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "diff": {"type": "string"},
                        },
                    },
                },
                "patch_summary": {"type": "string"},
                "tests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "authoring_subagent": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "merge_ready": {"type": "boolean"},
            },
        }

        # Write schemas to files
        schemas = {
            "research_snippet": research_snippet_schema,
            "claim_verification": claim_verification_schema,
            "code_patch": code_patch_schema,
        }

        for name, schema in schemas.items():
            schema_path = self.schema_registry_path / f"{name}.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)

    def validate(self, data: Dict[str, Any], schema_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate data against a registered schema.

        Args:
            data: Data to validate
            schema_name: Name of the schema to validate against

        Returns:
            Tuple of (is_valid, error_message)
        """
        if schema_name not in self.schema_cache:
            return False, f"Schema '{schema_name}' not found in registry"

        schema = self.schema_cache[schema_name]

        try:
            # Use Draft7Validator for comprehensive validation
            validator = Draft7Validator(schema, resolver=self.resolver)
            validator.validate(data)
            return True, None
        except jsonschema.ValidationError as e:
            error_msg = f"Validation failed: {e.message} at {'.'.join(str(p) for p in e.path)}"
            return False, error_msg
        except Exception as e:
            return False, f"Unexpected validation error: {str(e)}"

    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a schema by name.

        Args:
            schema_name: Name of the schema

        Returns:
            Schema dict or None if not found
        """
        return self.schema_cache.get(schema_name)

    def list_schemas(self) -> List[str]:
        """
        List all available schema names.

        Returns:
            List of schema names
        """
        return list(self.schema_cache.keys())
