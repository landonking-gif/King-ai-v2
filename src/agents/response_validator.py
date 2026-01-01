"""
Agent Response Validator.
Validates and normalizes LLM responses with JSON schema validation and type coercion.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union, get_origin, get_args
from enum import Enum

from src.utils.structured_logging import get_logger

logger = get_logger("response_validator")


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or [message]


class ResponseFormat(str, Enum):
    """Expected response format."""
    JSON = "json"
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    LIST = "list"


@dataclass
class ValidationResult:
    """Result of response validation."""
    valid: bool
    data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned: bool = False
    original: Optional[str] = None


class ResponseCleaner:
    """Cleans and normalizes LLM responses."""
    
    # Patterns for extracting JSON from various formats
    JSON_PATTERNS = [
        r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON block
        r'```\s*([\s\S]*?)\s*```',       # Generic code block
        r'\{[\s\S]*\}',                   # Raw JSON object
        r'\[[\s\S]*\]',                   # Raw JSON array
    ]
    
    # Common LLM response prefixes to strip
    STRIP_PREFIXES = [
        "Here is the response:",
        "Here's the JSON:",
        "The result is:",
        "Output:",
        "Response:",
        "Answer:",
    ]
    
    @classmethod
    def extract_json(cls, text: str) -> Optional[str]:
        """Extract JSON from text, handling various formats."""
        if not text:
            return None
        
        # Try each pattern
        for pattern in cls.JSON_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Try to parse each match
                for match in matches:
                    try:
                        # Clean up the match
                        cleaned = match.strip()
                        # Try to parse to validate it's JSON
                        json.loads(cleaned)
                        return cleaned
                    except json.JSONDecodeError:
                        continue
        
        return None
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean common LLM response artifacts."""
        if not text:
            return ""
        
        result = text.strip()
        
        # Remove common prefixes
        for prefix in cls.STRIP_PREFIXES:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        
        # Remove markdown code block markers if not extracting JSON
        if result.startswith("```") and result.endswith("```"):
            lines = result.split("\n")
            if len(lines) > 2:
                # Remove first and last lines (code block markers)
                result = "\n".join(lines[1:-1])
        
        return result
    
    @classmethod
    def fix_json(cls, text: str) -> str:
        """Attempt to fix common JSON errors."""
        if not text:
            return text
        
        result = text
        
        # Fix trailing commas
        result = re.sub(r',\s*([\]}])', r'\1', result)
        
        # Fix single quotes to double quotes
        # Only if it looks like JSON with single quotes
        if "'" in result and '"' not in result:
            result = result.replace("'", '"')
        
        # Fix unquoted keys
        result = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', result)
        
        # Fix True/False/None to JSON format
        result = re.sub(r'\bTrue\b', 'true', result)
        result = re.sub(r'\bFalse\b', 'false', result)
        result = re.sub(r'\bNone\b', 'null', result)
        
        return result


class SchemaValidator:
    """Validates data against a JSON-like schema."""
    
    @classmethod
    def validate(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str = "",
    ) -> List[str]:
        """
        Validate data against a schema.
        
        Schema format:
        {
            "type": "object",  # object, array, string, number, boolean, null
            "required": ["field1", "field2"],
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number", "minimum": 0},
            }
        }
        """
        errors = []
        
        schema_type = schema.get("type")
        
        if schema_type == "object":
            errors.extend(cls._validate_object(data, schema, path))
        elif schema_type == "array":
            errors.extend(cls._validate_array(data, schema, path))
        elif schema_type == "string":
            errors.extend(cls._validate_string(data, schema, path))
        elif schema_type == "number" or schema_type == "integer":
            errors.extend(cls._validate_number(data, schema, path))
        elif schema_type == "boolean":
            errors.extend(cls._validate_boolean(data, schema, path))
        elif schema_type == "null":
            if data is not None:
                errors.append(f"{path}: Expected null, got {type(data).__name__}")
        
        return errors
    
    @classmethod
    def _validate_object(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        errors = []
        
        if not isinstance(data, dict):
            return [f"{path}: Expected object, got {type(data).__name__}"]
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"{path}.{field}: Required field missing")
        
        # Validate properties
        properties = schema.get("properties", {})
        for key, value in data.items():
            if key in properties:
                field_path = f"{path}.{key}" if path else key
                errors.extend(cls.validate(value, properties[key], field_path))
        
        return errors
    
    @classmethod
    def _validate_array(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        errors = []
        
        if not isinstance(data, list):
            return [f"{path}: Expected array, got {type(data).__name__}"]
        
        # Validate length
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        
        if min_items is not None and len(data) < min_items:
            errors.append(f"{path}: Array has {len(data)} items, minimum is {min_items}")
        
        if max_items is not None and len(data) > max_items:
            errors.append(f"{path}: Array has {len(data)} items, maximum is {max_items}")
        
        # Validate items
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                errors.extend(cls.validate(item, item_schema, f"{path}[{i}]"))
        
        return errors
    
    @classmethod
    def _validate_string(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        errors = []
        
        if not isinstance(data, str):
            return [f"{path}: Expected string, got {type(data).__name__}"]
        
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        enum = schema.get("enum")
        
        if min_length is not None and len(data) < min_length:
            errors.append(f"{path}: String length {len(data)} is less than minimum {min_length}")
        
        if max_length is not None and len(data) > max_length:
            errors.append(f"{path}: String length {len(data)} exceeds maximum {max_length}")
        
        if pattern and not re.match(pattern, data):
            errors.append(f"{path}: String does not match pattern {pattern}")
        
        if enum and data not in enum:
            errors.append(f"{path}: Value '{data}' not in allowed values {enum}")
        
        return errors
    
    @classmethod
    def _validate_number(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        errors = []
        
        if not isinstance(data, (int, float)):
            return [f"{path}: Expected number, got {type(data).__name__}"]
        
        if schema.get("type") == "integer" and not isinstance(data, int):
            if not (isinstance(data, float) and data.is_integer()):
                errors.append(f"{path}: Expected integer, got float")
        
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        
        if minimum is not None and data < minimum:
            errors.append(f"{path}: Value {data} is less than minimum {minimum}")
        
        if maximum is not None and data > maximum:
            errors.append(f"{path}: Value {data} exceeds maximum {maximum}")
        
        return errors
    
    @classmethod
    def _validate_boolean(
        cls,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[str]:
        if not isinstance(data, bool):
            return [f"{path}: Expected boolean, got {type(data).__name__}"]
        return []


class ResponseValidator:
    """
    Validates and normalizes LLM responses.
    
    Features:
    - JSON extraction from various formats
    - Schema validation
    - Type coercion
    - Error recovery
    """
    
    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        response_format: ResponseFormat = ResponseFormat.JSON,
        strict: bool = False,
        coerce_types: bool = True,
    ):
        self.schema = schema
        self.response_format = response_format
        self.strict = strict
        self.coerce_types = coerce_types
    
    def validate(self, response: str) -> ValidationResult:
        """
        Validate an LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            ValidationResult with parsed data or errors
        """
        result = ValidationResult(valid=False, original=response)
        
        if not response:
            result.errors.append("Empty response")
            return result
        
        try:
            if self.response_format == ResponseFormat.JSON:
                return self._validate_json(response)
            elif self.response_format == ResponseFormat.TEXT:
                return self._validate_text(response)
            elif self.response_format == ResponseFormat.CODE:
                return self._validate_code(response)
            elif self.response_format == ResponseFormat.LIST:
                return self._validate_list(response)
            else:
                result.data = response
                result.valid = True
                return result
                
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            return result
    
    def _validate_json(self, response: str) -> ValidationResult:
        """Validate and extract JSON response."""
        result = ValidationResult(valid=False, original=response)
        
        # Try to extract JSON
        json_str = ResponseCleaner.extract_json(response)
        
        if not json_str:
            # Try cleaning and fixing
            cleaned = ResponseCleaner.clean_text(response)
            json_str = ResponseCleaner.fix_json(cleaned)
            result.cleaned = True
        
        if not json_str:
            result.errors.append("Could not extract JSON from response")
            return result
        
        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try fixing common issues
            fixed = ResponseCleaner.fix_json(json_str)
            try:
                data = json.loads(fixed)
                result.cleaned = True
                result.warnings.append(f"JSON was auto-fixed: {e}")
            except json.JSONDecodeError:
                result.errors.append(f"Invalid JSON: {e}")
                return result
        
        # Coerce types if enabled
        if self.coerce_types:
            data = self._coerce_types(data)
        
        # Validate against schema
        if self.schema:
            errors = SchemaValidator.validate(data, self.schema)
            if errors:
                if self.strict:
                    result.errors.extend(errors)
                    return result
                else:
                    result.warnings.extend(errors)
        
        result.valid = True
        result.data = data
        return result
    
    def _validate_text(self, response: str) -> ValidationResult:
        """Validate text response."""
        cleaned = ResponseCleaner.clean_text(response)
        
        return ValidationResult(
            valid=True,
            data=cleaned,
            cleaned=cleaned != response,
            original=response,
        )
    
    def _validate_code(self, response: str) -> ValidationResult:
        """Validate code response."""
        result = ValidationResult(valid=False, original=response)
        
        # Extract code from markdown blocks
        code_pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'
        matches = re.findall(code_pattern, response)
        
        if matches:
            code = matches[0]
            result.cleaned = True
        else:
            code = ResponseCleaner.clean_text(response)
        
        # Basic syntax check for Python
        if code.strip():
            try:
                compile(code, "<string>", "exec")
                result.valid = True
                result.data = code
            except SyntaxError as e:
                result.warnings.append(f"Code syntax warning: {e}")
                result.valid = True  # Still return the code
                result.data = code
        else:
            result.errors.append("Empty code response")
        
        return result
    
    def _validate_list(self, response: str) -> ValidationResult:
        """Validate list response."""
        result = ValidationResult(valid=False, original=response)
        
        # Try JSON array first
        json_result = self._validate_json(response)
        if json_result.valid and isinstance(json_result.data, list):
            return json_result
        
        # Try line-by-line parsing
        lines = response.strip().split("\n")
        items = []
        
        for line in lines:
            line = line.strip()
            # Remove common list markers
            line = re.sub(r'^[-*•]\s*', '', line)
            line = re.sub(r'^\d+[.)]\s*', '', line)
            
            if line:
                items.append(line)
        
        if items:
            result.valid = True
            result.data = items
            result.cleaned = True
        else:
            result.errors.append("Could not parse list from response")
        
        return result
    
    def _coerce_types(self, data: Any) -> Any:
        """Coerce data types based on schema hints."""
        if not self.schema:
            return data
        
        return self._coerce_value(data, self.schema)
    
    def _coerce_value(self, value: Any, schema: Dict[str, Any]) -> Any:
        """Coerce a single value based on schema."""
        schema_type = schema.get("type")
        
        if schema_type == "number" and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        
        elif schema_type == "integer" and isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        
        elif schema_type == "boolean" and isinstance(value, str):
            if value.lower() in ("true", "yes", "1"):
                return True
            elif value.lower() in ("false", "no", "0"):
                return False
        
        elif schema_type == "object" and isinstance(value, dict):
            properties = schema.get("properties", {})
            return {
                k: self._coerce_value(v, properties.get(k, {}))
                for k, v in value.items()
            }
        
        elif schema_type == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            return [self._coerce_value(item, item_schema) for item in value]
        
        return value


def validate_response(
    response: str,
    schema: Optional[Dict[str, Any]] = None,
    format: ResponseFormat = ResponseFormat.JSON,
    strict: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate an LLM response.
    
    Usage:
        result = validate_response(
            llm_output,
            schema={"type": "object", "required": ["answer"]},
        )
        if result.valid:
            print(result.data)
    """
    validator = ResponseValidator(
        schema=schema,
        response_format=format,
        strict=strict,
    )
    return validator.validate(response)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from an LLM response.
    
    Returns None if no valid JSON found.
    """
    result = validate_response(response, format=ResponseFormat.JSON)
    return result.data if result.valid else None
