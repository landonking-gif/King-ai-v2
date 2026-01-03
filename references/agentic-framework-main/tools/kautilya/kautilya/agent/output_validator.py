"""
Output Validator for Kautilya.

Provides self-critique and validation:
- Validates outputs against expected schemas
- Checks for completeness and accuracy
- Detects potential issues or hallucinations
- Suggests improvements before returning to user
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation."""

    BASIC = "basic"  # Quick structural checks
    STANDARD = "standard"  # Standard validation
    STRICT = "strict"  # Comprehensive validation with LLM


class IssueType(Enum):
    """Types of validation issues."""

    INCOMPLETE = "incomplete"
    INCONSISTENT = "inconsistent"
    POTENTIALLY_INCORRECT = "potentially_incorrect"
    MISSING_SOURCES = "missing_sources"
    FORMAT_ERROR = "format_error"
    CONFIDENCE_LOW = "confidence_low"
    HALLUCINATION_RISK = "hallucination_risk"


class IssueSeverity(Enum):
    """Severity of validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A validation issue found in the output."""

    type: IssueType
    severity: IssueSeverity
    message: str
    location: Optional[str] = None  # Where in the output
    suggestion: Optional[str] = None  # How to fix

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.type.value}: {self.message}"


@dataclass
class ValidationResult:
    """Result of output validation."""

    is_valid: bool
    confidence: float  # 0.0 - 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    improved_output: Optional[str] = None  # Suggested improvement

    def get_summary(self) -> str:
        """Get a summary of validation results."""
        lines = []
        lines.append(f"Valid: {self.is_valid}")
        lines.append(f"Confidence: {self.confidence:.0%}")

        if self.issues:
            lines.append(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        if self.suggestions:
            lines.append(f"\nSuggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)


class OutputValidator:
    """
    Output validator with self-critique capabilities.

    Validates outputs for:
    - Structural correctness
    - Completeness
    - Consistency
    - Accuracy (when possible)
    - Potential hallucinations
    """

    CRITIQUE_PROMPT = """You are a quality assurance expert reviewing an AI assistant's response.

Original task: {task}

Generated response:
{response}

Evaluate the response on these criteria:
1. COMPLETENESS: Does it fully address the task?
2. ACCURACY: Are the facts and information likely correct?
3. CONSISTENCY: Is it internally consistent?
4. RELEVANCE: Is everything relevant to the task?
5. SOURCES: Are claims properly supported?

For each criterion, rate 1-5 and note any issues.

Respond in JSON format:
{{
    "completeness": {{"score": 1-5, "issues": [...]}},
    "accuracy": {{"score": 1-5, "issues": [...]}},
    "consistency": {{"score": 1-5, "issues": [...]}},
    "relevance": {{"score": 1-5, "issues": [...]}},
    "sources": {{"score": 1-5, "issues": [...]}},
    "overall_valid": true/false,
    "suggestions": [...]
}}"""

    # Patterns that might indicate hallucination (lowercase for case-insensitive matching)
    HALLUCINATION_PATTERNS = [
        r"i believe",
        r"i think",
        r"probably",
        r"might be",
        r"could be",
        r"as far as i know",
        r"to my knowledge",
        r"i'm not sure",
        r"approximately",
        r"around \d+",
    ]

    # Patterns for unsupported claims
    UNSUPPORTED_CLAIM_PATTERNS = [
        r"studies show",
        r"research shows",
        r"research indicates",
        r"experts say",
        r"according to",
        r"it is well known",
        r"everyone knows",
    ]

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        default_level: ValidationLevel = ValidationLevel.STANDARD,
    ):
        """
        Initialize output validator.

        Args:
            llm_client: LLM client for advanced validation
            default_level: Default validation level
        """
        self.llm_client = llm_client
        self.default_level = default_level

    def validate(
        self,
        output: Any,
        task: Optional[str] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        level: Optional[ValidationLevel] = None,
    ) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: The output to validate
            task: The original task (for context)
            expected_schema: JSON schema to validate against
            level: Validation level

        Returns:
            ValidationResult with issues and suggestions
        """
        level = level or self.default_level
        issues: List[ValidationIssue] = []
        suggestions: List[str] = []

        output_str = str(output) if not isinstance(output, str) else output

        # Basic validation
        basic_issues = self._basic_validation(output_str)
        issues.extend(basic_issues)

        # Schema validation
        if expected_schema:
            schema_issues = self._schema_validation(output, expected_schema)
            issues.extend(schema_issues)

        # Content validation
        if level in (ValidationLevel.STANDARD, ValidationLevel.STRICT):
            content_issues = self._content_validation(output_str, task)
            issues.extend(content_issues)

        # Calculate confidence
        confidence = self._calculate_confidence(issues)

        # Determine validity
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == IssueSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) <= 1

        # Generate suggestions
        suggestions = self._generate_suggestions(issues, output_str)

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
        )

    async def validate_with_llm(
        self,
        output: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate using LLM-based critique.

        Args:
            output: The output to validate
            task: The original task
            context: Additional context

        Returns:
            ValidationResult with LLM critique
        """
        if not self.llm_client:
            return self.validate(output, task, level=ValidationLevel.STANDARD)

        # First do basic validation
        basic_result = self.validate(output, task, level=ValidationLevel.STANDARD)

        # Then do LLM critique
        critique_prompt = self.CRITIQUE_PROMPT.format(
            task=task,
            response=output[:2000],  # Truncate for token limits
        )

        try:
            response = await self.llm_client.chat(critique_prompt)
            llm_issues, llm_suggestions = self._parse_critique(response)

            # Merge results
            all_issues = basic_result.issues + llm_issues
            all_suggestions = basic_result.suggestions + llm_suggestions

            confidence = self._calculate_confidence(all_issues)
            is_valid = confidence > 0.6

            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                issues=all_issues,
                suggestions=all_suggestions,
            )

        except Exception as e:
            logger.warning(f"LLM critique failed: {e}")
            return basic_result

    def _basic_validation(self, output: str) -> List[ValidationIssue]:
        """Perform basic structural validation."""
        issues = []

        # Check for empty output
        if not output or not output.strip():
            issues.append(
                ValidationIssue(
                    type=IssueType.INCOMPLETE,
                    severity=IssueSeverity.CRITICAL,
                    message="Output is empty",
                    suggestion="Regenerate response",
                )
            )
            return issues

        # Check for very short output
        if len(output.strip()) < 10:
            issues.append(
                ValidationIssue(
                    type=IssueType.INCOMPLETE,
                    severity=IssueSeverity.WARNING,
                    message="Output is very short",
                    suggestion="Consider providing more detail",
                )
            )

        # Check for error indicators
        error_patterns = [
            r"error:",
            r"failed:",
            r"exception:",
            r"cannot",
            r"unable to",
        ]
        for pattern in error_patterns:
            if re.search(pattern, output.lower()):
                issues.append(
                    ValidationIssue(
                        type=IssueType.POTENTIALLY_INCORRECT,
                        severity=IssueSeverity.WARNING,
                        message=f"Output may contain error: '{pattern}'",
                        location=pattern,
                    )
                )

        return issues

    def _schema_validation(
        self, output: Any, schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate output against JSON schema."""
        issues = []

        # Try to parse as JSON if string
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                issues.append(
                    ValidationIssue(
                        type=IssueType.FORMAT_ERROR,
                        severity=IssueSeverity.ERROR,
                        message="Output is not valid JSON",
                        suggestion="Ensure output is properly formatted JSON",
                    )
                )
                return issues

        # Check required fields
        required = schema.get("required", [])
        if isinstance(output, dict):
            for field in required:
                if field not in output:
                    issues.append(
                        ValidationIssue(
                            type=IssueType.INCOMPLETE,
                            severity=IssueSeverity.ERROR,
                            message=f"Missing required field: {field}",
                            location=field,
                            suggestion=f"Add the '{field}' field to the output",
                        )
                    )

        # Check property types
        properties = schema.get("properties", {})
        if isinstance(output, dict):
            for prop, prop_schema in properties.items():
                if prop in output:
                    expected_type = prop_schema.get("type")
                    actual_value = output[prop]

                    if not self._check_type(actual_value, expected_type):
                        issues.append(
                            ValidationIssue(
                                type=IssueType.FORMAT_ERROR,
                                severity=IssueSeverity.WARNING,
                                message=f"Field '{prop}' has wrong type. Expected {expected_type}",
                                location=prop,
                            )
                        )

        return issues

    def _check_type(self, value: Any, expected_type: Optional[str]) -> bool:
        """Check if value matches expected JSON type."""
        if expected_type is None:
            return True

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True

        return isinstance(value, expected_python_type)

    def _content_validation(
        self, output: str, task: Optional[str]
    ) -> List[ValidationIssue]:
        """Validate content quality."""
        issues = []
        output_lower = output.lower()

        # Check for hallucination indicators
        hallucination_count = 0
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, output_lower):
                hallucination_count += 1

        if hallucination_count >= 3:
            issues.append(
                ValidationIssue(
                    type=IssueType.HALLUCINATION_RISK,
                    severity=IssueSeverity.WARNING,
                    message="Multiple uncertainty indicators detected",
                    suggestion="Verify facts and provide sources where possible",
                )
            )

        # Check for unsupported claims
        for pattern in self.UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pattern, output_lower):
                issues.append(
                    ValidationIssue(
                        type=IssueType.MISSING_SOURCES,
                        severity=IssueSeverity.INFO,
                        message=f"Claim may need citation: '{pattern}'",
                        suggestion="Add source references for claims",
                    )
                )

        # Check task relevance if task provided
        if task:
            task_keywords = set(task.lower().split())
            output_keywords = set(output_lower.split())
            overlap = task_keywords.intersection(output_keywords)

            if len(overlap) < len(task_keywords) * 0.2:
                issues.append(
                    ValidationIssue(
                        type=IssueType.INCONSISTENT,
                        severity=IssueSeverity.WARNING,
                        message="Output may not fully address the task",
                        suggestion="Ensure response addresses all aspects of the task",
                    )
                )

        return issues

    def _calculate_confidence(self, issues: List[ValidationIssue]) -> float:
        """Calculate confidence score based on issues."""
        if not issues:
            return 0.95

        # Start with high confidence and deduct based on issues
        confidence = 1.0

        severity_penalties = {
            IssueSeverity.INFO: 0.02,
            IssueSeverity.WARNING: 0.1,
            IssueSeverity.ERROR: 0.25,
            IssueSeverity.CRITICAL: 0.6,  # Critical issues severely impact confidence
        }

        for issue in issues:
            confidence -= severity_penalties.get(issue.severity, 0.05)

        return max(0.0, min(1.0, confidence))

    def _generate_suggestions(
        self, issues: List[ValidationIssue], output: str
    ) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []

        # Collect suggestions from issues
        for issue in issues:
            if issue.suggestion and issue.suggestion not in suggestions:
                suggestions.append(issue.suggestion)

        # Add general suggestions based on issue types
        issue_types = {i.type for i in issues}

        if IssueType.INCOMPLETE in issue_types:
            suggestions.append("Consider providing more comprehensive information")

        if IssueType.MISSING_SOURCES in issue_types:
            suggestions.append("Add citations or references to support claims")

        if IssueType.HALLUCINATION_RISK in issue_types:
            suggestions.append("Verify uncertain information before including")

        return suggestions[:5]  # Limit suggestions

    def _parse_critique(
        self, critique_response: str
    ) -> tuple[List[ValidationIssue], List[str]]:
        """Parse LLM critique response."""
        issues = []
        suggestions = []

        # Try to extract JSON
        try:
            json_match = re.search(r"\{[\s\S]*\}", critique_response)
            if json_match:
                data = json.loads(json_match.group())

                # Extract issues from each criterion
                for criterion in ["completeness", "accuracy", "consistency", "relevance", "sources"]:
                    if criterion in data:
                        crit_data = data[criterion]
                        score = crit_data.get("score", 5)
                        crit_issues = crit_data.get("issues", [])

                        if score <= 2:
                            severity = IssueSeverity.ERROR
                        elif score <= 3:
                            severity = IssueSeverity.WARNING
                        else:
                            severity = IssueSeverity.INFO

                        for issue_text in crit_issues:
                            issues.append(
                                ValidationIssue(
                                    type=IssueType.POTENTIALLY_INCORRECT,
                                    severity=severity,
                                    message=f"[{criterion}] {issue_text}",
                                )
                            )

                # Extract suggestions
                suggestions = data.get("suggestions", [])

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, do basic text extraction
            if "issue" in critique_response.lower():
                issues.append(
                    ValidationIssue(
                        type=IssueType.POTENTIALLY_INCORRECT,
                        severity=IssueSeverity.WARNING,
                        message="LLM identified potential issues (details unclear)",
                    )
                )

        return issues, suggestions

    def quick_check(self, output: str) -> bool:
        """Quick validity check without detailed analysis."""
        if not output or not output.strip():
            return False
        if len(output.strip()) < 5:
            return False
        if output.lower().startswith("error"):
            return False
        return True
