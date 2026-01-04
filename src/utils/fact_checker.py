"""
Fact-checking and hallucination detection utilities.

This module provides tools to validate LLM outputs against known facts
and detect potential hallucinations.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.utils.structured_logging import get_logger

logger = get_logger("fact_checker")


@dataclass
class FactCheckResult:
    """Result of fact-checking validation."""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    issues: List[str]
    warnings: List[str]


class FactChecker:
    """Validates LLM outputs for factual accuracy and hallucination detection."""
    
    # Patterns that indicate potential hallucination
    HALLUCINATION_PATTERNS = [
        r"I (recall|remember|know) that",
        r"based on my (knowledge|understanding|memory)",
        r"I've (seen|heard|learned) that",
        r"as far as I (know|understand)",
        r"I believe",
        r"in my (experience|opinion)",
    ]
    
    # Phrases that indicate uncertainty (good - AI acknowledging limits)
    UNCERTAINTY_PHRASES = [
        "I don't have",
        "I cannot confirm",
        "I'm not sure",
        "I don't know",
        "unclear",
        "uncertain",
        "possibly",
        "might be",
        "could be",
        "appears to",
        "seems to",
    ]
    
    # Phrases that indicate grounding (good - referencing provided data)
    GROUNDING_PHRASES = [
        "based on the provided",
        "according to the data",
        "from the context",
        "the analytics show",
        "the report indicates",
        "the database shows",
        "in the system",
    ]
    
    def __init__(self):
        """Initialize fact checker."""
        self.hallucination_regex = re.compile(
            "|".join(self.HALLUCINATION_PATTERNS),
            re.IGNORECASE
        )
    
    def check_response(
        self,
        response: str,
        context: Optional[str] = None,
        expected_data: Optional[Dict] = None
    ) -> FactCheckResult:
        """
        Check an LLM response for factual accuracy and hallucination.
        
        Args:
            response: The LLM's response text
            context: The context that was provided to the LLM
            expected_data: Optional dict of expected factual data
            
        Returns:
            FactCheckResult with validation details
        """
        issues = []
        warnings = []
        confidence = 1.0
        
        # Check for hallucination patterns
        hallucination_matches = self.hallucination_regex.findall(response)
        if hallucination_matches:
            issues.append(
                f"Detected {len(hallucination_matches)} potential hallucination patterns: "
                f"{', '.join(set(hallucination_matches))}"
            )
            confidence -= 0.3
        
        # Check for ungrounded claims (numbers without context)
        if self._contains_ungrounded_numbers(response, context):
            warnings.append("Response contains numbers that may not be from provided context")
            confidence -= 0.1
        
        # Check if response acknowledges uncertainty when appropriate
        has_uncertainty = any(
            phrase.lower() in response.lower()
            for phrase in self.UNCERTAINTY_PHRASES
        )
        
        has_grounding = any(
            phrase.lower() in response.lower()
            for phrase in self.GROUNDING_PHRASES
        )
        
        # If response makes definitive statements without grounding, warn
        if not has_grounding and not has_uncertainty:
            if self._makes_factual_claims(response):
                warnings.append(
                    "Response makes factual claims without citing sources or context"
                )
                confidence -= 0.15
        
        # Check for made-up URLs or references
        if self._contains_suspicious_urls(response):
            issues.append("Response contains URLs that appear to be fabricated")
            confidence -= 0.4
        
        # Check for specific data if provided
        if expected_data:
            data_issues = self._validate_against_expected(response, expected_data)
            if data_issues:
                issues.extend(data_issues)
                confidence -= 0.2 * len(data_issues)
        
        confidence = max(0.0, min(1.0, confidence))
        is_valid = len(issues) == 0 and confidence >= 0.7
        
        if not is_valid:
            logger.warning(
                "Potential hallucination detected",
                issues=issues,
                warnings=warnings,
                confidence=confidence
            )
        
        return FactCheckResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            warnings=warnings
        )
    
    def _contains_ungrounded_numbers(
        self,
        response: str,
        context: Optional[str]
    ) -> bool:
        """Check if response contains numbers not in the context."""
        if not context:
            # Can't validate without context
            return False
        
        # Extract numbers from response (dollars, percentages, counts)
        response_numbers = set(re.findall(r'\$?[\d,]+\.?\d*%?', response))
        context_numbers = set(re.findall(r'\$?[\d,]+\.?\d*%?', context))
        
        # Check if response has numbers not in context
        ungrounded = response_numbers - context_numbers
        
        # Ignore small common numbers (1, 2, 3, etc.)
        ungrounded = {n for n in ungrounded if not re.match(r'^[0-9]$', n.strip('$%,'))}
        
        return len(ungrounded) > 0
    
    def _makes_factual_claims(self, response: str) -> bool:
        """Check if response makes definitive factual statements."""
        # Look for definitive language
        definitive_patterns = [
            r'\bis\b',
            r'\bwas\b',
            r'\bwill\b',
            r'\bhas\b',
            r'\bhave\b',
            r'\bshows?\b',
            r'\bindicates?\b',
            r'\bproves?\b',
        ]
        
        count = sum(
            len(re.findall(pattern, response, re.IGNORECASE))
            for pattern in definitive_patterns
        )
        
        # If response has many definitive statements, it's making factual claims
        return count > 3
    
    def _contains_suspicious_urls(self, response: str) -> bool:
        """Check for fabricated URLs or citations."""
        # Extract URLs
        urls = re.findall(r'https?://[^\s]+', response)
        
        for url in urls:
            # Check for suspicious patterns
            if any(pattern in url.lower() for pattern in [
                'example.com',
                'test.com',
                'fake',
                'placeholder',
                'dummy'
            ]):
                return True
        
        return False
    
    def _validate_against_expected(
        self,
        response: str,
        expected_data: Dict
    ) -> List[str]:
        """Validate response against expected factual data."""
        issues = []
        
        for key, expected_value in expected_data.items():
            if expected_value is not None:
                # Check if the expected value appears in response
                value_str = str(expected_value)
                if value_str not in response:
                    # Only flag if the key topic is mentioned but value is wrong
                    if key.lower() in response.lower():
                        issues.append(
                            f"Expected '{key}' to be '{expected_value}' but not found in response"
                        )
        
        return issues


# Global instance
_fact_checker = FactChecker()


def check_for_hallucination(
    response: str,
    context: Optional[str] = None,
    expected_data: Optional[Dict] = None
) -> FactCheckResult:
    """
    Convenience function to check a response for hallucination.
    
    Args:
        response: The LLM response to check
        context: Optional context that was provided to the LLM
        expected_data: Optional expected factual data
        
    Returns:
        FactCheckResult with validation details
    """
    return _fact_checker.check_response(response, context, expected_data)
