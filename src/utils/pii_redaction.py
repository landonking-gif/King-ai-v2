"""
PII Redaction System.

Auto-detect and redact personally identifiable information.
Based on mother-harness security patterns.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib

from src.utils.structured_logging import get_logger

logger = get_logger("pii_redaction")


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    AWS_KEY = "aws_key"
    BANK_ACCOUNT = "bank_account"
    PASSPORT = "passport"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII instance."""
    pii_type: PIIType
    original: str
    redacted: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class RedactionResult:
    """Result of redacting text."""
    original: str
    redacted: str
    matches: List[PIIMatch]
    pii_found: bool
    
    @property
    def pii_count(self) -> int:
        return len(self.matches)
    
    @property
    def pii_types(self) -> Set[PIIType]:
        return {m.pii_type for m in self.matches}


class PIIPattern:
    """A pattern for detecting PII."""
    
    def __init__(
        self,
        pii_type: PIIType,
        pattern: str,
        replacement: str = None,
        flags: int = re.IGNORECASE,
        validator: Callable[[str], bool] = None,
        confidence: float = 1.0,
    ):
        self.pii_type = pii_type
        self.pattern = re.compile(pattern, flags)
        self.replacement = replacement or f"[REDACTED_{pii_type.value.upper()}]"
        self.validator = validator
        self.confidence = confidence
    
    def find_matches(self, text: str) -> List[PIIMatch]:
        """Find all matches in text."""
        matches = []
        for match in self.pattern.finditer(text):
            original = match.group()
            
            # Validate if validator provided
            if self.validator and not self.validator(original):
                continue
            
            matches.append(PIIMatch(
                pii_type=self.pii_type,
                original=original,
                redacted=self.replacement,
                start=match.start(),
                end=match.end(),
                confidence=self.confidence,
            ))
        
        return matches


def luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    # Luhn algorithm
    checksum = 0
    num_digits = len(digits)
    parity = num_digits % 2
    
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit = digit * 2
            if digit > 9:
                digit = digit - 9
        checksum += digit
    
    return checksum % 10 == 0


def validate_ssn(ssn: str) -> bool:
    """Validate SSN format."""
    # Remove separators
    clean = re.sub(r'[-\s]', '', ssn)
    if len(clean) != 9:
        return False
    
    # Check for invalid patterns
    if clean.startswith('000') or clean.startswith('666'):
        return False
    if clean[3:5] == '00' or clean[5:] == '0000':
        return False
    
    return True


class PIIRedactor:
    """
    Redact personally identifiable information from text.
    
    Features:
    - Multiple PII type detection
    - Pattern-based matching with validators
    - Reversible redaction with tokens
    - Batch processing
    - Audit logging
    """
    
    # Default patterns
    DEFAULT_PATTERNS = [
        # Email
        PIIPattern(
            PIIType.EMAIL,
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ),
        
        # Phone (various formats)
        PIIPattern(
            PIIType.PHONE,
            r'\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        ),
        
        # SSN
        PIIPattern(
            PIIType.SSN,
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            validator=validate_ssn,
        ),
        
        # Credit Card (major issuers)
        PIIPattern(
            PIIType.CREDIT_CARD,
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
            validator=luhn_check,
        ),
        
        # Credit Card with spaces/dashes
        PIIPattern(
            PIIType.CREDIT_CARD,
            r'\b(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b',
        ),
        
        # IP Address
        PIIPattern(
            PIIType.IP_ADDRESS,
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            confidence=0.8,  # Lower confidence, IPs aren't always PII
        ),
        
        # API Keys (common patterns)
        PIIPattern(
            PIIType.API_KEY,
            r'(?:api[_-]?key|apikey|api_secret|secret_key|access_token|auth_token)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?',
            flags=re.IGNORECASE,
        ),
        
        # AWS Access Key
        PIIPattern(
            PIIType.AWS_KEY,
            r'(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}',
        ),
        
        # Password in connection strings or config
        PIIPattern(
            PIIType.PASSWORD,
            r'(?:password|passwd|pwd)["\s:=]+["\']?([^\s"\'<>&]{4,})["\']?',
            flags=re.IGNORECASE,
        ),
        
        # Bank account (basic)
        PIIPattern(
            PIIType.BANK_ACCOUNT,
            r'\b[0-9]{8,17}\b',
            confidence=0.5,  # Low confidence, needs context
        ),
    ]
    
    def __init__(
        self,
        patterns: List[PIIPattern] = None,
        enabled_types: Set[PIIType] = None,
        hash_originals: bool = True,
        audit_callback: Callable[[RedactionResult], None] = None,
    ):
        """
        Initialize the redactor.
        
        Args:
            patterns: Custom patterns (uses defaults if None)
            enabled_types: Only detect these types (all if None)
            hash_originals: Store hashes of originals for reversibility
            audit_callback: Called after each redaction with results
        """
        self._patterns = patterns or self.DEFAULT_PATTERNS
        self._enabled_types = enabled_types
        self._hash_originals = hash_originals
        self._audit_callback = audit_callback
        
        # Lookup table for reversible redaction
        self._hash_to_original: Dict[str, str] = {}
        
        # Statistics
        self._stats = {
            "texts_processed": 0,
            "pii_detected": 0,
            "by_type": {t.value: 0 for t in PIIType},
        }
    
    def redact(
        self,
        text: str,
        min_confidence: float = 0.6,
    ) -> RedactionResult:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            min_confidence: Minimum confidence threshold
            
        Returns:
            Redaction result with redacted text and matches
        """
        if not text:
            return RedactionResult(
                original=text,
                redacted=text,
                matches=[],
                pii_found=False,
            )
        
        all_matches: List[PIIMatch] = []
        
        # Find all matches
        for pattern in self._patterns:
            if self._enabled_types and pattern.pii_type not in self._enabled_types:
                continue
            
            matches = pattern.find_matches(text)
            for match in matches:
                if match.confidence >= min_confidence:
                    all_matches.append(match)
        
        # Sort by position (reverse order for replacement)
        all_matches.sort(key=lambda m: m.start, reverse=True)
        
        # Remove overlapping matches (keep first/highest priority)
        filtered_matches = []
        covered_ranges = []
        for match in all_matches:
            overlaps = False
            for start, end in covered_ranges:
                if not (match.end <= start or match.start >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_matches.append(match)
                covered_ranges.append((match.start, match.end))
        
        # Apply redactions
        redacted = text
        for match in filtered_matches:
            # Generate reversible token if enabled
            if self._hash_originals:
                token_hash = hashlib.sha256(match.original.encode()).hexdigest()[:8]
                redacted_value = f"[REDACTED_{match.pii_type.value.upper()}_{token_hash}]"
                self._hash_to_original[token_hash] = match.original
                match.redacted = redacted_value
            
            redacted = redacted[:match.start] + match.redacted + redacted[match.end:]
        
        # Reverse the match list to be in order
        filtered_matches.reverse()
        
        # Update stats
        self._stats["texts_processed"] += 1
        self._stats["pii_detected"] += len(filtered_matches)
        for match in filtered_matches:
            self._stats["by_type"][match.pii_type.value] += 1
        
        result = RedactionResult(
            original=text,
            redacted=redacted,
            matches=filtered_matches,
            pii_found=len(filtered_matches) > 0,
        )
        
        # Audit callback
        if self._audit_callback and result.pii_found:
            try:
                self._audit_callback(result)
            except Exception as e:
                logger.warning(f"Audit callback failed: {e}")
        
        return result
    
    def redact_dict(
        self,
        data: Dict[str, Any],
        keys_to_skip: Set[str] = None,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Redact PII from all string values in a dictionary.
        
        Args:
            data: Dictionary to redact
            keys_to_skip: Keys to not redact (e.g., 'id', 'type')
            min_confidence: Minimum confidence threshold
            
        Returns:
            New dictionary with redacted values
        """
        keys_to_skip = keys_to_skip or {'id', 'type', 'created_at', 'updated_at'}
        
        return self._redact_value(data, keys_to_skip, min_confidence)
    
    def _redact_value(
        self,
        value: Any,
        keys_to_skip: Set[str],
        min_confidence: float,
    ) -> Any:
        """Recursively redact values."""
        if isinstance(value, str):
            result = self.redact(value, min_confidence)
            return result.redacted
        elif isinstance(value, dict):
            return {
                k: (v if k in keys_to_skip else self._redact_value(v, keys_to_skip, min_confidence))
                for k, v in value.items()
            }
        elif isinstance(value, list):
            return [self._redact_value(item, keys_to_skip, min_confidence) for item in value]
        else:
            return value
    
    def restore(self, redacted_text: str) -> str:
        """
        Restore original values from redacted text using stored hashes.
        
        Args:
            redacted_text: Text with redaction tokens
            
        Returns:
            Text with original values restored
        """
        if not self._hash_originals:
            raise ValueError("Restoration not enabled (hash_originals=False)")
        
        restored = redacted_text
        
        # Find all redaction tokens
        pattern = re.compile(r'\[REDACTED_[A-Z_]+_([a-f0-9]{8})\]')
        for match in pattern.finditer(redacted_text):
            token_hash = match.group(1)
            original = self._hash_to_original.get(token_hash)
            if original:
                restored = restored.replace(match.group(0), original, 1)
        
        return restored
    
    def add_pattern(self, pattern: PIIPattern) -> None:
        """Add a custom pattern."""
        self._patterns.append(pattern)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get redaction statistics."""
        return dict(self._stats)
    
    def clear_lookup(self) -> int:
        """Clear the hash-to-original lookup table. Returns count cleared."""
        count = len(self._hash_to_original)
        self._hash_to_original.clear()
        return count


# Convenience functions
def redact_text(text: str) -> str:
    """Quick redact without tracking."""
    redactor = PIIRedactor(hash_originals=False)
    return redactor.redact(text).redacted


def contains_pii(text: str) -> bool:
    """Check if text contains PII."""
    redactor = PIIRedactor(hash_originals=False)
    result = redactor.redact(text)
    return result.pii_found


def get_pii_types(text: str) -> Set[PIIType]:
    """Get types of PII found in text."""
    redactor = PIIRedactor(hash_originals=False)
    result = redactor.redact(text)
    return result.pii_types


# Global redactor instance
_pii_redactor: Optional[PIIRedactor] = None


def get_pii_redactor() -> PIIRedactor:
    """Get or create the global PII redactor."""
    global _pii_redactor
    if _pii_redactor is None:
        _pii_redactor = PIIRedactor()
    return _pii_redactor
