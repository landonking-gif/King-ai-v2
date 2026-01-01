"""
Security Scanner for Code and Evolution Proposals.

Provides automated security scanning for:
- Agent-generated code
- Evolution proposals
- Integration configurations
- API calls and webhooks

Uses Bandit for Python static analysis and custom rules.
"""

import ast
import re
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.structured_logging import get_logger

logger = get_logger("security_scanner")


class SecuritySeverity(str, Enum):
    """Security issue severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(str, Enum):
    """Types of security vulnerabilities."""
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    XSS = "xss"
    HARDCODED_SECRET = "hardcoded_secret"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    INSECURE_RANDOM = "insecure_random"
    WEAK_CRYPTO = "weak_crypto"
    DEBUG_CODE = "debug_code"
    UNSAFE_EVAL = "unsafe_eval"
    SSRF = "ssrf"
    INSECURE_PERMISSIONS = "insecure_permissions"
    DEPRECATED_API = "deprecated_api"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"


@dataclass
class SecurityIssue:
    """A detected security issue."""
    id: str
    severity: SecuritySeverity
    vulnerability_type: VulnerabilityType
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    confidence: float = 1.0


@dataclass
class ScanResult:
    """Result of a security scan."""
    scan_id: str
    timestamp: datetime
    target: str
    issues: List[SecurityIssue]
    scan_duration_ms: float
    scanner_version: str = "1.0.0"
    
    @property
    def has_critical(self) -> bool:
        return any(i.severity == SecuritySeverity.CRITICAL for i in self.issues)
    
    @property
    def has_high(self) -> bool:
        return any(i.severity == SecuritySeverity.HIGH for i in self.issues)
    
    @property
    def passed(self) -> bool:
        """Check if scan passed (no critical or high issues)."""
        return not self.has_critical and not self.has_high
    
    @property
    def summary(self) -> Dict[str, int]:
        """Get issue count by severity."""
        summary = {s.value: 0 for s in SecuritySeverity}
        for issue in self.issues:
            summary[issue.severity.value] += 1
        return summary


class PatternScanner:
    """
    Pattern-based security scanner.
    
    Uses regex patterns to detect common security issues.
    """
    
    # Patterns for hardcoded secrets
    SECRET_PATTERNS = [
        (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
        (r'(?i)(api_key|apikey|api-key)\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
        (r'(?i)(secret|token)\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded secret/token"),
        (r'(?i)bearer\s+[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+', "Hardcoded bearer token"),
        (r'sk_live_[a-zA-Z0-9]{24,}', "Stripe live key"),
        (r'sk_test_[a-zA-Z0-9]{24,}', "Stripe test key"),
        (r'AKIA[0-9A-Z]{16}', "AWS access key"),
        (r'(?i)private_key\s*=\s*["\']-----BEGIN', "Hardcoded private key"),
    ]
    
    # Patterns for dangerous functions
    DANGEROUS_PATTERNS = [
        (r'\beval\s*\(', "Use of eval()", VulnerabilityType.UNSAFE_EVAL),
        (r'\bexec\s*\(', "Use of exec()", VulnerabilityType.UNSAFE_EVAL),
        (r'subprocess\..*shell\s*=\s*True', "Shell injection risk", VulnerabilityType.COMMAND_INJECTION),
        (r'os\.system\s*\(', "Command injection risk", VulnerabilityType.COMMAND_INJECTION),
        (r'pickle\.loads?\s*\(', "Insecure deserialization", VulnerabilityType.INSECURE_DESERIALIZATION),
        (r'yaml\.load\s*\([^)]*\)', "Unsafe YAML load", VulnerabilityType.INSECURE_DESERIALIZATION),
        (r'random\.(random|randint|choice)', "Weak randomness", VulnerabilityType.INSECURE_RANDOM),
        (r'(?i)debug\s*=\s*True', "Debug mode enabled", VulnerabilityType.DEBUG_CODE),
        (r'__import__\s*\(', "Dynamic import", VulnerabilityType.UNSAFE_EVAL),
        (r'open\s*\([^)]*["\']w["\']', "File write operation", VulnerabilityType.INSECURE_PERMISSIONS),
    ]
    
    # SQL injection patterns
    SQL_PATTERNS = [
        (r'execute\s*\(\s*["\'].*%s', "SQL injection via string formatting"),
        (r'execute\s*\(\s*f["\']', "SQL injection via f-string"),
        (r'execute\s*\(\s*["\'].*\+', "SQL injection via concatenation"),
        (r'cursor\.execute\s*\(\s*["\'].*\.format\s*\(', "SQL injection via format()"),
    ]
    
    def __init__(self):
        self.issue_counter = 0
    
    def _next_id(self) -> str:
        self.issue_counter += 1
        return f"PS-{self.issue_counter:04d}"
    
    def scan_code(self, code: str, file_path: str = "<string>") -> List[SecurityIssue]:
        """Scan code for security issues using patterns."""
        issues = []
        lines = code.split("\n")
        
        # Check for hardcoded secrets
        for pattern, description in self.SECRET_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    # Skip if it's in a comment
                    if line.strip().startswith("#"):
                        continue
                    # Skip if it's a placeholder
                    if any(p in line.lower() for p in ["xxxxx", "your_", "example", "placeholder"]):
                        continue
                    
                    issues.append(SecurityIssue(
                        id=self._next_id(),
                        severity=SecuritySeverity.HIGH,
                        vulnerability_type=VulnerabilityType.HARDCODED_SECRET,
                        title=description,
                        description=f"Potential {description.lower()} detected",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip()[:100],
                        recommendation="Use environment variables or a secrets manager",
                        cwe_id="CWE-798"
                    ))
        
        # Check for dangerous functions
        for pattern, description, vuln_type in self.DANGEROUS_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    severity = SecuritySeverity.MEDIUM
                    if vuln_type in [VulnerabilityType.UNSAFE_EVAL, VulnerabilityType.COMMAND_INJECTION]:
                        severity = SecuritySeverity.HIGH
                    
                    issues.append(SecurityIssue(
                        id=self._next_id(),
                        severity=severity,
                        vulnerability_type=vuln_type,
                        title=description,
                        description=f"Potentially dangerous code pattern: {description}",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip()[:100],
                        recommendation=self._get_recommendation(vuln_type)
                    ))
        
        # Check for SQL injection
        for pattern, description in self.SQL_PATTERNS:
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        id=self._next_id(),
                        severity=SecuritySeverity.HIGH,
                        vulnerability_type=VulnerabilityType.SQL_INJECTION,
                        title="SQL Injection Risk",
                        description=description,
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip()[:100],
                        recommendation="Use parameterized queries",
                        cwe_id="CWE-89"
                    ))
        
        return issues
    
    def _get_recommendation(self, vuln_type: VulnerabilityType) -> str:
        recommendations = {
            VulnerabilityType.UNSAFE_EVAL: "Avoid eval/exec. Use ast.literal_eval for safe evaluation.",
            VulnerabilityType.COMMAND_INJECTION: "Use subprocess with shell=False and argument lists.",
            VulnerabilityType.INSECURE_DESERIALIZATION: "Use safe loaders (yaml.safe_load) or JSON.",
            VulnerabilityType.INSECURE_RANDOM: "Use secrets module for cryptographic randomness.",
            VulnerabilityType.DEBUG_CODE: "Ensure debug mode is disabled in production.",
            VulnerabilityType.INSECURE_PERMISSIONS: "Validate file paths and use restrictive permissions.",
        }
        return recommendations.get(vuln_type, "Review and fix the security issue.")


class BanditScanner:
    """
    Bandit-based security scanner.
    
    Uses the Bandit static analysis tool for comprehensive Python security scanning.
    """
    
    def __init__(self):
        self._bandit_available = False
        self._check_bandit()
    
    def _check_bandit(self):
        """Check if Bandit is available."""
        try:
            result = subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self._bandit_available = result.returncode == 0
            if self._bandit_available:
                logger.info(f"Bandit available: {result.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Bandit not available. Install with: pip install bandit")
    
    @property
    def is_available(self) -> bool:
        return self._bandit_available
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a file using Bandit."""
        if not self._bandit_available:
            return []
        
        try:
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", file_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return self._parse_bandit_output(result.stdout, file_path)
        except subprocess.SubprocessError as e:
            logger.error(f"Bandit scan failed: {e}")
            return []
    
    def scan_code(self, code: str) -> List[SecurityIssue]:
        """Scan code string using Bandit."""
        if not self._bandit_available:
            return []
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            return self.scan_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _parse_bandit_output(self, output: str, file_path: str) -> List[SecurityIssue]:
        """Parse Bandit JSON output into SecurityIssues."""
        if not output.strip():
            return []
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return []
        
        issues = []
        
        for result in data.get("results", []):
            severity = self._map_severity(result.get("issue_severity", "LOW"))
            
            issues.append(SecurityIssue(
                id=f"B-{result.get('test_id', 'UNKNOWN')}",
                severity=severity,
                vulnerability_type=self._map_vulnerability_type(result.get("test_id", "")),
                title=result.get("test_name", "Unknown Issue"),
                description=result.get("issue_text", ""),
                file_path=file_path,
                line_number=result.get("line_number"),
                code_snippet=result.get("code", "").strip()[:100],
                recommendation=result.get("more_info", ""),
                confidence=self._map_confidence(result.get("issue_confidence", "LOW"))
            ))
        
        return issues
    
    def _map_severity(self, bandit_severity: str) -> SecuritySeverity:
        mapping = {
            "LOW": SecuritySeverity.LOW,
            "MEDIUM": SecuritySeverity.MEDIUM,
            "HIGH": SecuritySeverity.HIGH,
        }
        return mapping.get(bandit_severity.upper(), SecuritySeverity.MEDIUM)
    
    def _map_confidence(self, bandit_confidence: str) -> float:
        mapping = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
        return mapping.get(bandit_confidence.upper(), 0.5)
    
    def _map_vulnerability_type(self, test_id: str) -> VulnerabilityType:
        # Map Bandit test IDs to vulnerability types
        mappings = {
            "B101": VulnerabilityType.DEBUG_CODE,  # assert
            "B102": VulnerabilityType.UNSAFE_EVAL,  # exec
            "B103": VulnerabilityType.INSECURE_PERMISSIONS,  # chmod
            "B104": VulnerabilityType.SSRF,  # hardcoded bind
            "B105": VulnerabilityType.HARDCODED_SECRET,  # hardcoded password
            "B106": VulnerabilityType.HARDCODED_SECRET,  # hardcoded password arg
            "B107": VulnerabilityType.HARDCODED_SECRET,  # hardcoded password default
            "B108": VulnerabilityType.PATH_TRAVERSAL,  # hardcoded tmp
            "B110": VulnerabilityType.DEBUG_CODE,  # try except pass
            "B301": VulnerabilityType.INSECURE_DESERIALIZATION,  # pickle
            "B302": VulnerabilityType.INSECURE_DESERIALIZATION,  # marshal
            "B303": VulnerabilityType.WEAK_CRYPTO,  # md5/sha1
            "B304": VulnerabilityType.WEAK_CRYPTO,  # ciphers
            "B305": VulnerabilityType.WEAK_CRYPTO,  # cipher modes
            "B306": VulnerabilityType.PATH_TRAVERSAL,  # mktemp
            "B307": VulnerabilityType.UNSAFE_EVAL,  # eval
            "B308": VulnerabilityType.XSS,  # mark_safe
            "B310": VulnerabilityType.SSRF,  # urllib
            "B311": VulnerabilityType.INSECURE_RANDOM,  # random
            "B312": VulnerabilityType.SSRF,  # telnetlib
            "B501": VulnerabilityType.WEAK_CRYPTO,  # ssl no verify
            "B502": VulnerabilityType.WEAK_CRYPTO,  # ssl bad version
            "B503": VulnerabilityType.WEAK_CRYPTO,  # ssl bad defaults
            "B504": VulnerabilityType.WEAK_CRYPTO,  # ssl no version
            "B505": VulnerabilityType.WEAK_CRYPTO,  # weak cryptographic key
            "B506": VulnerabilityType.INSECURE_DESERIALIZATION,  # yaml load
            "B507": VulnerabilityType.WEAK_CRYPTO,  # ssh no host key
            "B601": VulnerabilityType.COMMAND_INJECTION,  # paramiko calls
            "B602": VulnerabilityType.COMMAND_INJECTION,  # subprocess popen shell
            "B603": VulnerabilityType.COMMAND_INJECTION,  # subprocess without shell
            "B604": VulnerabilityType.COMMAND_INJECTION,  # any other function shell
            "B605": VulnerabilityType.COMMAND_INJECTION,  # start process with shell
            "B606": VulnerabilityType.COMMAND_INJECTION,  # start process no shell
            "B607": VulnerabilityType.COMMAND_INJECTION,  # start process partial path
            "B608": VulnerabilityType.SQL_INJECTION,  # hardcoded SQL
            "B609": VulnerabilityType.COMMAND_INJECTION,  # wildcard injection
            "B610": VulnerabilityType.SQL_INJECTION,  # django extra
            "B611": VulnerabilityType.SQL_INJECTION,  # django raw SQL
            "B701": VulnerabilityType.XSS,  # jinja2 autoescape
            "B702": VulnerabilityType.XSS,  # mako templates
            "B703": VulnerabilityType.XSS,  # django mark_safe
        }
        return mappings.get(test_id, VulnerabilityType.DEBUG_CODE)


class SecurityScanner:
    """
    Main security scanner combining multiple scanning methods.
    
    Features:
    - Pattern-based scanning
    - Bandit static analysis
    - Evolution proposal validation
    - API configuration checking
    """
    
    def __init__(self):
        self.pattern_scanner = PatternScanner()
        self.bandit_scanner = BanditScanner()
        self.scan_count = 0
    
    def _next_scan_id(self) -> str:
        self.scan_count += 1
        return f"SCAN-{datetime.utcnow().strftime('%Y%m%d')}-{self.scan_count:04d}"
    
    async def scan_code(
        self,
        code: str,
        file_path: str = "<generated>",
        use_bandit: bool = True
    ) -> ScanResult:
        """
        Scan code for security issues.
        
        Args:
            code: Python code to scan
            file_path: Optional file path for context
            use_bandit: Whether to use Bandit scanner
            
        Returns:
            ScanResult with all detected issues
        """
        start_time = datetime.utcnow()
        issues = []
        
        # Pattern-based scan
        issues.extend(self.pattern_scanner.scan_code(code, file_path))
        
        # Bandit scan if available and requested
        if use_bandit and self.bandit_scanner.is_available:
            issues.extend(self.bandit_scanner.scan_code(code))
        
        # Deduplicate issues
        seen = set()
        unique_issues = []
        for issue in issues:
            key = (issue.line_number, issue.vulnerability_type, issue.title)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = ScanResult(
            scan_id=self._next_scan_id(),
            timestamp=datetime.utcnow(),
            target=file_path,
            issues=unique_issues,
            scan_duration_ms=duration
        )
        
        logger.info(
            "Security scan complete",
            scan_id=result.scan_id,
            issues=len(unique_issues),
            passed=result.passed
        )
        
        return result
    
    async def scan_file(self, file_path: str) -> ScanResult:
        """Scan a file for security issues."""
        with open(file_path, "r") as f:
            code = f.read()
        return await self.scan_code(code, file_path)
    
    async def scan_evolution_proposal(
        self,
        code_changes: Dict[str, str],
        proposal_id: str
    ) -> Tuple[bool, ScanResult]:
        """
        Scan an evolution proposal for security issues.
        
        Args:
            code_changes: Dict mapping file paths to new code
            proposal_id: Evolution proposal ID
            
        Returns:
            Tuple of (approved, scan_result)
        """
        all_issues = []
        
        for file_path, code in code_changes.items():
            result = await self.scan_code(code, file_path)
            all_issues.extend(result.issues)
        
        combined_result = ScanResult(
            scan_id=f"EVO-{proposal_id}",
            timestamp=datetime.utcnow(),
            target=f"evolution:{proposal_id}",
            issues=all_issues,
            scan_duration_ms=0
        )
        
        # Evolutions must pass with no high/critical issues
        approved = combined_result.passed
        
        if not approved:
            logger.warning(
                "Evolution proposal failed security scan",
                proposal_id=proposal_id,
                critical=combined_result.has_critical,
                high=combined_result.has_high
            )
        
        return approved, combined_result
    
    async def validate_api_config(
        self,
        config: Dict[str, Any]
    ) -> List[SecurityIssue]:
        """
        Validate API/integration configuration for security issues.
        
        Checks for:
        - Hardcoded credentials
        - Insecure endpoints
        - Missing HTTPS
        - Weak authentication
        """
        issues = []
        
        # Check for insecure URLs
        for key, value in config.items():
            if isinstance(value, str):
                # Check for HTTP in URLs
                if value.startswith("http://") and "localhost" not in value:
                    issues.append(SecurityIssue(
                        id=f"CFG-{len(issues)+1:04d}",
                        severity=SecuritySeverity.HIGH,
                        vulnerability_type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
                        title="Insecure HTTP URL",
                        description=f"Configuration key '{key}' uses HTTP instead of HTTPS",
                        recommendation="Use HTTPS for all external connections"
                    ))
                
                # Check for potential secrets in wrong places
                if any(s in key.lower() for s in ["key", "secret", "password", "token"]):
                    if len(value) > 10 and not value.startswith("${"):
                        issues.append(SecurityIssue(
                            id=f"CFG-{len(issues)+1:04d}",
                            severity=SecuritySeverity.MEDIUM,
                            vulnerability_type=VulnerabilityType.HARDCODED_SECRET,
                            title="Potential hardcoded secret",
                            description=f"Configuration key '{key}' may contain a hardcoded secret",
                            recommendation="Use environment variables for secrets"
                        ))
        
        return issues


# Global instance
security_scanner = SecurityScanner()


async def scan_for_approval(code: str, context: str = "approval") -> Tuple[bool, Dict[str, Any]]:
    """
    Scan code for approval workflow.
    
    Returns:
        Tuple of (passed, scan_details)
    """
    result = await security_scanner.scan_code(code, f"<{context}>")
    
    return result.passed, {
        "scan_id": result.scan_id,
        "passed": result.passed,
        "summary": result.summary,
        "issues": [
            {
                "id": i.id,
                "severity": i.severity.value,
                "title": i.title,
                "line": i.line_number,
                "recommendation": i.recommendation
            }
            for i in result.issues
        ]
    }
