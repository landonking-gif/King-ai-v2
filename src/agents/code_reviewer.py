"""
Code Reviewer Agent - Automated code review and suggestions.
Analyzes code for quality, security, and best practices.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from src.agents.base import SubAgent
from src.utils.code_analyzer import CodeAnalyzer
from src.agents.code_templates import Language
from src.utils.metrics import TASKS_EXECUTED


class IssueSeverity(str, Enum):
    """Severity levels for code issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueCategory(str, Enum):
    """Categories of code issues."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    CORRECTNESS = "correctness"
    STYLE = "style"
    DOCUMENTATION = "documentation"


@dataclass
class CodeIssue:
    """A code issue found during review."""
    line: Optional[int]
    category: IssueCategory
    severity: IssueSeverity
    message: str
    suggestion: str
    code_snippet: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet
        }


@dataclass
class CodeReview:
    """Complete code review result."""
    file_path: Optional[str]
    language: Language
    issues: List[CodeIssue]
    overall_score: float  # 0-100
    summary: str
    recommendations: List[str]
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language.value,
            "issues": [i.to_dict() for i in self.issues],
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class CodeReviewerAgent(SubAgent):
    """
    Agent for automated code review.
    Identifies issues and provides actionable suggestions.
    """
    
    name = "code_reviewer"
    description = "Automated code review and quality analysis"
    
    REVIEW_PROMPT = """Review the following {language} code for issues and improvements.

Code:
```{language}
{code}
```

Analyze for:
1. Security vulnerabilities (injection, exposure, etc.)
2. Performance issues (complexity, memory leaks)
3. Maintainability (complexity, coupling)
4. Correctness (bugs, edge cases)
5. Style (naming, formatting)
6. Documentation (missing docs, unclear code)

For each issue found, provide:
- Line number (if applicable)
- Category (security/performance/maintainability/correctness/style/documentation)
- Severity (critical/high/medium/low/info)
- Description of the issue
- Suggested fix

Format each issue as:
ISSUE:
Line: [number or "N/A"]
Category: [category]
Severity: [severity]
Message: [description]
Suggestion: [how to fix]
---

At the end, provide:
SUMMARY: [overall assessment]
SCORE: [0-100]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
"""
    
    # Common security patterns to check
    SECURITY_PATTERNS = {
        Language.PYTHON: [
            (r'eval\(', "Use of eval() - potential code injection", IssueSeverity.CRITICAL),
            (r'exec\(', "Use of exec() - potential code injection", IssueSeverity.CRITICAL),
            (r'pickle\.loads?\(', "Pickle deserialization - potential security risk", IssueSeverity.HIGH),
            (r'subprocess\.(call|run|Popen).*shell\s*=\s*True', "Shell=True in subprocess - command injection risk", IssueSeverity.HIGH),
            (r'password\s*=\s*["\']', "Hardcoded password detected", IssueSeverity.CRITICAL),
            (r'api_key\s*=\s*["\']', "Hardcoded API key detected", IssueSeverity.CRITICAL),
        ],
        Language.JAVASCRIPT: [
            (r'eval\(', "Use of eval() - potential code injection", IssueSeverity.CRITICAL),
            (r'innerHTML\s*=', "innerHTML assignment - potential XSS", IssueSeverity.HIGH),
            (r'document\.write\(', "document.write() - potential XSS", IssueSeverity.MEDIUM),
        ]
    }
    
    def __init__(self):
        """Initialize code reviewer."""
        super().__init__()
        self.analyzer = CodeAnalyzer()
    
    async def execute(self, task: dict) -> dict:
        """Execute a code review task."""
        try:
            input_data = task.get("input", {})
            code = input_data.get("code", task.get("code", ""))
            language = Language(input_data.get("language", task.get("language", "python")))
            file_path = input_data.get("file_path", task.get("file_path"))
            
            review = await self.review(code, language, file_path)
            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            
            return {
                "success": True,
                "output": review.to_dict(),
                "metadata": {"type": "code_review", "issues": len(review.issues)}
            }
            
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {
                "success": False,
                "error": str(e),
                "metadata": {"type": "code_review"}
            }
    
    async def review(
        self,
        code: str,
        language: Language,
        file_path: str = None
    ) -> CodeReview:
        """
        Perform comprehensive code review.
        
        Args:
            code: Code to review
            language: Programming language
            file_path: Optional file path for context
            
        Returns:
            Complete code review
        """
        issues: List[CodeIssue] = []
        
        # Static pattern checks
        pattern_issues = self._check_patterns(code, language)
        issues.extend(pattern_issues)
        
        # LLM-based review
        llm_issues, summary, score, recommendations = await self._llm_review(code, language)
        issues.extend(llm_issues)
        
        # Dedup and sort issues
        issues = self._deduplicate_issues(issues)
        issues.sort(key=lambda i: (
            list(IssueSeverity).index(i.severity),
            i.line or 0
        ))
        
        return CodeReview(
            file_path=file_path,
            language=language,
            issues=issues,
            overall_score=score,
            summary=summary,
            recommendations=recommendations
        )
    
    def _check_patterns(
        self,
        code: str,
        language: Language
    ) -> List[CodeIssue]:
        """Check code against known security patterns."""
        import re
        
        issues = []
        patterns = self.SECURITY_PATTERNS.get(language, [])
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, message, severity in patterns:
                if re.search(pattern, line):
                    issues.append(CodeIssue(
                        line=i,
                        category=IssueCategory.SECURITY,
                        severity=severity,
                        message=message,
                        suggestion="Review and address this security concern",
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    async def _llm_review(
        self,
        code: str,
        language: Language
    ) -> tuple:
        """Perform LLM-based code review."""
        prompt = self.REVIEW_PROMPT.format(
            language=language.value,
            code=code[:8000]  # Limit code size
        )
        
        response = await self._ask_llm(prompt)
        
        # Parse issues
        issues = self._parse_issues(response)
        
        # Parse summary and score
        summary = self._extract_summary(response)
        score = self._extract_score(response)
        recommendations = self._extract_recommendations(response)
        
        return issues, summary, score, recommendations
    
    def _parse_issues(self, response: str) -> List[CodeIssue]:
        """Parse issues from LLM response."""
        issues = []
        
        issue_blocks = response.split('ISSUE:')[1:] if 'ISSUE:' in response else []
        
        for block in issue_blocks:
            if '---' in block:
                block = block.split('---')[0]
            
            issue = self._parse_issue_block(block)
            if issue:
                issues.append(issue)
        
        return issues
    
    def _parse_issue_block(self, block: str) -> Optional[CodeIssue]:
        """Parse a single issue block."""
        try:
            lines_dict = {}
            current_key = None
            
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    lines_dict[key] = value.strip()
                    current_key = key
                elif current_key and line.strip():
                    lines_dict[current_key] += ' ' + line.strip()
            
            # Parse line number
            line_num = None
            if 'line' in lines_dict:
                try:
                    line_num = int(lines_dict['line'])
                except ValueError:
                    pass
            
            # Parse category
            category = IssueCategory.MAINTAINABILITY
            if 'category' in lines_dict:
                cat_str = lines_dict['category'].lower()
                for cat in IssueCategory:
                    if cat.value in cat_str:
                        category = cat
                        break
            
            # Parse severity
            severity = IssueSeverity.MEDIUM
            if 'severity' in lines_dict:
                sev_str = lines_dict['severity'].lower()
                for sev in IssueSeverity:
                    if sev.value in sev_str:
                        severity = sev
                        break
            
            return CodeIssue(
                line=line_num,
                category=category,
                severity=severity,
                message=lines_dict.get('message', 'Issue detected'),
                suggestion=lines_dict.get('suggestion', 'Review this code')
            )
            
        except Exception:
            return None
    
    def _extract_summary(self, response: str) -> str:
        """Extract summary from response."""
        if 'SUMMARY:' in response:
            parts = response.split('SUMMARY:')[1]
            if 'SCORE:' in parts:
                parts = parts.split('SCORE:')[0]
            return parts.strip()[:500]
        return "Code review completed"
    
    def _extract_score(self, response: str) -> float:
        """Extract score from response."""
        import re
        
        if 'SCORE:' in response:
            score_part = response.split('SCORE:')[1][:20]
            match = re.search(r'(\d+)', score_part)
            if match:
                return min(100, max(0, float(match.group(1))))
        
        return 50.0  # Default score
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response."""
        recommendations = []
        
        if 'RECOMMENDATIONS:' in response:
            rec_part = response.split('RECOMMENDATIONS:')[1]
            for line in rec_part.split('\n'):
                line = line.strip().lstrip('- •*')
                if line and len(line) > 10:
                    recommendations.append(line)
                if len(recommendations) >= 5:
                    break
        
        return recommendations
    
    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues."""
        seen = set()
        unique = []
        
        for issue in issues:
            key = (issue.line, issue.message[:50])
            if key not in seen:
                seen.add(key)
                unique.append(issue)
        
        return unique
