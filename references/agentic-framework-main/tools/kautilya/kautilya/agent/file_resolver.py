"""
Intelligent File Resolver for Kautilya.

Provides smart file path resolution with:
- Exact path matching
- Relative path resolution
- Common location search
- Glob pattern matching
- Fuzzy name matching
- Helpful suggestions when not found
"""

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FileMatch:
    """Represents a matched file with metadata."""

    path: Path
    confidence: float  # 0.0 - 1.0
    match_type: str  # "exact", "relative", "search", "glob", "fuzzy"
    original_reference: str = ""

    def __str__(self) -> str:
        return f"{self.path} ({self.match_type}, {self.confidence:.0%})"


@dataclass
class FileNotFoundResult:
    """Result when file is not found, with suggestions."""

    reference: str
    searched_paths: List[str]
    suggestions: List[str] = field(default_factory=list)
    error_message: str = ""

    def format_error(self) -> str:
        """Format a helpful error message."""
        lines = [f"File not found: {self.reference}"]
        lines.append("")
        lines.append("Searched in:")
        for path in self.searched_paths[:5]:
            lines.append(f"  - {path}")
        if self.suggestions:
            lines.append("")
            lines.append("Did you mean:")
            for suggestion in self.suggestions[:5]:
                lines.append(f"  - {suggestion}")
        return "\n".join(lines)


class FileResolver:
    """
    Intelligent file path resolver.

    Resolves file references using multiple strategies:
    1. Exact path match
    2. Relative to context directory
    3. Search in common locations
    4. Glob pattern matching
    5. Fuzzy name matching

    Provides helpful suggestions when files aren't found.
    """

    # Common locations to search for files
    DEFAULT_SEARCH_PATHS = [
        ".",
        "./reports",
        "./documents",
        "./docs",
        "./data",
        "./files",
        "./samples",
        "./examples",
        "./input",
        "./output",
        "./assets",
        "~/Documents",
        "~/Downloads",
        "~/Desktop",
    ]

    # Document file extensions
    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
        ".csv",
        ".html",
        ".htm",
        ".txt",
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
    }

    # Code file extensions
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
    }

    # All searchable extensions
    SEARCHABLE_EXTENSIONS = DOCUMENT_EXTENSIONS | CODE_EXTENSIONS

    def __init__(
        self,
        search_paths: Optional[List[str]] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize file resolver.

        Args:
            search_paths: Custom search paths to use
            project_root: Project root directory (detected if not provided)
        """
        self.search_paths = search_paths or self.DEFAULT_SEARCH_PATHS.copy()
        self.project_root = project_root or self._detect_project_root()

        # Add project root to search paths if detected
        if self.project_root and str(self.project_root) not in self.search_paths:
            self.search_paths.insert(0, str(self.project_root))

    def _detect_project_root(self) -> Optional[Path]:
        """Detect project root by looking for markers."""
        current = Path.cwd()

        markers = [".git", "pyproject.toml", "setup.py", "package.json", "CLAUDE.md"]

        for _ in range(10):
            for marker in markers:
                if (current / marker).exists():
                    return current

            parent = current.parent
            if parent == current:
                break
            current = parent

        return Path.cwd()

    def resolve(
        self,
        reference: str,
        context_dir: Optional[Path] = None,
        file_type: Optional[str] = None,
    ) -> FileMatch:
        """
        Resolve a file reference to an actual path.

        Args:
            reference: File path or reference (can include @)
            context_dir: Directory context for relative paths
            file_type: Expected file type ("document", "code", or None for any)

        Returns:
            FileMatch with resolved path

        Raises:
            FileNotFoundError: If file cannot be found (with suggestions)
        """
        # Clean up reference
        reference = reference.lstrip("@").strip("\"'").strip()
        original_ref = reference

        # Strategy 1: Exact path
        exact_path = Path(reference).expanduser()
        if exact_path.is_absolute() and exact_path.exists():
            logger.debug(f"Exact match: {exact_path}")
            return FileMatch(exact_path, 1.0, "exact", original_ref)

        # Strategy 2: Relative to CWD
        cwd_path = Path.cwd() / reference
        if cwd_path.exists():
            logger.debug(f"CWD relative match: {cwd_path}")
            return FileMatch(cwd_path.resolve(), 0.98, "relative", original_ref)

        # Strategy 3: Relative to context directory
        if context_dir:
            context_path = Path(context_dir) / reference
            if context_path.exists():
                logger.debug(f"Context relative match: {context_path}")
                return FileMatch(context_path.resolve(), 0.95, "relative", original_ref)

        # Strategy 4: Relative to project root
        if self.project_root:
            project_path = self.project_root / reference
            if project_path.exists():
                logger.debug(f"Project relative match: {project_path}")
                return FileMatch(project_path.resolve(), 0.93, "relative", original_ref)

        # Strategy 5: Search common locations
        for search_path in self.search_paths:
            base = Path(search_path).expanduser()
            if not base.exists():
                continue

            candidate = base / reference
            if candidate.exists():
                logger.debug(f"Search path match: {candidate}")
                return FileMatch(candidate.resolve(), 0.90, "search", original_ref)

        # Strategy 6: Glob pattern matching
        filename = Path(reference).name
        glob_match = self._glob_search(filename, file_type)
        if glob_match:
            logger.debug(f"Glob match: {glob_match}")
            return FileMatch(glob_match, 0.80, "glob", original_ref)

        # Strategy 7: Fuzzy matching
        fuzzy_match = self._fuzzy_search(filename, file_type)
        if fuzzy_match:
            logger.debug(f"Fuzzy match: {fuzzy_match}")
            return FileMatch(fuzzy_match, 0.60, "fuzzy", original_ref)

        # Not found - build helpful error
        suggestions = self._get_suggestions(filename, file_type)
        searched = [str(Path(p).expanduser()) for p in self.search_paths if Path(p).expanduser().exists()]

        result = FileNotFoundResult(
            reference=original_ref,
            searched_paths=searched,
            suggestions=suggestions,
        )

        raise FileNotFoundError(result.format_error())

    def resolve_multiple(
        self,
        references: List[str],
        context_dir: Optional[Path] = None,
    ) -> Tuple[List[FileMatch], List[FileNotFoundResult]]:
        """
        Resolve multiple file references.

        Args:
            references: List of file paths/references
            context_dir: Directory context for relative paths

        Returns:
            Tuple of (successful matches, failed references)
        """
        matches = []
        failures = []

        for ref in references:
            try:
                match = self.resolve(ref, context_dir)
                matches.append(match)
            except FileNotFoundError as e:
                failures.append(
                    FileNotFoundResult(
                        reference=ref,
                        searched_paths=[],
                        error_message=str(e),
                    )
                )

        return matches, failures

    def _glob_search(
        self,
        filename: str,
        file_type: Optional[str] = None,
    ) -> Optional[Path]:
        """Search for file using glob patterns."""
        # Determine extensions to search
        if file_type == "document":
            extensions = self.DOCUMENT_EXTENSIONS
        elif file_type == "code":
            extensions = self.CODE_EXTENSIONS
        else:
            extensions = self.SEARCHABLE_EXTENSIONS

        # Build search patterns
        patterns = [
            f"**/{filename}",  # Exact filename
            f"**/*{filename}*",  # Contains filename
        ]

        # Add extension-specific patterns if no extension in filename
        if not Path(filename).suffix:
            for ext in extensions:
                patterns.append(f"**/{filename}{ext}")
                patterns.append(f"**/*{filename}*{ext}")

        # Search in each path
        all_matches = []
        for search_path in self.search_paths:
            base = Path(search_path).expanduser()
            if not base.exists():
                continue

            for pattern in patterns:
                try:
                    matches = list(base.glob(pattern))
                    for match in matches:
                        if match.is_file():
                            all_matches.append(match)
                except Exception:
                    continue

        if not all_matches:
            return None

        # Score and rank matches
        def score_match(path: Path) -> Tuple[int, int, int]:
            name = path.name.lower()
            target = filename.lower()
            return (
                0 if name == target else 1,  # Exact name match first
                0 if target in name else 1,  # Contains target
                len(str(path)),  # Shorter paths preferred
            )

        all_matches.sort(key=score_match)
        return all_matches[0].resolve()

    def _fuzzy_search(
        self,
        filename: str,
        file_type: Optional[str] = None,
    ) -> Optional[Path]:
        """Search for file using fuzzy matching."""
        if file_type == "document":
            extensions = self.DOCUMENT_EXTENSIONS
        elif file_type == "code":
            extensions = self.CODE_EXTENSIONS
        else:
            extensions = self.SEARCHABLE_EXTENSIONS

        best_match = None
        best_ratio = 0.7  # Minimum similarity threshold (stricter to avoid false matches)

        for search_path in self.search_paths:
            base = Path(search_path).expanduser()
            if not base.exists():
                continue

            try:
                for file_path in base.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if file_path.suffix.lower() not in extensions:
                        continue

                    # Compare filenames
                    ratio = SequenceMatcher(
                        None,
                        filename.lower(),
                        file_path.name.lower(),
                    ).ratio()

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = file_path

            except PermissionError:
                continue

        return best_match.resolve() if best_match else None

    def _get_suggestions(
        self,
        filename: str,
        file_type: Optional[str] = None,
        max_suggestions: int = 5,
    ) -> List[str]:
        """Get suggestions for similar files."""
        if file_type == "document":
            extensions = self.DOCUMENT_EXTENSIONS
        elif file_type == "code":
            extensions = self.CODE_EXTENSIONS
        else:
            extensions = self.SEARCHABLE_EXTENSIONS

        candidates = []

        for search_path in self.search_paths:
            base = Path(search_path).expanduser()
            if not base.exists():
                continue

            try:
                for file_path in base.rglob("*"):
                    if not file_path.is_file():
                        continue

                    if file_path.suffix.lower() not in extensions:
                        continue

                    # Calculate similarity
                    ratio = SequenceMatcher(
                        None,
                        filename.lower(),
                        file_path.name.lower(),
                    ).ratio()

                    if ratio > 0.3:  # Lower threshold for suggestions
                        candidates.append((ratio, str(file_path)))

            except PermissionError:
                continue

        # Sort by similarity and return top suggestions
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [path for _, path in candidates[:max_suggestions]]

    def add_search_path(self, path: str) -> None:
        """Add a search path dynamically."""
        if path not in self.search_paths:
            self.search_paths.append(path)

    def extract_file_references(self, text: str) -> List[str]:
        """
        Extract @file references from text.

        Args:
            text: Text potentially containing @file references

        Returns:
            List of file references (without @)
        """
        import re

        # Pattern: @path or @"path with spaces"
        pattern = r'@"([^"]+)"|@(\S+)'
        references = []

        for match in re.finditer(pattern, text):
            ref = match.group(1) or match.group(2)
            # Filter out non-file references (like @mentions)
            if ref and (
                "/" in ref
                or "\\" in ref
                or "." in ref
                or ref.endswith(tuple(self.SEARCHABLE_EXTENSIONS))
            ):
                references.append(ref)

        return references
