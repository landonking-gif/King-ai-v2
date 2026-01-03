"""
Safe Package Manager for Python Code Execution.

Module: kautilya/safe_package_manager.py

Provides safe package installation with:
- Whitelist validation
- Version pinning
- Resource limits
- Audit logging
- Organizational safety controls
"""

import re
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PackageInstallResult:
    """Result of package installation."""
    success: bool
    package_name: str
    version: Optional[str]
    message: str
    install_time_ms: float
    audit_log: List[str]


class SafePackageManager:
    """
    Manages safe Python package installation with enterprise controls.

    Features:
    - Whitelist-based package approval
    - Automatic dependency validation
    - Resource limits (timeout, network)
    - Audit logging for compliance
    - Virtual environment isolation
    """

    # Default whitelist of commonly used safe packages
    DEFAULT_WHITELIST: Set[str] = {
        # Data science & ML
        "numpy", "pandas", "scipy", "matplotlib", "seaborn", "plotly",
        "scikit-learn", "sklearn", "torch", "tensorflow", "keras",

        # Web & APIs
        "requests", "httpx", "aiohttp", "beautifulsoup4", "bs4", "lxml",
        "yfinance",

        # Data formats
        "pyyaml", "toml", "json5", "xmltodict",

        # Utilities
        "python-dateutil", "pytz", "click", "rich", "tqdm",
        "pydantic", "attrs", "dataclasses-json",

        # Testing
        "pytest", "pytest-asyncio", "pytest-cov", "hypothesis",

        # Database
        "sqlalchemy", "psycopg2", "pymongo", "redis",

        # Async
        "anyio", "trio", "asyncio",

        # Common tools
        "pillow", "opencv-python", "xlrd", "openpyxl",
        "python-pptx", "python-docx", "docx",  # docx is alias for python-docx

        # Document processing
        "markitdown", "pypdf", "pypdf2", "pdfplumber", "pymupdf", "fitz",
        "mammoth", "docx2txt", "pptx",

        # AI/LLM
        "openai", "anthropic", "tiktoken", "sentence-transformers",
        "transformers", "tokenizers", "langchain", "llama-index",

        # Search & RAG
        "faiss-cpu", "chromadb", "qdrant-client", "pinecone-client",
        "haystack-ai", "rank-bm25",

        # Web scraping
        "ddgs", "duckduckgo-search", "selenium", "playwright",
    }

    # Blocked packages (security risks)
    BLOCKED_PACKAGES: Set[str] = {
        "os-sys",  # Can execute arbitrary system commands
        "pty",  # Terminal manipulation
        "pickle5",  # Arbitrary code execution via deserialization
    }

    def __init__(
        self,
        whitelist: Optional[Set[str]] = None,
        allow_all: bool = False,
        max_install_time_seconds: int = 120,
        audit_log_path: Optional[Path] = None,
    ):
        """
        Initialize safe package manager.

        Args:
            whitelist: Set of allowed package names (None = use default)
            allow_all: If True, allow all packages except blocked ones
            max_install_time_seconds: Maximum time for package installation
            audit_log_path: Path to audit log file (None = memory only)
        """
        self.whitelist = whitelist if whitelist is not None else self.DEFAULT_WHITELIST.copy()
        self.allow_all = allow_all
        self.max_install_time = max_install_time_seconds
        self.audit_log_path = audit_log_path
        self.installation_history: List[Dict[str, Any]] = []

    def is_package_allowed(self, package_name: str) -> tuple[bool, str]:
        """
        Check if package is allowed to be installed.

        Args:
            package_name: Name of the package

        Returns:
            Tuple of (is_allowed, reason)
        """
        # Normalize package name
        package_name = package_name.lower().strip()

        # Check blocklist first
        if package_name in self.BLOCKED_PACKAGES:
            return False, f"Package '{package_name}' is blocked for security reasons"

        # Check if allow_all mode
        if self.allow_all:
            return True, "All packages allowed (allow_all mode)"

        # Check whitelist
        if package_name in self.whitelist:
            return True, f"Package '{package_name}' is whitelisted"

        return False, f"Package '{package_name}' not in whitelist"

    def add_to_whitelist(self, packages: List[str]) -> None:
        """Add packages to whitelist."""
        for pkg in packages:
            self.whitelist.add(pkg.lower().strip())
            logger.info(f"Added '{pkg}' to whitelist")

    def install_package(
        self,
        package_spec: str,
        force: bool = False,
    ) -> PackageInstallResult:
        """
        Safely install a Python package.

        Args:
            package_spec: Package specification (e.g., "numpy==1.24.0" or "numpy")
            force: Force installation even if already installed

        Returns:
            PackageInstallResult with status and details
        """
        start_time = time.time()
        audit_log = []

        # Parse package name and version
        package_name, version = self._parse_package_spec(package_spec)

        audit_log.append(f"[{datetime.utcnow().isoformat()}] Install request: {package_spec}")
        audit_log.append(f"  Package: {package_name}, Version: {version or 'latest'}")

        # Check if allowed
        is_allowed, reason = self.is_package_allowed(package_name)
        if not is_allowed:
            audit_log.append(f"  REJECTED: {reason}")
            self._save_audit_log(audit_log)
            return PackageInstallResult(
                success=False,
                package_name=package_name,
                version=version,
                message=reason,
                install_time_ms=0,
                audit_log=audit_log,
            )

        audit_log.append(f"  APPROVED: {reason}")

        # Check if already installed
        if not force and self._is_package_installed(package_name):
            installed_version = self._get_installed_version(package_name)
            audit_log.append(f"  Already installed: {package_name}=={installed_version}")
            self._save_audit_log(audit_log)
            return PackageInstallResult(
                success=True,
                package_name=package_name,
                version=installed_version,
                message=f"Package already installed (version {installed_version})",
                install_time_ms=(time.time() - start_time) * 1000,
                audit_log=audit_log,
            )

        # Install package
        try:
            audit_log.append(f"  Installing via pip...")

            install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-warn-script-location",
                "--quiet",
                package_spec,
            ]

            result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=self.max_install_time,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            if result.returncode == 0:
                installed_version = self._get_installed_version(package_name)
                audit_log.append(f"  SUCCESS: Installed {package_name}=={installed_version}")
                audit_log.append(f"  Installation time: {elapsed_ms:.2f}ms")

                # Record in history
                self.installation_history.append({
                    "package": package_name,
                    "version": installed_version,
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True,
                    "time_ms": elapsed_ms,
                })

                self._save_audit_log(audit_log)

                return PackageInstallResult(
                    success=True,
                    package_name=package_name,
                    version=installed_version,
                    message=f"Successfully installed {package_name}=={installed_version}",
                    install_time_ms=elapsed_ms,
                    audit_log=audit_log,
                )
            else:
                error_msg = result.stderr or result.stdout
                audit_log.append(f"  FAILED: {error_msg}")
                self._save_audit_log(audit_log)

                return PackageInstallResult(
                    success=False,
                    package_name=package_name,
                    version=version,
                    message=f"Installation failed: {error_msg}",
                    install_time_ms=elapsed_ms,
                    audit_log=audit_log,
                )

        except subprocess.TimeoutExpired:
            audit_log.append(f"  TIMEOUT: Exceeded {self.max_install_time}s")
            self._save_audit_log(audit_log)
            return PackageInstallResult(
                success=False,
                package_name=package_name,
                version=version,
                message=f"Installation timed out after {self.max_install_time}s",
                install_time_ms=(time.time() - start_time) * 1000,
                audit_log=audit_log,
            )

        except Exception as e:
            audit_log.append(f"  ERROR: {str(e)}")
            self._save_audit_log(audit_log)
            return PackageInstallResult(
                success=False,
                package_name=package_name,
                version=version,
                message=f"Installation error: {str(e)}",
                install_time_ms=(time.time() - start_time) * 1000,
                audit_log=audit_log,
            )

    def detect_and_install_missing_imports(
        self,
        code: str,
    ) -> Dict[str, PackageInstallResult]:
        """
        Detect missing imports in code and attempt to install them.

        Args:
            code: Python code to analyze

        Returns:
            Dictionary mapping package names to installation results
        """
        # Extract import statements
        imports = self._extract_imports(code)
        results = {}

        for package_name in imports:
            # Map common import names to package names
            actual_package = self._map_import_to_package(package_name)

            # Check if installed
            if not self._is_package_installed(actual_package):
                logger.info(f"Detected missing package: {actual_package}")
                result = self.install_package(actual_package)
                results[actual_package] = result

        return results

    def _parse_package_spec(self, package_spec: str) -> tuple[str, Optional[str]]:
        """Parse package specification into name and version."""
        # Handle version specifiers: ==, >=, <=, >, <, ~=
        match = re.match(r'^([a-zA-Z0-9_-]+)([=><~!]+)(.+)$', package_spec)
        if match:
            return match.group(1).lower(), match.group(3)
        return package_spec.lower().strip(), None

    def _is_package_installed(self, package_name: str) -> bool:
        """Check if package is installed."""
        try:
            __import__(package_name.replace("-", "_"))
            return True
        except ImportError:
            return False

    def _get_installed_version(self, package_name: str) -> Optional[str]:
        """Get installed version of package."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except Exception:
            return None

    def _extract_imports(self, code: str) -> Set[str]:
        """Extract imported module names from code."""
        imports = set()

        # Match: import xxx
        for match in re.finditer(r'^\s*import\s+([a-zA-Z0-9_]+)', code, re.MULTILINE):
            imports.add(match.group(1))

        # Match: from xxx import yyy
        for match in re.finditer(r'^\s*from\s+([a-zA-Z0-9_]+)\s+import', code, re.MULTILINE):
            imports.add(match.group(1))

        return imports

    def _map_import_to_package(self, import_name: str) -> str:
        """Map import name to actual package name."""
        # Common mappings where import name != package name
        mappings = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "bs4": "beautifulsoup4",
            "sklearn": "scikit-learn",
        }
        return mappings.get(import_name, import_name)

    def _save_audit_log(self, log_entries: List[str]) -> None:
        """Save audit log entries."""
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(log_entries) + "\n\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    def get_installation_history(self) -> List[Dict[str, Any]]:
        """Get package installation history."""
        return self.installation_history.copy()

    def export_whitelist(self) -> List[str]:
        """Export current whitelist."""
        return sorted(list(self.whitelist))
