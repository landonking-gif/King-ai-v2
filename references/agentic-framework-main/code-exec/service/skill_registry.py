"""
Skill Registry - Centralized skill discovery and management.

Module: code-exec/service/skill_registry.py

Provides auto-discovery of skills from multiple paths by scanning for:
- SKILL.md (Anthropic format)
- skill.yaml (Native format)

Supports:
- Framework-level skills (code-exec/skills/)
- Project-level skills (./skills)
- User-level skills (~/.kautilya/skills/)
- Custom paths via environment variable or config
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import from skill_parser (same directory)
try:
    from .skill_parser import (
        AnthropicSkillMetadata,
        FormatDetector,
        SkillFormat,
        SkillParser,
        SkillResourceLoader,
        SkillResources,
    )
except ImportError:
    # Fallback for direct execution
    from skill_parser import (
        AnthropicSkillMetadata,
        FormatDetector,
        SkillFormat,
        SkillParser,
        SkillResourceLoader,
        SkillResources,
    )

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Complete metadata for a discovered skill."""

    name: str
    path: Path
    source_path: Path  # Which search path it was found in
    format: SkillFormat
    resources: Optional[SkillResources] = None

    # From SKILL.md or skill.yaml
    description: str = ""
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Handler information (for native/hybrid skills)
    handler_module: Optional[str] = None
    handler_function: Optional[str] = None

    # Safety and policy
    safety_flags: List[str] = field(default_factory=list)
    requires_approval: bool = False

    # Anthropic compatibility
    anthropic_compatible: bool = False

    @property
    def format_label(self) -> str:
        """Get human-readable format label."""
        if self.format.is_hybrid:
            return "Hybrid"
        elif self.format.is_anthropic_only:
            return "Anthropic"
        elif self.format.is_native_only:
            return "Native"
        return "Unknown"

    @property
    def has_handler(self) -> bool:
        """Check if skill has a Python handler."""
        return self.format.has_handler_py

    @property
    def has_resources(self) -> bool:
        """Check if skill has scripts/references/assets."""
        if not self.resources:
            return False
        return bool(
            self.resources.scripts
            or self.resources.references
            or self.resources.assets
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "source_path": str(self.source_path),
            "format": self.format_label,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "handler_module": self.handler_module,
            "handler_function": self.handler_function,
            "safety_flags": self.safety_flags,
            "requires_approval": self.requires_approval,
            "anthropic_compatible": self.anthropic_compatible,
            "has_handler": self.has_handler,
            "has_resources": self.has_resources,
            "resources": self.resources.to_dict() if self.resources else None,
        }


class SkillRegistry:
    """
    Centralized skill discovery and management.

    Auto-discovers skills by scanning configured paths for:
    - SKILL.md (Anthropic format)
    - skill.yaml (Native format)

    Supports multiple skill locations with priority:
    1. Environment variable: KAUTILYA_SKILLS_PATH (colon-separated)
    2. Project-level: ./skills
    3. Framework-level: {framework_root}/code-exec/skills
    4. User-level: ~/.kautilya/skills

    Usage:
        registry = SkillRegistry()
        skills = registry.discover_all()

        skill = registry.get_skill("deep_research")
        if skill:
            print(f"Found: {skill.path}")
    """

    # Cache settings
    _DEFAULT_CACHE_TTL = 300  # 5 minutes

    # Singleton instance for global access
    _instance: Optional["SkillRegistry"] = None

    def __init__(
        self,
        skill_paths: Optional[List[Path]] = None,
        cache_ttl: int = _DEFAULT_CACHE_TTL,
        auto_discover: bool = True,
    ):
        """
        Initialize the skill registry.

        Args:
            skill_paths: Custom skill paths (overrides defaults if provided)
            cache_ttl: Cache time-to-live in seconds
            auto_discover: Whether to discover skills on init
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, SkillMetadata] = {}
        self._cache_timestamp: float = 0
        self._skill_paths: List[Path] = []

        # Set up skill paths
        if skill_paths:
            self._skill_paths = [p for p in skill_paths if p.exists()]
        else:
            self._skill_paths = self._get_default_paths()

        # Auto-discover on init
        if auto_discover:
            self.refresh()

    @classmethod
    def get_instance(cls) -> "SkillRegistry":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = SkillRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None

    @property
    def skill_paths(self) -> List[Path]:
        """Get configured skill paths."""
        return self._skill_paths.copy()

    def _get_default_paths(self) -> List[Path]:
        """
        Get default skill paths with priority ordering.

        Priority:
        1. KAUTILYA_SKILLS_PATH environment variable
        2. Project-level ./skills
        3. Framework-level code-exec/skills
        4. User-level ~/.kautilya/skills
        """
        paths: List[Path] = []

        # 1. Environment variable (highest priority)
        env_paths = os.getenv("KAUTILYA_SKILLS_PATH", "")
        if env_paths:
            for p in env_paths.split(":"):
                path = Path(p.strip()).resolve()
                if path.exists() and path not in paths:
                    paths.append(path)
                    logger.debug(f"Added skill path from env: {path}")

        # 2. Project-level skills (current working directory)
        project_skills = Path.cwd() / "skills"
        if project_skills.exists() and project_skills not in paths:
            paths.append(project_skills)
            logger.debug(f"Added project skill path: {project_skills}")

        # 3. Framework-level skills (code-exec/skills)
        framework_root = self._find_framework_root()
        if framework_root:
            framework_skills = framework_root / "code-exec" / "skills"
            if framework_skills.exists() and framework_skills not in paths:
                paths.append(framework_skills)
                logger.debug(f"Added framework skill path: {framework_skills}")

        # 4. User-level skills (~/.kautilya/skills)
        user_skills = Path.home() / ".kautilya" / "skills"
        if user_skills.exists() and user_skills not in paths:
            paths.append(user_skills)
            logger.debug(f"Added user skill path: {user_skills}")

        return paths

    def _find_framework_root(self) -> Optional[Path]:
        """
        Find the framework root directory.

        Searches upward from this file for markers like:
        - CLAUDE.md
        - .git with agent-framework
        - pyproject.toml with agent-framework
        """
        current = Path(__file__).resolve().parent

        # Walk up to find framework root
        for _ in range(10):
            # Check for CLAUDE.md (framework marker)
            if (current / "CLAUDE.md").exists():
                return current

            # Check for .git directory
            if (current / ".git").exists():
                return current

            # Check for code-exec directory (we're inside it)
            if (current / "code-exec").exists():
                return current

            parent = current.parent
            if parent == current:
                break
            current = parent

        # Fallback: assume we're in code-exec/service
        return Path(__file__).resolve().parent.parent.parent

    def add_skill_path(self, path: Path) -> bool:
        """
        Add a new skill path to search.

        Args:
            path: Path to add

        Returns:
            True if path was added, False if already exists or invalid
        """
        path = path.resolve()
        if not path.exists():
            logger.warning(f"Skill path does not exist: {path}")
            return False

        if path in self._skill_paths:
            return False

        self._skill_paths.append(path)
        self._invalidate_cache()
        return True

    def remove_skill_path(self, path: Path) -> bool:
        """
        Remove a skill path from search.

        Args:
            path: Path to remove

        Returns:
            True if path was removed
        """
        path = path.resolve()
        if path in self._skill_paths:
            self._skill_paths.remove(path)
            self._invalidate_cache()
            return True
        return False

    def _invalidate_cache(self) -> None:
        """Invalidate the skill cache."""
        self._cache = {}
        self._cache_timestamp = 0

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self._cache:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl

    def refresh(self) -> int:
        """
        Refresh skill cache by rescanning all paths.

        Returns:
            Number of skills discovered
        """
        self._cache = {}
        discovered = 0

        for base_path in self._skill_paths:
            skills = self._discover_in_path(base_path)
            for skill in skills:
                # Use skill name as key (first occurrence wins)
                if skill.name not in self._cache:
                    self._cache[skill.name] = skill
                    discovered += 1
                else:
                    logger.debug(
                        f"Skill '{skill.name}' already found in {self._cache[skill.name].source_path}, "
                        f"skipping duplicate in {base_path}"
                    )

        self._cache_timestamp = time.time()
        logger.info(f"Discovered {discovered} skills from {len(self._skill_paths)} paths")
        return discovered

    def _discover_in_path(self, base_path: Path) -> List[SkillMetadata]:
        """
        Discover skills in a single directory.

        Args:
            base_path: Directory to scan

        Returns:
            List of discovered skills
        """
        skills: List[SkillMetadata] = []

        if not base_path.exists() or not base_path.is_dir():
            return skills

        for item in base_path.iterdir():
            if not item.is_dir() or item.name.startswith("."):
                continue

            # Check for skill markers
            skill_md = item / "SKILL.md"
            skill_yaml = item / "skill.yaml"

            if not skill_md.exists() and not skill_yaml.exists():
                continue

            try:
                skill = self._load_skill_metadata(item, base_path)
                if skill:
                    skills.append(skill)
            except Exception as e:
                logger.warning(f"Failed to load skill from {item}: {e}")

        return skills

    def _load_skill_metadata(
        self, skill_dir: Path, source_path: Path
    ) -> Optional[SkillMetadata]:
        """
        Load complete metadata for a skill.

        Args:
            skill_dir: Path to skill directory
            source_path: Base path where skill was found

        Returns:
            SkillMetadata or None if loading fails
        """
        # Detect format
        format_info = FormatDetector.detect_format(skill_dir)

        # Discover resources
        resources = SkillResourceLoader.discover_resources(skill_dir)

        # Initialize metadata
        metadata = SkillMetadata(
            name=skill_dir.name,
            path=skill_dir,
            source_path=source_path,
            format=format_info,
            resources=resources,
        )

        # Load from skill.yaml (native format) - preferred for detailed config
        if format_info.has_skill_yaml:
            self._load_from_skill_yaml(skill_dir / "skill.yaml", metadata)

        # Load from SKILL.md (Anthropic format) - fills in gaps
        if format_info.has_skill_md:
            self._load_from_skill_md(skill_dir / "SKILL.md", metadata)
            metadata.anthropic_compatible = True

        return metadata

    def _load_from_skill_yaml(self, yaml_path: Path, metadata: SkillMetadata) -> None:
        """Load metadata from skill.yaml."""
        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f) or {}

            metadata.name = config.get("name", metadata.name)
            metadata.description = config.get("description", metadata.description)
            metadata.version = config.get("version", metadata.version)
            metadata.author = config.get("author", metadata.author)
            metadata.tags = config.get("tags", metadata.tags)
            metadata.safety_flags = config.get("safety_flags", metadata.safety_flags)
            metadata.requires_approval = config.get(
                "requires_approval", metadata.requires_approval
            )

            # Handler configuration
            handler = config.get("handler", "")
            if handler and "." in handler:
                parts = handler.rsplit(".", 1)
                metadata.handler_module = parts[0]
                metadata.handler_function = parts[1]
            elif handler:
                metadata.handler_function = handler

            # Dependencies
            deps = config.get("dependencies", [])
            if isinstance(deps, list):
                metadata.dependencies = deps

            # Anthropic compatibility flag
            metadata.anthropic_compatible = config.get(
                "anthropic_compatible", metadata.anthropic_compatible
            )

        except Exception as e:
            logger.warning(f"Failed to load skill.yaml from {yaml_path}: {e}")

    def _load_from_skill_md(self, md_path: Path, metadata: SkillMetadata) -> None:
        """Load metadata from SKILL.md."""
        try:
            anthropic_meta, _ = SkillParser.parse_skill_md(md_path)

            # Only fill in gaps (don't overwrite skill.yaml values)
            if not metadata.description:
                metadata.description = anthropic_meta.description
            if metadata.version == "1.0.0" and anthropic_meta.version:
                metadata.version = anthropic_meta.version
            if not metadata.author and anthropic_meta.author:
                metadata.author = anthropic_meta.author
            if not metadata.tags and anthropic_meta.tags:
                metadata.tags = anthropic_meta.tags
            if not metadata.dependencies and anthropic_meta.dependencies:
                metadata.dependencies = anthropic_meta.dependencies

        except Exception as e:
            logger.warning(f"Failed to load SKILL.md from {md_path}: {e}")

    def discover_all(self) -> List[SkillMetadata]:
        """
        Get all discovered skills.

        Returns:
            List of all skill metadata
        """
        if not self._is_cache_valid():
            self.refresh()
        return list(self._cache.values())

    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        """
        Get skill by name.

        Args:
            name: Skill name (e.g., "deep_research")

        Returns:
            SkillMetadata or None if not found
        """
        if not self._is_cache_valid():
            self.refresh()

        # Try exact match first
        if name in self._cache:
            return self._cache[name]

        # Try with underscores/hyphens normalized
        normalized = name.replace("-", "_")
        if normalized in self._cache:
            return self._cache[normalized]

        normalized = name.replace("_", "-")
        if normalized in self._cache:
            return self._cache[normalized]

        return None

    def get_skill_path(self, name: str) -> Optional[Path]:
        """
        Get absolute path to a skill directory.

        Args:
            name: Skill name

        Returns:
            Path or None if not found
        """
        skill = self.get_skill(name)
        return skill.path if skill else None

    def get_skill_handler(self, name: str) -> Optional[Tuple[str, str]]:
        """
        Get handler module and function for a skill.

        Args:
            name: Skill name

        Returns:
            Tuple of (module, function) or None
        """
        skill = self.get_skill(name)
        if skill and skill.handler_module and skill.handler_function:
            return (skill.handler_module, skill.handler_function)
        return None

    def skill_exists(self, name: str) -> bool:
        """Check if a skill exists."""
        return self.get_skill(name) is not None

    def list_skills(self) -> List[str]:
        """Get list of all skill names."""
        if not self._is_cache_valid():
            self.refresh()
        return list(self._cache.keys())

    def list_skills_by_tag(self, tag: str) -> List[SkillMetadata]:
        """Get skills with a specific tag."""
        return [s for s in self.discover_all() if tag in s.tags]

    def list_skills_by_format(self, format_type: str) -> List[SkillMetadata]:
        """
        Get skills by format type.

        Args:
            format_type: "hybrid", "anthropic", or "native"
        """
        skills = self.discover_all()
        if format_type == "hybrid":
            return [s for s in skills if s.format.is_hybrid]
        elif format_type == "anthropic":
            return [s for s in skills if s.format.is_anthropic_only]
        elif format_type == "native":
            return [s for s in skills if s.format.is_native_only]
        return skills

    def list_skills_with_handler(self) -> List[SkillMetadata]:
        """Get skills that have Python handlers."""
        return [s for s in self.discover_all() if s.has_handler]

    def validate_skill(self, name: str) -> Tuple[bool, List[str]]:
        """
        Validate a skill for completeness and correctness.

        Args:
            name: Skill name

        Returns:
            Tuple of (is_valid, list of issues)
        """
        skill = self.get_skill(name)
        if not skill:
            return False, [f"Skill not found: {name}"]

        issues: List[str] = []

        # Check for required fields
        if not skill.description:
            issues.append("Missing description")

        # Validate resources if present
        if skill.resources:
            resource_issues = SkillResourceLoader.validate_resources(skill.resources)
            issues.extend(resource_issues)

        # Check handler exists for native/hybrid
        if skill.format.has_handler_py:
            handler_path = skill.path / "handler.py"
            if not handler_path.exists():
                issues.append("handler.py referenced but not found")

        # Check schema exists for native/hybrid
        if skill.format.has_schema_json:
            schema_path = skill.path / "schema.json"
            if not schema_path.exists():
                issues.append("schema.json referenced but not found")

        return len(issues) == 0, issues

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all discovered skills.

        Returns:
            Dictionary with summary statistics
        """
        skills = self.discover_all()

        return {
            "total_skills": len(skills),
            "skill_paths": [str(p) for p in self._skill_paths],
            "by_format": {
                "hybrid": len([s for s in skills if s.format.is_hybrid]),
                "anthropic": len([s for s in skills if s.format.is_anthropic_only]),
                "native": len([s for s in skills if s.format.is_native_only]),
            },
            "with_handler": len([s for s in skills if s.has_handler]),
            "with_resources": len([s for s in skills if s.has_resources]),
            "anthropic_compatible": len([s for s in skills if s.anthropic_compatible]),
            "skills": [s.to_dict() for s in skills],
        }

    def __repr__(self) -> str:
        return f"SkillRegistry(paths={len(self._skill_paths)}, skills={len(self._cache)})"


# Convenience functions for global access
def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    return SkillRegistry.get_instance()


def discover_skills() -> List[SkillMetadata]:
    """Discover all skills using global registry."""
    return get_skill_registry().discover_all()


def get_skill(name: str) -> Optional[SkillMetadata]:
    """Get a skill by name using global registry."""
    return get_skill_registry().get_skill(name)


def skill_exists(name: str) -> bool:
    """Check if a skill exists using global registry."""
    return get_skill_registry().skill_exists(name)
