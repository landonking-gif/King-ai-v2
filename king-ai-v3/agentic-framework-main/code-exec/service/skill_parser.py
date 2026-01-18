"""
Skill Parser for Anthropic SKILL.md format and format converters.
Module: code-exec/service/skill_parser.py

Supports:
- Parsing SKILL.md (Anthropic format) with YAML frontmatter
- Converting between Anthropic and native skill formats
- Progressive disclosure skill loading
- Dual-format skill validation
"""

import hashlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError

# Handle imports - works both when loaded as package and via importlib
try:
    from .models import SafetyFlag, SkillMetadata
except (ImportError, ValueError):
    # Fallback when loaded via importlib or without package context
    import sys
    if "models" in sys.modules:
        SafetyFlag = sys.modules["models"].SafetyFlag
        SkillMetadata = sys.modules["models"].SkillMetadata
    else:
        # Last resort: direct import
        from models import SafetyFlag, SkillMetadata

logger = logging.getLogger(__name__)


class SkillParserError(Exception):
    """Base exception for skill parser errors."""

    pass


class InvalidSkillFormatError(SkillParserError):
    """Raised when skill format is invalid."""

    pass


@dataclass
class AnthropicSkillMetadata:
    """Metadata from SKILL.md frontmatter."""

    name: str
    description: str
    dependencies: Optional[List[str]] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class SkillFormat:
    """Detected skill format."""

    has_skill_md: bool  # Anthropic SKILL.md format
    has_skill_yaml: bool  # Native skill.yaml format
    has_schema_json: bool  # Native schema.json
    has_handler_py: bool  # Native handler.py
    is_hybrid: bool  # Both formats present
    is_anthropic_only: bool  # Only SKILL.md
    is_native_only: bool  # Only native format


@dataclass
class SkillResource:
    """Individual skill resource file."""

    path: Path  # Relative path within skill directory
    absolute_path: Path  # Absolute path for loading
    resource_type: str  # "script", "reference", "asset"
    file_type: str  # File extension (py, md, png, etc.)
    size_bytes: int  # File size
    description: Optional[str] = None  # Parsed from markdown link text


@dataclass
class SkillResources:
    """Discovered resources for a skill (Anthropic format)."""

    skill_dir: Path
    scripts: List[SkillResource]  # Executable scripts (scripts/)
    references: List[SkillResource]  # Documentation files (references/)
    assets: List[SkillResource]  # Output assets (assets/)
    markdown_links: List[Dict[str, str]]  # Parsed [text](path) links from SKILL.md

    def get_script(self, name: str) -> Optional[SkillResource]:
        """Get script by name."""
        for s in self.scripts:
            if s.path.name == name or s.path.stem == name:
                return s
        return None

    def get_reference(self, name: str) -> Optional[SkillResource]:
        """Get reference by name or path."""
        for r in self.references:
            if r.path.name == name or str(r.path) == name:
                return r
        return None

    def get_asset(self, name: str) -> Optional[SkillResource]:
        """Get asset by name."""
        for a in self.assets:
            if a.path.name == name:
                return a
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "skill_dir": str(self.skill_dir),
            "scripts": [
                {"path": str(s.path), "type": s.file_type, "size": s.size_bytes}
                for s in self.scripts
            ],
            "references": [
                {"path": str(r.path), "type": r.file_type, "size": r.size_bytes}
                for r in self.references
            ],
            "assets": [
                {"path": str(a.path), "type": a.file_type, "size": a.size_bytes}
                for a in self.assets
            ],
            "markdown_links": self.markdown_links,
        }


class SkillResourceLoader:
    """
    Load and manage skill resources (Anthropic format).

    Supports progressive disclosure:
    - Level 1: Metadata only (name, description)
    - Level 2: Full SKILL.md content
    - Level 3: On-demand reference loading

    Directory structure:
        skill-name/
        ├── SKILL.md              # Required - main instructions
        ├── scripts/              # Executable Python/Bash scripts
        │   └── process.py        # Claude can run WITHOUT loading into context
        ├── references/           # Documentation loaded on-demand
        │   ├── schema.md         # e.g., database schemas
        │   └── api.md            # e.g., API documentation
        └── assets/               # Files for output generation
            └── template.pptx     # Not loaded into context
    """

    # Supported file types by resource category
    SCRIPT_EXTENSIONS = {".py", ".sh", ".bash", ".js", ".ts"}
    REFERENCE_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".xml"}
    ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".pptx", ".docx", ".xlsx"}

    # Markdown link pattern: [text](path)
    MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    @classmethod
    def discover_resources(cls, skill_dir: Path) -> SkillResources:
        """
        Discover all resources in a skill directory.

        Args:
            skill_dir: Path to skill directory

        Returns:
            SkillResources with discovered scripts, references, and assets
        """
        scripts: List[SkillResource] = []
        references: List[SkillResource] = []
        assets: List[SkillResource] = []
        markdown_links: List[Dict[str, str]] = []

        # Discover scripts/
        scripts_dir = skill_dir / "scripts"
        if scripts_dir.exists() and scripts_dir.is_dir():
            for file_path in scripts_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in cls.SCRIPT_EXTENSIONS:
                    scripts.append(
                        SkillResource(
                            path=file_path.relative_to(skill_dir),
                            absolute_path=file_path,
                            resource_type="script",
                            file_type=file_path.suffix.lower().lstrip("."),
                            size_bytes=file_path.stat().st_size,
                        )
                    )

        # Discover references/
        references_dir = skill_dir / "references"
        if references_dir.exists() and references_dir.is_dir():
            for file_path in references_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in cls.REFERENCE_EXTENSIONS:
                    references.append(
                        SkillResource(
                            path=file_path.relative_to(skill_dir),
                            absolute_path=file_path,
                            resource_type="reference",
                            file_type=file_path.suffix.lower().lstrip("."),
                            size_bytes=file_path.stat().st_size,
                        )
                    )

        # Discover assets/
        assets_dir = skill_dir / "assets"
        if assets_dir.exists() and assets_dir.is_dir():
            for file_path in assets_dir.rglob("*"):
                if file_path.is_file():
                    assets.append(
                        SkillResource(
                            path=file_path.relative_to(skill_dir),
                            absolute_path=file_path,
                            resource_type="asset",
                            file_type=file_path.suffix.lower().lstrip("."),
                            size_bytes=file_path.stat().st_size,
                        )
                    )

        # Parse markdown links from SKILL.md
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            markdown_links = cls.parse_markdown_links(skill_md)

        return SkillResources(
            skill_dir=skill_dir,
            scripts=scripts,
            references=references,
            assets=assets,
            markdown_links=markdown_links,
        )

    @classmethod
    def parse_markdown_links(cls, skill_md_path: Path) -> List[Dict[str, str]]:
        """
        Parse markdown links from SKILL.md to find file references.

        Extracts [text](path) patterns and categorizes them.

        Args:
            skill_md_path: Path to SKILL.md file

        Returns:
            List of dicts with {text, path, type, exists}
        """
        links: List[Dict[str, str]] = []

        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()

            skill_dir = skill_md_path.parent

            for match in cls.MARKDOWN_LINK_PATTERN.finditer(content):
                text = match.group(1)
                path = match.group(2)

                # Skip external URLs
                if path.startswith(("http://", "https://", "mailto:", "#")):
                    continue

                # Handle anchor links (e.g., file.md#section)
                anchor = None
                file_path = path
                if "#" in path:
                    file_path, anchor = path.split("#", 1)

                # Determine resource type from path
                resource_type = "unknown"
                if file_path.startswith("scripts/"):
                    resource_type = "script"
                elif file_path.startswith("references/") or file_path.startswith("reference/"):
                    resource_type = "reference"
                elif file_path.startswith("assets/"):
                    resource_type = "asset"
                elif file_path.endswith(".md"):
                    resource_type = "reference"  # Assume .md files are references
                elif file_path.endswith((".py", ".sh")):
                    resource_type = "script"

                # Check if file exists (use file_path without anchor)
                full_path = skill_dir / file_path
                exists = full_path.exists()

                links.append({
                    "text": text,
                    "path": path,
                    "file_path": file_path,
                    "anchor": anchor,
                    "type": resource_type,
                    "exists": exists,
                    "absolute_path": str(full_path) if exists else None,
                })

        except Exception as e:
            logger.warning(f"Failed to parse markdown links from {skill_md_path}: {e}")

        return links

    @classmethod
    def load_reference(
        cls,
        skill_dir: Path,
        reference_path: str,
        max_size_kb: int = 100,
    ) -> Optional[str]:
        """
        Load a reference file content (progressive disclosure - Level 3).

        This is called on-demand when Claude needs specific documentation.

        Args:
            skill_dir: Path to skill directory
            reference_path: Relative path to reference file
            max_size_kb: Maximum file size to load (default 100KB)

        Returns:
            File content as string, or None if not found/too large
        """
        # Normalize path
        if reference_path.startswith("references/"):
            full_path = skill_dir / reference_path
        else:
            full_path = skill_dir / "references" / reference_path

        if not full_path.exists():
            # Try without references/ prefix
            full_path = skill_dir / reference_path
            if not full_path.exists():
                logger.warning(f"Reference file not found: {reference_path}")
                return None

        # Check file size
        size_kb = full_path.stat().st_size / 1024
        if size_kb > max_size_kb:
            logger.warning(
                f"Reference file too large ({size_kb:.1f}KB > {max_size_kb}KB): {reference_path}"
            )
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load reference {reference_path}: {e}")
            return None

    @classmethod
    def get_script_path(cls, skill_dir: Path, script_name: str) -> Optional[Path]:
        """
        Get the absolute path to a script (for execution without loading).

        Args:
            skill_dir: Path to skill directory
            script_name: Script filename or relative path

        Returns:
            Absolute path to script if found
        """
        # Try scripts/ directory first
        if script_name.startswith("scripts/"):
            full_path = skill_dir / script_name
        else:
            full_path = skill_dir / "scripts" / script_name

        if full_path.exists():
            return full_path

        # Try without scripts/ prefix
        full_path = skill_dir / script_name
        if full_path.exists():
            return full_path

        return None

    @classmethod
    def execute_script(
        cls,
        skill_dir: Path,
        script_name: str,
        args: Optional[List[str]] = None,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Execute a skill script without loading it into context.

        This provides deterministic, token-efficient execution.

        Args:
            skill_dir: Path to skill directory
            script_name: Script filename
            args: Command-line arguments
            timeout_seconds: Execution timeout

        Returns:
            Dict with {success, stdout, stderr, return_code}
        """
        import subprocess

        script_path = cls.get_script_path(skill_dir, script_name)
        if not script_path:
            return {
                "success": False,
                "error": f"Script not found: {script_name}",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

        # Determine executor based on file type
        ext = script_path.suffix.lower()
        if ext == ".py":
            cmd = ["python", str(script_path)]
        elif ext in (".sh", ".bash"):
            cmd = ["bash", str(script_path)]
        elif ext == ".js":
            cmd = ["node", str(script_path)]
        else:
            cmd = [str(script_path)]

        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(skill_dir),
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Script timed out after {timeout_seconds}s",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
            }

    @classmethod
    def get_asset_path(cls, skill_dir: Path, asset_name: str) -> Optional[Path]:
        """
        Get the absolute path to an asset (for output generation).

        Args:
            skill_dir: Path to skill directory
            asset_name: Asset filename or relative path

        Returns:
            Absolute path to asset if found
        """
        if asset_name.startswith("assets/"):
            full_path = skill_dir / asset_name
        else:
            full_path = skill_dir / "assets" / asset_name

        if full_path.exists():
            return full_path

        return None

    @classmethod
    def validate_resources(cls, resources: SkillResources) -> List[str]:
        """
        Validate skill resources for completeness and correctness.

        Args:
            resources: Discovered skill resources

        Returns:
            List of validation warnings/errors
        """
        issues: List[str] = []

        # Check for broken markdown links
        for link in resources.markdown_links:
            if not link.get("exists"):
                issues.append(f"Broken link in SKILL.md: [{link['text']}]({link['path']})")

        # Check for scripts without execute permission (Unix)
        import os
        for script in resources.scripts:
            if os.name != "nt":  # Skip on Windows
                if not os.access(script.absolute_path, os.X_OK):
                    issues.append(f"Script not executable: {script.path}")

        # Check for very large references (>500KB)
        for ref in resources.references:
            size_kb = ref.size_bytes / 1024
            if size_kb > 500:
                issues.append(
                    f"Large reference file ({size_kb:.0f}KB): {ref.path} - consider splitting"
                )

        return issues


class SkillParser:
    """Parser for Anthropic SKILL.md format."""

    # Anthropic limits
    MAX_NAME_LENGTH = 64
    MAX_DESCRIPTION_LENGTH = 200
    MAX_SKILL_MD_SIZE_KB = 500

    @staticmethod
    def parse_skill_md(skill_md_path: Path) -> Tuple[AnthropicSkillMetadata, str]:
        """
        Parse SKILL.md file with YAML frontmatter.

        Args:
            skill_md_path: Path to SKILL.md file

        Returns:
            Tuple of (metadata, markdown_content)

        Raises:
            InvalidSkillFormatError: If format is invalid
        """
        if not skill_md_path.exists():
            raise InvalidSkillFormatError(f"SKILL.md not found: {skill_md_path}")

        # Check file size (max 500KB per Anthropic spec)
        size_kb = skill_md_path.stat().st_size / 1024
        if size_kb > SkillParser.MAX_SKILL_MD_SIZE_KB:
            raise InvalidSkillFormatError(
                f"SKILL.md exceeds {SkillParser.MAX_SKILL_MD_SIZE_KB}KB limit: {size_kb:.1f}KB"
            )

        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse YAML frontmatter
        if not content.startswith("---\n"):
            raise InvalidSkillFormatError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---\n", 2)
        if len(parts) < 3:
            raise InvalidSkillFormatError(
                "SKILL.md must have closing --- after frontmatter"
            )

        frontmatter_str = parts[1]
        markdown = parts[2].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            raise InvalidSkillFormatError(f"Invalid YAML frontmatter: {e}")

        # Validate required fields
        if "name" not in frontmatter:
            raise InvalidSkillFormatError("SKILL.md frontmatter must have 'name' field")
        if "description" not in frontmatter:
            raise InvalidSkillFormatError("SKILL.md frontmatter must have 'description' field")

        # Validate constraints
        name = frontmatter["name"]
        description = frontmatter["description"]

        if not isinstance(name, str) or len(name) > SkillParser.MAX_NAME_LENGTH:
            raise InvalidSkillFormatError(
                f"Skill name must be string <= {SkillParser.MAX_NAME_LENGTH} characters"
            )

        if not isinstance(description, str) or len(description) > SkillParser.MAX_DESCRIPTION_LENGTH:
            raise InvalidSkillFormatError(
                f"Skill description must be string <= {SkillParser.MAX_DESCRIPTION_LENGTH} chars"
            )

        # Parse dependencies (e.g., "python>=3.8, pypdf>=3.0.0")
        dependencies = None
        if "dependencies" in frontmatter:
            deps_str = frontmatter["dependencies"]
            if isinstance(deps_str, str):
                dependencies = [d.strip() for d in deps_str.split(",")]
            elif isinstance(deps_str, list):
                dependencies = deps_str

        metadata = AnthropicSkillMetadata(
            name=name,
            description=description,
            dependencies=dependencies,
            version=frontmatter.get("version", "1.0.0"),
            author=frontmatter.get("author"),
            tags=frontmatter.get("tags", []),
        )

        return metadata, markdown

    @staticmethod
    def load_skill_metadata_only(skill_dir: Path) -> Dict[str, Any]:
        """
        Load only skill metadata (Level 1 - Progressive Disclosure).
        Lightweight loading for skill discovery.

        Args:
            skill_dir: Path to skill directory

        Returns:
            Minimal metadata dict
        """
        skill_md = skill_dir / "SKILL.md"
        skill_yaml = skill_dir / "skill.yaml"

        metadata = {"name": skill_dir.name, "format": "unknown"}

        # Try SKILL.md first
        if skill_md.exists():
            try:
                # Only read first 1KB for frontmatter
                with open(skill_md, "r", encoding="utf-8") as f:
                    header = f.read(1024)

                if header.startswith("---\n"):
                    parts = header.split("---\n", 2)
                    if len(parts) >= 2:
                        frontmatter = yaml.safe_load(parts[1])
                        metadata.update(
                            {
                                "name": frontmatter.get("name", skill_dir.name),
                                "description": frontmatter.get("description", ""),
                                "format": "anthropic",
                            }
                        )
            except Exception as e:
                logger.warning(f"Failed to load SKILL.md metadata from {skill_dir}: {e}")

        # Try skill.yaml
        elif skill_yaml.exists():
            try:
                with open(skill_yaml, "r") as f:
                    config = yaml.safe_load(f)
                metadata.update(
                    {
                        "name": config.get("name", skill_dir.name),
                        "description": config.get("description", ""),
                        "version": config.get("version", "1.0.0"),
                        "format": "native",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load skill.yaml from {skill_dir}: {e}")

        return metadata


class FormatDetector:
    """Detect skill format (Anthropic, Native, or Hybrid)."""

    @staticmethod
    def detect_format(skill_dir: Path) -> SkillFormat:
        """
        Detect skill format from directory contents.

        Args:
            skill_dir: Path to skill directory

        Returns:
            SkillFormat with detected formats
        """
        has_skill_md = (skill_dir / "SKILL.md").exists()
        has_skill_yaml = (skill_dir / "skill.yaml").exists()
        has_schema_json = (skill_dir / "schema.json").exists()
        has_handler_py = (skill_dir / "handler.py").exists()

        is_anthropic_only = has_skill_md and not has_skill_yaml
        is_native_only = has_skill_yaml and has_schema_json and not has_skill_md
        is_hybrid = has_skill_md and has_skill_yaml and has_schema_json

        return SkillFormat(
            has_skill_md=has_skill_md,
            has_skill_yaml=has_skill_yaml,
            has_schema_json=has_schema_json,
            has_handler_py=has_handler_py,
            is_hybrid=is_hybrid,
            is_anthropic_only=is_anthropic_only,
            is_native_only=is_native_only,
        )

    @staticmethod
    def detect_format_with_resources(skill_dir: Path) -> Tuple[SkillFormat, SkillResources]:
        """
        Detect skill format and discover all resources.

        This is the comprehensive detection method that returns both
        format information and all available resources.

        Args:
            skill_dir: Path to skill directory

        Returns:
            Tuple of (SkillFormat, SkillResources)
        """
        skill_format = FormatDetector.detect_format(skill_dir)
        resources = SkillResourceLoader.discover_resources(skill_dir)
        return skill_format, resources

    @staticmethod
    def get_skill_summary(skill_dir: Path) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a skill's format and resources.

        Useful for skill listing and validation.

        Args:
            skill_dir: Path to skill directory

        Returns:
            Dict with format, resources, and validation info
        """
        skill_format, resources = FormatDetector.detect_format_with_resources(skill_dir)
        validation_issues = SkillResourceLoader.validate_resources(resources)

        return {
            "name": skill_dir.name,
            "path": str(skill_dir),
            "format": {
                "has_skill_md": skill_format.has_skill_md,
                "has_skill_yaml": skill_format.has_skill_yaml,
                "has_schema_json": skill_format.has_schema_json,
                "has_handler_py": skill_format.has_handler_py,
                "is_hybrid": skill_format.is_hybrid,
                "is_anthropic_only": skill_format.is_anthropic_only,
                "is_native_only": skill_format.is_native_only,
            },
            "resources": {
                "scripts_count": len(resources.scripts),
                "references_count": len(resources.references),
                "assets_count": len(resources.assets),
                "markdown_links_count": len(resources.markdown_links),
                "scripts": [str(s.path) for s in resources.scripts],
                "references": [str(r.path) for r in resources.references],
                "assets": [str(a.path) for a in resources.assets],
            },
            "validation": {
                "is_valid": len(validation_issues) == 0,
                "issues": validation_issues,
            },
        }


class SkillConverter:
    """Convert between Anthropic and native skill formats."""

    @staticmethod
    def anthropic_to_native(
        skill_dir: Path,
        add_safety_flags: Optional[List[SafetyFlag]] = None,
        requires_approval: bool = False,
    ) -> SkillMetadata:
        """
        Convert Anthropic SKILL.md to native format.

        Creates skill.yaml and schema.json from SKILL.md.
        Does not create handler.py (requires manual implementation).

        Args:
            skill_dir: Path to skill directory with SKILL.md
            add_safety_flags: Safety flags to assign (requires manual review)
            requires_approval: Whether skill requires human approval

        Returns:
            SkillMetadata for registered skill

        Raises:
            InvalidSkillFormatError: If conversion fails
        """
        skill_md_path = skill_dir / "SKILL.md"

        # Parse SKILL.md
        metadata, markdown = SkillParser.parse_skill_md(skill_md_path)

        # Generate basic schema from markdown content
        # (This is a stub - real implementation would use LLM to infer schema)
        input_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input data for skill",
                }
            },
            "required": ["input"],
        }

        output_schema = {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "Result from skill execution",
                }
            },
            "required": ["result"],
        }

        # Create skill.yaml
        skill_config = {
            "name": metadata.name,
            "version": metadata.version or "1.0.0",
            "description": metadata.description,
            "safety_flags": [flag.value for flag in (add_safety_flags or [SafetyFlag.NONE])],
            "requires_approval": requires_approval,
            "handler": f"handler.{metadata.name.replace('-', '_')}",
            "tags": metadata.tags or [],
            "anthropic_compatible": True,
            "dependencies": metadata.dependencies or [],
        }

        skill_yaml_path = skill_dir / "skill.yaml"
        with open(skill_yaml_path, "w") as f:
            yaml.dump(skill_config, f, default_flow_style=False)

        # Create schema.json
        schema_data = {"input": input_schema, "output": output_schema}

        schema_json_path = skill_dir / "schema.json"
        with open(schema_json_path, "w") as f:
            json.dump(schema_data, f, indent=2)

        # Create handler stub
        handler_py_path = skill_dir / "handler.py"
        if not handler_py_path.exists():
            handler_stub = f'''"""
Handler for {metadata.name} skill.
Generated from SKILL.md conversion.
"""

from typing import Any, Dict


async def {metadata.name.replace("-", "_")}(input: str) -> Dict[str, Any]:
    """
    {metadata.description}

    Args:
        input: Input data

    Returns:
        Result dictionary

    Note:
        This is a stub. Implement actual logic based on SKILL.md instructions:
        {skill_md_path}
    """
    # TODO: Implement skill logic from SKILL.md
    raise NotImplementedError(
        f"Handler for {{metadata.name}} needs manual implementation. "
        f"See SKILL.md for instructions."
    )
'''
            with open(handler_py_path, "w") as f:
                f.write(handler_stub)

        logger.info(f"Converted {metadata.name} from Anthropic to native format")

        # Return metadata
        return SkillMetadata(
            name=metadata.name,
            version=metadata.version or "1.0.0",
            description=metadata.description,
            safety_flags=add_safety_flags or [SafetyFlag.NONE],
            requires_approval=requires_approval,
            input_schema=input_schema,
            output_schema=output_schema,
            handler_path=skill_config["handler"],
            tags=metadata.tags or [],
        )

    @staticmethod
    def native_to_anthropic(skill_dir: Path, include_handler: bool = True) -> Path:
        """
        Convert native skill format to Anthropic SKILL.md.

        Args:
            skill_dir: Path to skill directory with native format
            include_handler: Include handler.py as resource file

        Returns:
            Path to generated SKILL.md

        Raises:
            InvalidSkillFormatError: If conversion fails
        """
        skill_yaml_path = skill_dir / "skill.yaml"
        schema_json_path = skill_dir / "schema.json"

        if not skill_yaml_path.exists() or not schema_json_path.exists():
            raise InvalidSkillFormatError(
                f"Native skill format requires skill.yaml and schema.json in {skill_dir}"
            )

        # Load native format
        with open(skill_yaml_path, "r") as f:
            skill_config = yaml.safe_load(f)

        with open(schema_json_path, "r") as f:
            schemas = json.load(f)

        # Generate SKILL.md
        name = skill_config["name"]
        description = skill_config["description"]
        version = skill_config.get("version", "1.0.0")
        tags = skill_config.get("tags", [])

        # Create frontmatter
        frontmatter = {
            "name": name,
            "description": description,
            "version": version,
        }

        if tags:
            frontmatter["tags"] = tags

        if "dependencies" in skill_config:
            frontmatter["dependencies"] = skill_config["dependencies"]

        # Generate markdown content
        markdown_parts = [f"# {name}", "", description, "", "## Input Schema", ""]

        # Format input schema as markdown
        input_schema = schemas.get("input", {})
        if "properties" in input_schema:
            markdown_parts.append("**Parameters:**")
            for prop_name, prop_schema in input_schema["properties"].items():
                prop_desc = prop_schema.get("description", "")
                prop_type = prop_schema.get("type", "any")
                required = prop_name in input_schema.get("required", [])
                req_marker = " (required)" if required else " (optional)"
                markdown_parts.append(f"- `{prop_name}` ({prop_type}){req_marker}: {prop_desc}")

        markdown_parts.extend(["", "## Output Schema", ""])

        # Format output schema
        output_schema = schemas.get("output", {})
        if "properties" in output_schema:
            markdown_parts.append("**Returns:**")
            for prop_name, prop_schema in output_schema["properties"].items():
                prop_desc = prop_schema.get("description", "")
                prop_type = prop_schema.get("type", "any")
                markdown_parts.append(f"- `{prop_name}` ({prop_type}): {prop_desc}")

        # Add safety information
        safety_flags = skill_config.get("safety_flags", [])
        if safety_flags and safety_flags != ["none"]:
            markdown_parts.extend(
                [
                    "",
                    "## Safety Considerations",
                    "",
                    f"This skill has the following safety flags: {', '.join(safety_flags)}",
                ]
            )

        # Add handler reference
        if include_handler:
            handler_py_path = skill_dir / "handler.py"
            if handler_py_path.exists():
                markdown_parts.extend(
                    [
                        "",
                        "## Implementation",
                        "",
                        "See `handler.py` for the deterministic implementation of this skill.",
                    ]
                )

        markdown_content = "\n".join(markdown_parts)

        # Write SKILL.md
        skill_md_path = skill_dir / "SKILL.md"
        with open(skill_md_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            yaml.dump(frontmatter, f, default_flow_style=False)
            f.write("---\n\n")
            f.write(markdown_content)

        logger.info(f"Converted {name} from native to Anthropic format")

        return skill_md_path


class SkillPackager:
    """Package skills for distribution."""

    @staticmethod
    def package_skill_zip(skill_dir: Path, output_path: Optional[Path] = None) -> Path:
        """
        Package skill directory as ZIP for Anthropic marketplace.

        Args:
            skill_dir: Path to skill directory
            output_path: Optional output path (defaults to skill_dir parent)

        Returns:
            Path to generated ZIP file
        """
        if output_path is None:
            output_path = skill_dir.parent / f"{skill_dir.name}.zip"

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in skill_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(skill_dir.parent)
                    zipf.write(file_path, arcname)

        logger.info(f"Packaged {skill_dir.name} to {output_path}")
        return output_path

    @staticmethod
    def unpack_skill_zip(zip_path: Path, target_dir: Path) -> Path:
        """
        Unpack skill ZIP file.

        Args:
            zip_path: Path to ZIP file
            target_dir: Target directory for extraction

        Returns:
            Path to extracted skill directory
        """
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(target_dir)

        # Find skill directory (should be top-level directory in ZIP)
        extracted_items = list(target_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            skill_dir = extracted_items[0]
        else:
            skill_dir = target_dir

        logger.info(f"Unpacked {zip_path} to {skill_dir}")
        return skill_dir
