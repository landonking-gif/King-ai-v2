"""
Skill Registry for loading and managing registered skills.
Module: code-exec/service/registry.py

Supports both native format (skill.yaml + schema.json) and Anthropic format (SKILL.md).
"""

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError

from .models import SafetyFlag, SkillMetadata
from .skill_parser import FormatDetector, SkillConverter, SkillParser

logger = logging.getLogger(__name__)


class SkillRegistryError(Exception):
    """Base exception for skill registry errors."""

    pass


class SkillNotFoundError(SkillRegistryError):
    """Raised when a skill is not found in the registry."""

    pass


class SkillValidationError(SkillRegistryError):
    """Raised when skill validation fails."""

    pass


class SkillRegistry:
    """Registry for managing skill definitions and metadata."""

    def __init__(self, skills_directory: str) -> None:
        """
        Initialize the skill registry.

        Args:
            skills_directory: Path to directory containing skill definitions
        """
        self.skills_directory = Path(skills_directory)
        self.skills: Dict[str, SkillMetadata] = {}
        self._handlers: Dict[str, Any] = {}

        if not self.skills_directory.exists():
            logger.warning(f"Skills directory does not exist: {self.skills_directory}")
            self.skills_directory.mkdir(parents=True, exist_ok=True)

    async def load_all_skills(self) -> None:
        """Load all skills from the skills directory."""
        logger.info(f"Loading skills from: {self.skills_directory}")

        if not self.skills_directory.exists():
            logger.warning(f"Skills directory not found: {self.skills_directory}")
            return

        for skill_dir in self.skills_directory.iterdir():
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue

            try:
                await self._load_skill_from_directory(skill_dir)
            except Exception as e:
                logger.error(f"Failed to load skill from {skill_dir}: {e}")
                continue

        logger.info(f"Loaded {len(self.skills)} skills successfully")

    async def _load_skill_from_directory(self, skill_dir: Path) -> None:
        """
        Load a skill from a directory (supports both native and Anthropic formats).

        Args:
            skill_dir: Path to skill directory

        Raises:
            SkillValidationError: If skill validation fails
        """
        # Detect skill format
        format_info = FormatDetector.detect_format(skill_dir)

        # Hybrid format: Load native format (preferred)
        if format_info.is_hybrid or format_info.is_native_only:
            await self._load_native_skill(skill_dir)

        # Anthropic-only format: Auto-convert or skip (requires handler implementation)
        elif format_info.is_anthropic_only:
            logger.warning(
                f"Skill {skill_dir.name} is Anthropic-only format (SKILL.md). "
                "Auto-conversion requires manual handler implementation. Skipping."
            )
            # Note: For Anthropic-only skills, we'd need to either:
            # 1. Auto-convert and require manual handler implementation
            # 2. Support instruction-based execution (future feature)
            # For now, we skip and log a warning

        else:
            raise SkillValidationError(
                f"No valid skill format found in {skill_dir}. "
                "Expected: skill.yaml + schema.json OR SKILL.md"
            )

    async def _load_native_skill(self, skill_dir: Path) -> None:
        """
        Load a skill in native format (skill.yaml + schema.json).

        Args:
            skill_dir: Path to skill directory

        Raises:
            SkillValidationError: If skill validation fails
        """
        skill_yaml_path = skill_dir / "skill.yaml"
        schema_json_path = skill_dir / "schema.json"

        if not skill_yaml_path.exists():
            raise SkillValidationError(f"skill.yaml not found in {skill_dir}")

        if not schema_json_path.exists():
            raise SkillValidationError(f"schema.json not found in {skill_dir}")

        # Load skill metadata
        with open(skill_yaml_path, "r") as f:
            skill_config = yaml.safe_load(f)

        # Load schemas
        with open(schema_json_path, "r") as f:
            schemas = json.load(f)

        # Validate schemas
        input_schema = schemas.get("input")
        output_schema = schemas.get("output")

        if not input_schema or not output_schema:
            raise SkillValidationError(
                f"schema.json must contain 'input' and 'output' schemas in {skill_dir}"
            )

        # Validate JSON schemas are valid
        try:
            Draft7Validator.check_schema(input_schema)
            Draft7Validator.check_schema(output_schema)
        except JsonSchemaValidationError as e:
            raise SkillValidationError(f"Invalid JSON schema in {skill_dir}: {e}")

        # Parse safety flags
        safety_flags = [
            SafetyFlag(flag) for flag in skill_config.get("safety_flags", [])
        ]

        # Create skill metadata
        metadata = SkillMetadata(
            name=skill_config["name"],
            version=skill_config["version"],
            description=skill_config["description"],
            safety_flags=safety_flags,
            requires_approval=skill_config.get("requires_approval", False),
            input_schema=input_schema,
            output_schema=output_schema,
            handler_path=skill_config["handler"],
            tags=skill_config.get("tags", []),
        )

        # Load handler function
        handler_module_path = skill_dir / "handler.py"
        if not handler_module_path.exists():
            raise SkillValidationError(f"handler.py not found in {skill_dir}")

        handler_func = self._load_handler_function(
            handler_module_path, skill_config["handler"]
        )

        # Register skill
        self.skills[metadata.name] = metadata
        self._handlers[metadata.name] = handler_func

        # Check if hybrid format
        skill_md_path = skill_dir / "SKILL.md"
        format_label = "hybrid" if skill_md_path.exists() else "native"
        logger.info(f"Loaded skill: {metadata.name} v{metadata.version} ({format_label} format)")

    def _load_handler_function(self, handler_path: Path, handler_spec: str) -> Any:
        """
        Load handler function from Python module.

        Args:
            handler_path: Path to handler.py file
            handler_spec: Handler specification (e.g., "handler.function_name")

        Returns:
            Handler function

        Raises:
            SkillValidationError: If handler cannot be loaded
        """
        try:
            # Load module
            spec = importlib.util.spec_from_file_location("handler", handler_path)
            if spec is None or spec.loader is None:
                raise SkillValidationError(f"Failed to load module spec: {handler_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract function name from handler spec (e.g., "handler.execute")
            func_name = handler_spec.split(".")[-1]

            # Get handler function
            if not hasattr(module, func_name):
                raise SkillValidationError(
                    f"Handler function '{func_name}' not found in {handler_path}"
                )

            return getattr(module, func_name)

        except Exception as e:
            raise SkillValidationError(f"Failed to load handler from {handler_path}: {e}")

    def get_skill(self, skill_name: str) -> SkillMetadata:
        """
        Get skill metadata by name.

        Args:
            skill_name: Name of the skill

        Returns:
            Skill metadata

        Raises:
            SkillNotFoundError: If skill is not found
        """
        if skill_name not in self.skills:
            raise SkillNotFoundError(f"Skill not found: {skill_name}")

        return self.skills[skill_name]

    def get_handler(self, skill_name: str) -> Any:
        """
        Get skill handler function by name.

        Args:
            skill_name: Name of the skill

        Returns:
            Handler function

        Raises:
            SkillNotFoundError: If skill handler is not found
        """
        if skill_name not in self._handlers:
            raise SkillNotFoundError(f"Skill handler not found: {skill_name}")

        return self._handlers[skill_name]

    def list_skills(
        self, tags: Optional[List[str]] = None, safety_flag: Optional[SafetyFlag] = None
    ) -> List[SkillMetadata]:
        """
        List all registered skills with optional filtering.

        Args:
            tags: Filter by tags (returns skills matching any tag)
            safety_flag: Filter by safety flag

        Returns:
            List of skill metadata
        """
        skills = list(self.skills.values())

        if tags:
            skills = [s for s in skills if any(tag in s.tags for tag in tags)]

        if safety_flag:
            skills = [s for s in skills if safety_flag in s.safety_flags]

        return skills

    def validate_input(self, skill_name: str, inputs: Dict[str, Any]) -> None:
        """
        Validate input arguments against skill's input schema.

        Args:
            skill_name: Name of the skill
            inputs: Input arguments to validate

        Raises:
            SkillNotFoundError: If skill is not found
            SkillValidationError: If validation fails
        """
        skill = self.get_skill(skill_name)

        try:
            Draft7Validator(skill.input_schema).validate(inputs)
        except JsonSchemaValidationError as e:
            raise SkillValidationError(f"Input validation failed for {skill_name}: {e}")

    def validate_output(self, skill_name: str, output: Any) -> None:
        """
        Validate output result against skill's output schema.

        Args:
            skill_name: Name of the skill
            output: Output to validate

        Raises:
            SkillNotFoundError: If skill is not found
            SkillValidationError: If validation fails
        """
        skill = self.get_skill(skill_name)

        try:
            Draft7Validator(skill.output_schema).validate(output)
        except JsonSchemaValidationError as e:
            raise SkillValidationError(f"Output validation failed for {skill_name}: {e}")
