"""
Tests for Skill Parser and Format Converters.
Module: code-exec/service/test_skill_parser.py
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from skill_parser import (
    AnthropicSkillMetadata,
    FormatDetector,
    InvalidSkillFormatError,
    SkillConverter,
    SkillPackager,
    SkillParser,
)


class TestSkillParser:
    """Test SKILL.md parser."""

    def test_parse_valid_skill_md(self, tmp_path: Path) -> None:
        """Test parsing valid SKILL.md format."""
        skill_md = tmp_path / "SKILL.md"
        content = """---
name: test-skill
description: A test skill for validation
version: 1.0.0
dependencies: python>=3.8, requests>=2.0.0
tags:
  - testing
  - example
---

# Test Skill

This is a test skill for validation purposes.

## When to Use

Use this skill when testing the parser.

## How to Use

1. Parse the SKILL.md file
2. Validate the metadata
3. Check the markdown content
"""
        skill_md.write_text(content)

        metadata, markdown = SkillParser.parse_skill_md(skill_md)

        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill for validation"
        assert metadata.version == "1.0.0"
        assert metadata.dependencies == ["python>=3.8", "requests>=2.0.0"]
        assert metadata.tags == ["testing", "example"]
        assert "This is a test skill" in markdown

    def test_parse_skill_md_missing_frontmatter(self, tmp_path: Path) -> None:
        """Test that parser rejects SKILL.md without frontmatter."""
        skill_md = tmp_path / "SKILL.md"
        content = """# Test Skill

This has no frontmatter.
"""
        skill_md.write_text(content)

        with pytest.raises(InvalidSkillFormatError, match="must start with YAML frontmatter"):
            SkillParser.parse_skill_md(skill_md)

    def test_parse_skill_md_missing_name(self, tmp_path: Path) -> None:
        """Test that parser rejects SKILL.md without name field."""
        skill_md = tmp_path / "SKILL.md"
        content = """---
description: A test skill
---

# Test
"""
        skill_md.write_text(content)

        with pytest.raises(InvalidSkillFormatError, match="must have 'name' field"):
            SkillParser.parse_skill_md(skill_md)

    def test_parse_skill_md_name_too_long(self, tmp_path: Path) -> None:
        """Test that parser rejects names exceeding 64 characters."""
        skill_md = tmp_path / "SKILL.md"
        long_name = "a" * 65
        content = f"""---
name: {long_name}
description: Test
---

# Test
"""
        skill_md.write_text(content)

        with pytest.raises(InvalidSkillFormatError, match="must be string <= 64 characters"):
            SkillParser.parse_skill_md(skill_md)

    def test_parse_skill_md_description_too_long(self, tmp_path: Path) -> None:
        """Test that parser rejects descriptions exceeding 200 characters."""
        skill_md = tmp_path / "SKILL.md"
        long_desc = "a" * 201
        content = f"""---
name: test
description: {long_desc}
---

# Test
"""
        skill_md.write_text(content)

        with pytest.raises(InvalidSkillFormatError, match="must be string <= 200 chars"):
            SkillParser.parse_skill_md(skill_md)

    def test_parse_skill_md_file_too_large(self, tmp_path: Path) -> None:
        """Test that parser rejects files exceeding 500KB."""
        skill_md = tmp_path / "SKILL.md"
        # Create content > 500KB
        large_content = """---
name: test
description: Test
---

# Test

""" + (
            "x" * 600000
        )
        skill_md.write_text(large_content)

        with pytest.raises(InvalidSkillFormatError, match="exceeds 500KB limit"):
            SkillParser.parse_skill_md(skill_md)

    def test_load_skill_metadata_only(self, tmp_path: Path) -> None:
        """Test lightweight metadata loading (progressive disclosure Level 1)."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        content = """---
name: test-skill
description: A test skill
---

# Test Skill
"""
        skill_md.write_text(content)

        metadata = SkillParser.load_skill_metadata_only(skill_dir)

        assert metadata["name"] == "test-skill"
        assert metadata["description"] == "A test skill"
        assert metadata["format"] == "anthropic"


class TestFormatDetector:
    """Test format detection."""

    def test_detect_native_format(self, tmp_path: Path) -> None:
        """Test detection of native format."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "skill.yaml").write_text("name: test")
        (skill_dir / "schema.json").write_text("{}")
        (skill_dir / "handler.py").write_text("def test(): pass")

        format_info = FormatDetector.detect_format(skill_dir)

        assert format_info.is_native_only
        assert not format_info.is_anthropic_only
        assert not format_info.is_hybrid
        assert format_info.has_skill_yaml
        assert format_info.has_schema_json
        assert format_info.has_handler_py
        assert not format_info.has_skill_md

    def test_detect_anthropic_format(self, tmp_path: Path) -> None:
        """Test detection of Anthropic format."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n\n# Test")

        format_info = FormatDetector.detect_format(skill_dir)

        assert format_info.is_anthropic_only
        assert not format_info.is_native_only
        assert not format_info.is_hybrid
        assert format_info.has_skill_md
        assert not format_info.has_skill_yaml

    def test_detect_hybrid_format(self, tmp_path: Path) -> None:
        """Test detection of hybrid format."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n\n# Test")
        (skill_dir / "skill.yaml").write_text("name: test")
        (skill_dir / "schema.json").write_text("{}")

        format_info = FormatDetector.detect_format(skill_dir)

        assert format_info.is_hybrid
        assert not format_info.is_native_only
        assert not format_info.is_anthropic_only
        assert format_info.has_skill_md
        assert format_info.has_skill_yaml
        assert format_info.has_schema_json


class TestSkillConverter:
    """Test skill format conversions."""

    def test_native_to_anthropic(self, tmp_path: Path) -> None:
        """Test converting native format to Anthropic SKILL.md."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        # Create native format skill
        skill_config = {
            "name": "test-skill",
            "version": "1.0.0",
            "description": "A test skill",
            "safety_flags": ["none"],
            "requires_approval": False,
            "handler": "handler.test_skill",
            "tags": ["testing"],
        }

        (skill_dir / "skill.yaml").write_text(yaml.dump(skill_config))

        schema = {
            "input": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Input text"},
                },
                "required": ["text"],
            },
            "output": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Output result"},
                },
                "required": ["result"],
            },
        }

        (skill_dir / "schema.json").write_text(json.dumps(schema, indent=2))

        # Convert to Anthropic
        skill_md_path = SkillConverter.native_to_anthropic(skill_dir, include_handler=False)

        assert skill_md_path.exists()

        # Parse and validate
        metadata, markdown = SkillParser.parse_skill_md(skill_md_path)

        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill"
        assert metadata.version == "1.0.0"
        assert "Input Schema" in markdown
        assert "Output Schema" in markdown

    def test_anthropic_to_native(self, tmp_path: Path) -> None:
        """Test converting Anthropic SKILL.md to native format."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        # Create SKILL.md
        content = """---
name: test-skill
description: A test skill for conversion
version: 1.0.0
dependencies: python>=3.8
---

# Test Skill

This skill tests conversion from Anthropic to native format.
"""
        (skill_dir / "SKILL.md").write_text(content)

        # Convert to native
        from models import SafetyFlag

        metadata = SkillConverter.anthropic_to_native(
            skill_dir, add_safety_flags=[SafetyFlag.NONE], requires_approval=False
        )

        assert metadata.name == "test-skill"
        assert metadata.description == "A test skill for conversion"

        # Check generated files
        assert (skill_dir / "skill.yaml").exists()
        assert (skill_dir / "schema.json").exists()
        assert (skill_dir / "handler.py").exists()

        # Validate generated skill.yaml
        with open(skill_dir / "skill.yaml", "r") as f:
            config = yaml.safe_load(f)
            assert config["name"] == "test-skill"
            assert config["anthropic_compatible"] is True


class TestSkillPackager:
    """Test skill packaging."""

    def test_package_skill_zip(self, tmp_path: Path) -> None:
        """Test packaging skill as ZIP."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()

        # Create skill files
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n\n# Test")
        (skill_dir / "skill.yaml").write_text("name: test")
        (skill_dir / "handler.py").write_text("def test(): pass")

        # Package
        zip_path = SkillPackager.package_skill_zip(skill_dir)

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

    def test_unpack_skill_zip(self, tmp_path: Path) -> None:
        """Test unpacking skill ZIP."""
        # Create skill directory
        skill_dir = tmp_path / "source" / "test-skill"
        skill_dir.mkdir(parents=True)

        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: test\n---\n\n# Test")
        (skill_dir / "handler.py").write_text("def test(): pass")

        # Package
        zip_path = SkillPackager.package_skill_zip(skill_dir)

        # Unpack to different location
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        extracted_dir = SkillPackager.unpack_skill_zip(zip_path, target_dir)

        assert extracted_dir.exists()
        assert (extracted_dir / "SKILL.md").exists()
        assert (extracted_dir / "handler.py").exists()


class TestIntegration:
    """Integration tests for full workflow."""

    def test_hybrid_skill_creation(self, tmp_path: Path) -> None:
        """Test creating a hybrid skill (both formats)."""
        skill_dir = tmp_path / "hybrid-skill"
        skill_dir.mkdir()

        # Create native format
        skill_config = {
            "name": "hybrid-skill",
            "version": "1.0.0",
            "description": "A hybrid format skill",
            "safety_flags": ["none"],
            "requires_approval": False,
            "handler": "handler.hybrid_skill",
        }

        (skill_dir / "skill.yaml").write_text(yaml.dump(skill_config))

        schema = {
            "input": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            "output": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        }

        (skill_dir / "schema.json").write_text(json.dumps(schema))
        (skill_dir / "handler.py").write_text("def hybrid_skill(input): return {'result': input}")

        # Add Anthropic format
        SkillConverter.native_to_anthropic(skill_dir, include_handler=True)

        # Verify hybrid format
        format_info = FormatDetector.detect_format(skill_dir)

        assert format_info.is_hybrid
        assert format_info.has_skill_md
        assert format_info.has_skill_yaml
        assert format_info.has_schema_json
        assert format_info.has_handler_py

    def test_import_export_roundtrip(self, tmp_path: Path) -> None:
        """Test importing and exporting a skill."""
        # Create original skill
        original_dir = tmp_path / "original" / "test-skill"
        original_dir.mkdir(parents=True)

        (original_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: test\n---\n\n# Test"
        )
        (original_dir / "handler.py").write_text("def test(): pass")

        # Package
        zip_path = SkillPackager.package_skill_zip(original_dir)

        # Unpack to new location
        import_dir = tmp_path / "imported"
        import_dir.mkdir()

        extracted_dir = SkillPackager.unpack_skill_zip(zip_path, import_dir)

        # Verify imported skill
        assert (extracted_dir / "SKILL.md").exists()
        assert (extracted_dir / "handler.py").exists()

        # Re-export
        zip_path2 = SkillPackager.package_skill_zip(extracted_dir)

        assert zip_path2.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
