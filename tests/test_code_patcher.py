"""Tests for code patching and generation."""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.code_patcher import CodePatcher, CodePatch, PatchSet, PatchStatus
from src.utils.code_transformer import CodeTransformer


class TestCodePatcher:
    """Tests for CodePatcher."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("def hello():\n    return 'hello'\n")
            yield tmpdir
    
    @pytest.fixture
    def patcher(self, temp_project):
        return CodePatcher(temp_project)
    
    def test_create_patch(self, patcher, temp_project):
        """Test patch creation."""
        new_content = "def hello():\n    return 'world'\n"
        patch = patcher.create_patch("test.py", new_content)
        
        assert patch.file_path == "test.py"
        assert "hello" in patch.original_content
        assert patch.new_content == new_content
        assert patch.status == PatchStatus.PENDING
    
    def test_patch_diff(self, patcher, temp_project):
        """Test diff generation."""
        new_content = "def hello():\n    return 'world'\n"
        patch = patcher.create_patch("test.py", new_content)
        
        diff = patch.diff
        assert "-    return 'hello'" in diff
        assert "+    return 'world'" in diff
    
    def test_validate_patch(self, patcher, temp_project):
        """Test patch validation."""
        # Valid patch
        valid_content = "def hello():\n    return 'world'\n"
        patch = patcher.create_patch("test.py", valid_content)
        is_valid, errors = patcher.validate_patch(patch)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid syntax
        invalid_content = "def hello(\n    return 'world'\n"
        patch = patcher.create_patch("test.py", invalid_content)
        is_valid, errors = patcher.validate_patch(patch)
        assert not is_valid
        assert any("Syntax" in e for e in errors)
    
    def test_apply_patch(self, patcher, temp_project):
        """Test applying a patch."""
        new_content = "def hello():\n    return 'world'\n"
        patch = patcher.create_patch("test.py", new_content)
        
        success = patcher.apply_patch(patch)
        
        assert success
        assert patch.status == PatchStatus.APPLIED
        
        # Verify file was updated
        full_path = Path(temp_project) / "test.py"
        assert full_path.read_text() == new_content
    
    def test_patchset_atomic(self, patcher, temp_project):
        """Test atomic patchset application."""
        # Create another test file
        (Path(temp_project) / "test2.py").write_text("x = 1\n")
        
        patchset = PatchSet(id="test", description="Test patchset")
        patchset.add_patch(patcher.create_patch("test.py", "y = 2\n"))
        patchset.add_patch(patcher.create_patch("test2.py", "z = 3\n"))
        
        success = patcher.apply_patchset(patchset)
        
        assert success
        assert patchset.status == PatchStatus.APPLIED


class TestCodeTransformer:
    """Tests for CodeTransformer."""
    
    @pytest.fixture
    def transformer(self):
        return CodeTransformer()
    
    def test_add_import(self, transformer):
        """Test adding imports."""
        source = "def hello(): pass"
        result = transformer.add_import(source, "from typing import Optional")
        
        assert result.success
        assert "from typing import Optional" in result.code
    
    def test_add_decorator(self, transformer):
        """Test adding decorators."""
        source = "def hello(): pass"
        result = transformer.add_decorator(source, "hello", "staticmethod")
        
        assert result.success
        assert "@staticmethod" in result.code
    
    def test_rename_function(self, transformer):
        """Test renaming functions."""
        source = "def old_name(): pass\nold_name()"
        result = transformer.rename_function(source, "old_name", "new_name")
        
        assert result.success
        assert "def new_name" in result.code
        assert "new_name()" in result.code
        assert "old_name" not in result.code
