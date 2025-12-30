"""Tests for Git manager and version control."""

import pytest
import tempfile
import os
from pathlib import Path

from src.utils.git_manager import GitManager, GitError, GitStatus


class TestGitManager:
    """Tests for GitManager."""
    
    @pytest.fixture
    def git_repo(self):
        """Create a temporary Git repository."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize git repo
            os.system(f"cd {tmpdir} && git init")
            os.system(f"cd {tmpdir} && git config user.email 'test@test.com'")
            os.system(f"cd {tmpdir} && git config user.name 'Test'")
            
            # Create initial commit
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# Test file\n")
            os.system(f"cd {tmpdir} && git add . && git commit -m 'Initial commit'")
            
            yield tmpdir
    
    @pytest.fixture
    def manager(self, git_repo):
        return GitManager(git_repo)
    
    def test_get_status(self, manager, git_repo):
        """Test getting repository status."""
        status = manager.get_status()
        
        assert isinstance(status, GitStatus)
        assert status.is_clean
    
    def test_create_branch(self, manager):
        """Test branch creation."""
        branch = manager.create_evolution_branch("test-123")
        
        assert branch == "evolution/test-123"
        assert manager.get_current_branch() == branch
    
    def test_commit(self, manager, git_repo):
        """Test creating commits."""
        # Make a change
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Modified\n")
        
        commit_hash = manager.commit("Test commit")
        
        assert commit_hash
        assert len(commit_hash) == 40
    
    def test_rollback(self, manager, git_repo):
        """Test rollback functionality."""
        # Get initial commit
        initial = manager._run_git("rev-parse", "HEAD")
        
        # Make changes and commit
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Modified\n")
        manager.commit("Modification")
        
        # Rollback
        success = manager.rollback_to_commit(initial, hard=True)
        
        assert success
        assert test_file.read_text() == "# Test file\n"
    
    def test_commit_history(self, manager, git_repo):
        """Test getting commit history."""
        # Create some commits
        for i in range(3):
            test_file = Path(git_repo) / f"file{i}.py"
            test_file.write_text(f"# File {i}\n")
            manager.commit(f"Commit {i}")
        
        history = manager.get_commit_history(count=5)
        
        assert len(history) >= 3
        assert all(h.hash for h in history)
    
    def test_list_branches(self, manager):
        """Test listing branches."""
        # Get the default branch name
        default_branch = manager.get_current_branch()
        
        # Create a few branches
        manager.create_evolution_branch("test-1")
        manager.checkout_branch(default_branch)
        manager.create_evolution_branch("test-2")
        
        branches = manager.list_branches("evolution/*")
        
        assert len(branches) >= 2
        assert any(b.name == "evolution/test-1" for b in branches)
        assert any(b.name == "evolution/test-2" for b in branches)
    
    def test_stash_and_pop(self, manager, git_repo):
        """Test stashing changes."""
        # Make a change
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Stashed content\n")
        
        # Stash it
        success = manager.stash_changes("Test stash")
        assert success
        
        # File should be reverted
        assert test_file.read_text() == "# Test file\n"
        
        # Pop stash
        success = manager.pop_stash()
        assert success
        
        # File should have stashed content back
        assert test_file.read_text() == "# Stashed content\n"
    
    def test_get_diff(self, manager, git_repo):
        """Test getting diffs."""
        # Get initial commit
        initial = manager._run_git("rev-parse", "HEAD")
        
        # Make a change
        test_file = Path(git_repo) / "test.py"
        test_file.write_text("# Modified content\n")
        manager.commit("Change")
        
        # Get diff
        diff = manager.get_diff(from_ref=initial, to_ref="HEAD")
        
        assert "Modified content" in diff
        assert "Test file" in diff
    
    def test_tag_creation(self, manager):
        """Test creating tags."""
        manager.tag("v1.0.0", message="Release version 1.0.0")
        
        # Verify tag was created
        tags_output = manager._run_git("tag", "-l")
        assert "v1.0.0" in tags_output
