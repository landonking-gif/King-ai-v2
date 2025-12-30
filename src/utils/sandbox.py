
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from src.utils.code_patcher import CodePatcher

class Sandbox:
    """
    The Safety Net.
    Provides an isolated environment to test AI-generated code changes via pytest
    before they are ever applied to production.
    """
    
    def __init__(self, project_root: str = None):
        # Default to 3 levels up from this file (src/utils/sandbox.py -> root)
        if project_root:
             self.project_root = Path(project_root)
        else:
             self.project_root = Path(__file__).parent.parent.parent.absolute()
             
        self.temp_dir = None
        # Patcher will be initialized when sandbox is created
        self.patcher = None

    def create_sandbox(self, target_files: list[str]):
        """
        Creates a temp directory and copies specific target files + necessary context.
        Note: For a full test, we might need to copy the whole 'src' folder or use
        symlinks, but for safety/speed, we'll try to keep it minimal or copy 'src'.
        """
        self.temp_dir = Path(tempfile.mkdtemp(prefix="king_ai_sandbox_"))
        
        # Initialize patcher with the sandbox directory
        self.patcher = CodePatcher(str(self.temp_dir))
        
        # Determine strict copying vs full src copy
        # For Python imports to work reliably during tests, we usually need the whole 'src' structure.
        src_path = self.project_root / "src"
        if src_path.exists():
            shutil.copytree(src_path, self.temp_dir / "src")
        
        # Also copy tests folder
        tests_path = self.project_root / "tests"
        if tests_path.exists():
            shutil.copytree(tests_path, self.temp_dir / "tests")
            
        # Copy config needed?
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            shutil.copy(pyproject, self.temp_dir)
            
        return self.temp_dir

    def apply_patch(self, relative_file_path: str, new_code: str) -> bool:
        """
        Applies code change to the SANDBOXED version of the file.
        """
        if not self.temp_dir:
            raise RuntimeError("Sandbox not created. Call create_sandbox() first.")
        
        if not self.patcher:
            raise RuntimeError("Patcher not initialized. Call create_sandbox() first.")
            
        # Create and apply patch using new API
        try:
            patch = self.patcher.create_patch(
                relative_file_path,
                new_code,
                description="Sandbox test patch"
            )
            
            # Validate before applying
            is_valid, errors = self.patcher.validate_patch(patch)
            if not is_valid:
                print(f"Sandbox Patch Validation Failed: {', '.join(errors)}")
                return False
            
            # Apply the patch
            success = self.patcher.apply_patch(patch)
            return success
            
        except Exception as e:
            print(f"Sandbox Patch Failed: {e}")
            return False

    def run_tests(self) -> dict:
        """
        Runs the full test suite (or specific tests) inside the sandbox.
        Returns keys: 'success', 'output', 'exit_code'
        """
        if not self.temp_dir:
            raise RuntimeError("Sandbox not created.")
            
        try:
            # Run pytest in the temp dir
            # We add the temp dir to PYTHONPATH so it resolves 'src' correctly
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.temp_dir)
            
            result = subprocess.run(
                ["pytest", "tests"],
                cwd=str(self.temp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=60 # Prevent infinite loops
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout + "\n" + result.stderr,
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Tests timed out!",
                "exit_code": -1
            }
        except Exception as e:
             return {
                "success": False,
                "output": str(e),
                "exit_code": -1
            }

    def cleanup(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

