# King AI v2 - Implementation Plan Part 7
## Evolution Engine - Code Patching & Generation

**Target Timeline:** Week 5-6
**Objective:** Implement safe code patching and generation systems for self-modification.

---

## Overview of All Parts

| Part | Title | Status |
|------|-------|--------|
| 1 | Infrastructure Layer & Core System Hardening | ✅ Complete |
| 2 | Master AI Brain - Core Enhancements | ✅ Complete |
| 3 | Master AI Brain - Context & Memory System | ✅ Complete |
| 4 | Master AI Brain - Planning & ReAct Implementation | ✅ Complete |
| 5 | Evolution Engine - Core Models & Proposal System | ✅ Complete |
| 6 | Evolution Engine - Code Analysis & AST Tools | ✅ Complete |
| **7** | **Evolution Engine - Code Patching & Generation** | 🔄 Current |
| 8 | Evolution Engine - Git Integration & Rollback | ⏳ Pending |
| 9 | Evolution Engine - Sandbox Testing | ⏳ Pending |
| 10 | Sub-Agent: Research (Web/API) | ⏳ Pending |
| 11 | Sub-Agent: Code Generator | ⏳ Pending |
| 12 | Sub-Agent: Content (Blog/SEO) | ⏳ Pending |
| 13 | Sub-Agent: Commerce - Shopify | ⏳ Pending |
| 14 | Sub-Agent: Commerce - Suppliers | ⏳ Pending |
| 15 | Sub-Agent: Finance - Stripe | ⏳ Pending |
| 16 | Sub-Agent: Finance - Plaid/Banking | ⏳ Pending |
| 17 | Sub-Agent: Analytics | ⏳ Pending |
| 18 | Sub-Agent: Legal | ⏳ Pending |
| 19 | Business: Lifecycle Engine | ⏳ Pending |
| 20 | Business: Playbook System | ⏳ Pending |
| 21 | Business: Portfolio Management | ⏳ Pending |
| 22 | Dashboard: React Components | ⏳ Pending |
| 23 | Dashboard: Approval Workflows | ⏳ Pending |
| 24 | Dashboard: WebSocket & Monitoring | ⏳ Pending |

---

## Part 7 Scope

This part focuses on:
1. Safe code patching with diff generation
2. LLM-based code generation for improvements
3. Code transformation utilities
4. Patch validation before applying
5. Atomic file operations with backup

---

## Task 7.1: Create Code Patcher Utility

**File:** `src/utils/code_patcher.py` (REPLACE EXISTING FILE)

```python
"""
Code Patcher - Safe code modification with diff generation and rollback.
Handles atomic file operations with backup capabilities.
"""

import os
import shutil
import difflib
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.utils.ast_parser import ASTParser
from src.utils.structured_logging import get_logger

logger = get_logger("code_patcher")


class PatchStatus(str, Enum):
    """Status of a patch operation."""
    PENDING = "pending"
    VALIDATED = "validated"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class CodePatch:
    """Represents a single code patch."""
    file_path: str
    original_content: str
    new_content: str
    description: str = ""
    status: PatchStatus = PatchStatus.PENDING
    applied_at: Optional[datetime] = None
    error: Optional[str] = None
    
    @property
    def diff(self) -> str:
        """Generate unified diff."""
        original_lines = self.original_content.splitlines(keepends=True)
        new_lines = self.new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}"
        )
        return ''.join(diff)
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get addition/deletion stats."""
        original_lines = self.original_content.splitlines()
        new_lines = self.new_content.splitlines()
        
        matcher = difflib.SequenceMatcher(None, original_lines, new_lines)
        
        additions = 0
        deletions = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                deletions += i2 - i1
                additions += j2 - j1
            elif tag == 'delete':
                deletions += i2 - i1
            elif tag == 'insert':
                additions += j2 - j1
        
        return {
            "additions": additions,
            "deletions": deletions,
            "total_changes": additions + deletions
        }


@dataclass
class PatchSet:
    """Collection of patches to apply atomically."""
    id: str
    description: str
    patches: List[CodePatch] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    status: PatchStatus = PatchStatus.PENDING
    backup_dir: Optional[str] = None
    
    def add_patch(self, patch: CodePatch):
        """Add a patch to the set."""
        self.patches.append(patch)
    
    @property
    def total_stats(self) -> Dict[str, int]:
        """Get total stats across all patches."""
        total = {"additions": 0, "deletions": 0, "total_changes": 0, "files": len(self.patches)}
        for patch in self.patches:
            stats = patch.stats
            total["additions"] += stats["additions"]
            total["deletions"] += stats["deletions"]
            total["total_changes"] += stats["total_changes"]
        return total


class CodePatcher:
    """
    Handles safe code modifications with validation and rollback.
    """
    
    def __init__(self, project_root: str, backup_dir: str = None):
        """
        Initialize the patcher.
        
        Args:
            project_root: Root directory of the project
            backup_dir: Directory for backups (default: .king_ai_backups)
        """
        self.project_root = Path(project_root)
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / ".king_ai_backups"
        self.parser = ASTParser()
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_patch(
        self,
        file_path: str,
        new_content: str,
        description: str = ""
    ) -> CodePatch:
        """
        Create a patch for a file.
        
        Args:
            file_path: Relative path to the file
            new_content: New content for the file
            description: Description of the change
            
        Returns:
            CodePatch object
        """
        full_path = self.project_root / file_path
        
        # Read original content
        if full_path.exists():
            original_content = full_path.read_text(encoding='utf-8')
        else:
            original_content = ""
        
        return CodePatch(
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            description=description
        )
    
    def create_patch_from_function(
        self,
        file_path: str,
        function_name: str,
        new_function_code: str
    ) -> CodePatch:
        """
        Create a patch that replaces a specific function.
        
        Args:
            file_path: Path to the file
            function_name: Name of the function to replace
            new_function_code: New function implementation
            
        Returns:
            CodePatch object
        """
        full_path = self.project_root / file_path
        original_content = full_path.read_text(encoding='utf-8')
        
        # Parse the file to find the function
        structure = self.parser.parse_source(original_content, file_path)
        func = structure.get_function(function_name)
        
        if not func:
            raise ValueError(f"Function '{function_name}' not found in {file_path}")
        
        # Replace the function
        lines = original_content.splitlines(keepends=True)
        
        # Get the function lines
        before = lines[:func.lineno - 1]
        after = lines[func.end_lineno:]
        
        # Ensure new function has proper newline
        if not new_function_code.endswith('\n'):
            new_function_code += '\n'
        
        new_content = ''.join(before) + new_function_code + ''.join(after)
        
        return CodePatch(
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            description=f"Replace function: {function_name}"
        )
    
    def validate_patch(self, patch: CodePatch) -> Tuple[bool, List[str]]:
        """
        Validate a patch before applying.
        
        Args:
            patch: The patch to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check syntax
        if patch.file_path.endswith('.py'):
            is_valid, error = self.parser.validate_syntax(patch.new_content)
            if not is_valid:
                errors.append(f"Syntax error: {error}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ('os.system(', "Direct system command execution"),
            ('subprocess.call(', "Subprocess call without validation"),
            ('eval(', "Eval usage is dangerous"),
            ('exec(', "Exec usage is dangerous"),
            ('__import__(', "Dynamic import"),
            ('open(', None),  # Check if writing to sensitive files
        ]
        
        for pattern, message in dangerous_patterns:
            if pattern in patch.new_content and pattern not in patch.original_content:
                if message:
                    errors.append(f"Dangerous pattern: {message}")
        
        # Check file path safety
        if '..' in patch.file_path:
            errors.append("Path traversal detected in file path")
        
        sensitive_files = ['settings.py', '.env', 'secrets', 'credentials']
        if any(s in patch.file_path.lower() for s in sensitive_files):
            errors.append("Modification of sensitive file requires extra review")
        
        patch.status = PatchStatus.VALIDATED if not errors else PatchStatus.PENDING
        return len(errors) == 0, errors
    
    def validate_patchset(self, patchset: PatchSet) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate all patches in a set.
        
        Returns:
            Tuple of (all_valid, dict of file_path -> errors)
        """
        all_errors = {}
        all_valid = True
        
        for patch in patchset.patches:
            valid, errors = self.validate_patch(patch)
            if not valid:
                all_valid = False
                all_errors[patch.file_path] = errors
        
        patchset.status = PatchStatus.VALIDATED if all_valid else PatchStatus.PENDING
        return all_valid, all_errors
    
    def apply_patch(self, patch: CodePatch, create_backup: bool = True) -> bool:
        """
        Apply a single patch.
        
        Args:
            patch: The patch to apply
            create_backup: Whether to backup original file
            
        Returns:
            Success status
        """
        try:
            full_path = self.project_root / patch.file_path
            
            # Create backup
            if create_backup and full_path.exists():
                backup_path = self._create_backup(patch.file_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write new content
            full_path.write_text(patch.new_content, encoding='utf-8')
            
            patch.status = PatchStatus.APPLIED
            patch.applied_at = datetime.now()
            
            logger.info(f"Applied patch to {patch.file_path}", stats=patch.stats)
            return True
            
        except Exception as e:
            patch.status = PatchStatus.FAILED
            patch.error = str(e)
            logger.error(f"Failed to apply patch: {e}")
            return False
    
    def apply_patchset(self, patchset: PatchSet) -> bool:
        """
        Apply all patches in a set atomically.
        If any patch fails, rollback all applied patches.
        
        Args:
            patchset: The patchset to apply
            
        Returns:
            Success status
        """
        # Create a dedicated backup directory for this patchset
        backup_subdir = self.backup_dir / f"patchset_{patchset.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_subdir.mkdir(parents=True, exist_ok=True)
        patchset.backup_dir = str(backup_subdir)
        
        applied_patches = []
        
        try:
            for patch in patchset.patches:
                # Backup to patchset-specific directory
                full_path = self.project_root / patch.file_path
                if full_path.exists():
                    backup_path = backup_subdir / patch.file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, backup_path)
                
                # Apply the patch
                success = self.apply_patch(patch, create_backup=False)
                
                if not success:
                    raise RuntimeError(f"Failed to apply patch to {patch.file_path}: {patch.error}")
                
                applied_patches.append(patch)
            
            patchset.status = PatchStatus.APPLIED
            patchset.applied_at = datetime.now()
            
            logger.info(f"Applied patchset {patchset.id}", stats=patchset.total_stats)
            return True
            
        except Exception as e:
            logger.error(f"Patchset failed, rolling back: {e}")
            
            # Rollback all applied patches
            for patch in reversed(applied_patches):
                self._rollback_patch(patch, backup_subdir)
            
            patchset.status = PatchStatus.FAILED
            return False
    
    def _create_backup(self, file_path: str) -> Path:
        """Create a backup of a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{Path(file_path).name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        
        full_path = self.project_root / file_path
        shutil.copy2(full_path, backup_path)
        
        return backup_path
    
    def _rollback_patch(self, patch: CodePatch, backup_dir: Path):
        """Rollback a single patch from backup."""
        try:
            backup_path = backup_dir / patch.file_path
            full_path = self.project_root / patch.file_path
            
            if backup_path.exists():
                shutil.copy2(backup_path, full_path)
                patch.status = PatchStatus.ROLLED_BACK
                logger.info(f"Rolled back: {patch.file_path}")
            else:
                # File was newly created, delete it
                if full_path.exists():
                    full_path.unlink()
                patch.status = PatchStatus.ROLLED_BACK
                
        except Exception as e:
            logger.error(f"Rollback failed for {patch.file_path}: {e}")
    
    def rollback_patchset(self, patchset: PatchSet) -> bool:
        """
        Rollback an applied patchset.
        
        Args:
            patchset: The patchset to rollback
            
        Returns:
            Success status
        """
        if not patchset.backup_dir:
            logger.error("No backup directory for patchset")
            return False
        
        backup_dir = Path(patchset.backup_dir)
        
        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {backup_dir}")
            return False
        
        success = True
        for patch in patchset.patches:
            try:
                self._rollback_patch(patch, backup_dir)
            except Exception as e:
                logger.error(f"Failed to rollback {patch.file_path}: {e}")
                success = False
        
        if success:
            patchset.status = PatchStatus.ROLLED_BACK
        
        return success
    
    def generate_diff_report(self, patchset: PatchSet) -> str:
        """Generate a human-readable diff report."""
        report = []
        report.append(f"# Patch Report: {patchset.description}")
        report.append(f"ID: {patchset.id}")
        report.append(f"Created: {patchset.created_at}")
        report.append(f"Status: {patchset.status.value}")
        report.append("")
        
        stats = patchset.total_stats
        report.append(f"## Summary")
        report.append(f"- Files changed: {stats['files']}")
        report.append(f"- Additions: +{stats['additions']}")
        report.append(f"- Deletions: -{stats['deletions']}")
        report.append("")
        
        for patch in patchset.patches:
            report.append(f"## {patch.file_path}")
            report.append(f"Description: {patch.description}")
            report.append("```diff")
            report.append(patch.diff)
            report.append("```")
            report.append("")
        
        return '\n'.join(report)
```

---

## Task 7.2: Create Code Generator

**File:** `src/utils/code_generator.py` (CREATE NEW FILE)

```python
"""
Code Generator - LLM-based code generation for system improvements.
Generates safe, validated code modifications.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from src.utils.llm_router import LLMRouter, TaskContext
from src.utils.ast_parser import ASTParser, CodeStructure
from src.utils.code_patcher import CodePatch, CodePatcher
from src.utils.structured_logging import get_logger
from src.utils.retry import with_retry, LLM_RETRY_CONFIG

logger = get_logger("code_generator")


@dataclass
class GenerationRequest:
    """Request for code generation."""
    goal: str
    file_path: str
    context: str = ""
    constraints: List[str] = None
    existing_code: Optional[str] = None
    function_name: Optional[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


@dataclass
class GenerationResult:
    """Result of code generation."""
    success: bool
    code: Optional[str] = None
    explanation: str = ""
    patch: Optional[CodePatch] = None
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


CODE_GENERATION_PROMPT = '''You are an expert Python developer. Generate code based on the following request.

GOAL: {goal}

FILE: {file_path}

EXISTING CODE:
```python
{existing_code}
```

CONTEXT: {context}

CONSTRAINTS:
{constraints}

REQUIREMENTS:
1. Write clean, well-documented Python code
2. Include type hints for all functions
3. Include docstrings for all public functions
4. Follow PEP 8 style guidelines
5. Handle errors appropriately
6. Do NOT use dangerous functions (eval, exec, os.system)

Respond with JSON:
{{
    "code": "your generated code here",
    "explanation": "brief explanation of what the code does",
    "imports_needed": ["list", "of", "imports"]
}}
'''


FUNCTION_IMPROVEMENT_PROMPT = '''You are an expert Python developer. Improve this function.

FUNCTION TO IMPROVE:
```python
{function_code}
```

IMPROVEMENT GOAL: {goal}

CONTEXT: {context}

REQUIREMENTS:
1. Keep the same function signature (name and parameters)
2. Improve based on the goal while maintaining functionality
3. Add proper type hints and docstring
4. Handle edge cases
5. Follow best practices

Respond with JSON:
{{
    "improved_code": "the improved function code",
    "changes_made": ["list of changes"],
    "explanation": "why these changes improve the code"
}}
'''


class CodeGenerator:
    """
    Generates code using LLM with validation and safety checks.
    """
    
    def __init__(self, llm_router: LLMRouter, project_root: str):
        """
        Initialize the generator.
        
        Args:
            llm_router: LLM router for inference
            project_root: Root directory of the project
        """
        self.llm = llm_router
        self.project_root = project_root
        self.parser = ASTParser()
        self.patcher = CodePatcher(project_root)
    
    @with_retry(LLM_RETRY_CONFIG)
    async def generate_code(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate code based on a request.
        
        Args:
            request: Generation request details
            
        Returns:
            Generation result with code and patch
        """
        logger.info(f"Generating code for: {request.goal[:50]}...")
        
        # Read existing code if not provided
        if request.existing_code is None:
            try:
                full_path = f"{self.project_root}/{request.file_path}"
                with open(full_path, 'r') as f:
                    request.existing_code = f.read()
            except FileNotFoundError:
                request.existing_code = "# New file"
        
        # Build prompt
        prompt = CODE_GENERATION_PROMPT.format(
            goal=request.goal,
            file_path=request.file_path,
            existing_code=request.existing_code[:3000],
            context=request.context[:1000],
            constraints='\n'.join(f"- {c}" for c in request.constraints)
        )
        
        # Generate with LLM
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=2000,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        # Parse response
        try:
            data = self._parse_json_response(response)
            generated_code = data.get("code", "")
            explanation = data.get("explanation", "")
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"Failed to parse LLM response: {e}"
            )
        
        # Validate the generated code
        is_valid, errors = self._validate_code(generated_code, request.file_path)
        
        if not is_valid:
            return GenerationResult(
                success=False,
                code=generated_code,
                explanation=explanation,
                validation_errors=errors
            )
        
        # Create a patch
        patch = self.patcher.create_patch(
            file_path=request.file_path,
            new_content=generated_code,
            description=request.goal
        )
        
        return GenerationResult(
            success=True,
            code=generated_code,
            explanation=explanation,
            patch=patch
        )
    
    @with_retry(LLM_RETRY_CONFIG)
    async def improve_function(
        self,
        file_path: str,
        function_name: str,
        goal: str,
        context: str = ""
    ) -> GenerationResult:
        """
        Improve a specific function.
        
        Args:
            file_path: Path to the file
            function_name: Name of function to improve
            goal: Improvement goal
            context: Additional context
            
        Returns:
            Generation result
        """
        logger.info(f"Improving function: {function_name}")
        
        # Read and parse the file
        full_path = f"{self.project_root}/{file_path}"
        with open(full_path, 'r') as f:
            source = f.read()
        
        structure = self.parser.parse_source(source, file_path)
        func = structure.get_function(function_name)
        
        if not func:
            return GenerationResult(
                success=False,
                explanation=f"Function '{function_name}' not found"
            )
        
        # Extract function code
        function_code = self.parser.get_line_range(
            source,
            func.lineno,
            func.end_lineno
        )
        
        # Generate improvement
        prompt = FUNCTION_IMPROVEMENT_PROMPT.format(
            function_code=function_code,
            goal=goal,
            context=context
        )
        
        llm_context = TaskContext(
            task_type="code_generation",
            risk_level="medium",
            requires_accuracy=True,
            token_estimate=1500,
            priority="high"
        )
        
        response = await self.llm.complete(prompt, context=llm_context)
        
        try:
            data = self._parse_json_response(response)
            improved_code = data.get("improved_code", "")
            changes = data.get("changes_made", [])
            explanation = data.get("explanation", "")
        except Exception as e:
            return GenerationResult(
                success=False,
                explanation=f"Failed to parse response: {e}"
            )
        
        # Validate improved code
        is_valid, errors = self._validate_function(improved_code)
        
        if not is_valid:
            return GenerationResult(
                success=False,
                code=improved_code,
                explanation=explanation,
                validation_errors=errors
            )
        
        # Create patch for function replacement
        try:
            patch = self.patcher.create_patch_from_function(
                file_path=file_path,
                function_name=function_name,
                new_function_code=improved_code
            )
        except Exception as e:
            return GenerationResult(
                success=False,
                code=improved_code,
                explanation=f"Failed to create patch: {e}"
            )
        
        return GenerationResult(
            success=True,
            code=improved_code,
            explanation=f"{explanation}\n\nChanges: {', '.join(changes)}",
            patch=patch
        )
    
    async def add_method_to_class(
        self,
        file_path: str,
        class_name: str,
        method_signature: str,
        method_description: str
    ) -> GenerationResult:
        """
        Add a new method to an existing class.
        
        Args:
            file_path: Path to the file
            class_name: Name of the class
            method_signature: Signature like "async def process(self, data: str) -> bool"
            method_description: What the method should do
            
        Returns:
            Generation result
        """
        request = GenerationRequest(
            goal=f"Add method to class {class_name}: {method_signature}\nDescription: {method_description}",
            file_path=file_path,
            constraints=[
                f"Add the method inside the {class_name} class",
                "Follow existing class patterns",
                "Include proper type hints and docstring"
            ]
        )
        
        return await self.generate_code(request)
    
    def _validate_code(self, code: str, file_path: str) -> Tuple[bool, List[str]]:
        """Validate generated code."""
        errors = []
        
        if not code.strip():
            errors.append("Empty code generated")
            return False, errors
        
        # Syntax check for Python files
        if file_path.endswith('.py'):
            is_valid, error = self.parser.validate_syntax(code)
            if not is_valid:
                errors.append(f"Syntax error: {error}")
        
        # Check for dangerous patterns
        dangerous = [
            ('eval(', "eval() is dangerous"),
            ('exec(', "exec() is dangerous"),
            ('os.system(', "os.system() is dangerous"),
            ('__import__(', "dynamic import is dangerous"),
        ]
        
        for pattern, message in dangerous:
            if pattern in code:
                errors.append(message)
        
        return len(errors) == 0, errors
    
    def _validate_function(self, code: str) -> Tuple[bool, List[str]]:
        """Validate a function code snippet."""
        errors = []
        
        if not code.strip():
            errors.append("Empty function code")
            return False, errors
        
        # Check it's a valid function definition
        if not re.match(r'^\s*(async\s+)?def\s+\w+', code):
            errors.append("Code does not appear to be a function definition")
        
        # Validate syntax
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        
        return len(errors) == 0, errors
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        # Clean up common issues
        response = response.strip()
        
        return json.loads(response)
```

---

## Task 7.3: Create Code Transformer Utilities

**File:** `src/utils/code_transformer.py` (CREATE NEW FILE)

```python
"""
Code Transformer - AST-based code transformations.
Provides safe, structural code modifications.
"""

import ast
import astor  # pip install astor
from typing import Optional, List, Callable, Any
from dataclasses import dataclass

from src.utils.structured_logging import get_logger

logger = get_logger("code_transformer")


@dataclass
class TransformResult:
    """Result of a code transformation."""
    success: bool
    code: str = ""
    changes_made: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.changes_made is None:
            self.changes_made = []


class AddImportTransformer(ast.NodeTransformer):
    """AST transformer that adds imports."""
    
    def __init__(self, imports_to_add: List[str]):
        self.imports_to_add = imports_to_add
        self.existing_imports = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            self.existing_imports.add(alias.name)
        return node
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.existing_imports.add(node.module)
        return node


class AddDecoratorTransformer(ast.NodeTransformer):
    """AST transformer that adds decorators to functions."""
    
    def __init__(self, function_name: str, decorator: str):
        self.function_name = function_name
        self.decorator = decorator
        self.modified = False
    
    def visit_FunctionDef(self, node):
        if node.name == self.function_name:
            # Add decorator
            decorator_node = ast.Name(id=self.decorator, ctx=ast.Load())
            node.decorator_list.insert(0, decorator_node)
            self.modified = True
        return node
    
    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)


class TypeHintAdder(ast.NodeTransformer):
    """AST transformer that adds type hints to functions."""
    
    def __init__(self, type_hints: dict):
        """
        Args:
            type_hints: Dict mapping arg names to type strings
        """
        self.type_hints = type_hints
        self.modified = False
    
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg in self.type_hints and not arg.annotation:
                arg.annotation = ast.parse(self.type_hints[arg.arg], mode='eval').body
                self.modified = True
        return node


class CodeTransformer:
    """
    High-level code transformation utilities.
    Uses AST for safe, structural modifications.
    """
    
    def add_import(self, source: str, import_statement: str) -> TransformResult:
        """
        Add an import statement to source code.
        
        Args:
            source: Original source code
            import_statement: Import to add (e.g., "from typing import Optional")
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            
            # Check if import already exists
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if import_statement.endswith(alias.name):
                            return TransformResult(
                                success=True,
                                code=source,
                                changes_made=["Import already exists"]
                            )
            
            # Parse the import statement
            import_node = ast.parse(import_statement).body[0]
            
            # Find the right position (after existing imports)
            insert_pos = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    # Skip docstrings
                    insert_pos = i + 1
                else:
                    break
            
            tree.body.insert(insert_pos, import_node)
            
            new_code = astor.to_source(tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=[f"Added import: {import_statement}"]
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def add_decorator(
        self,
        source: str,
        function_name: str,
        decorator: str
    ) -> TransformResult:
        """
        Add a decorator to a function.
        
        Args:
            source: Original source code
            function_name: Name of function to decorate
            decorator: Decorator to add (without @)
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            
            transformer = AddDecoratorTransformer(function_name, decorator)
            new_tree = transformer.visit(tree)
            
            if not transformer.modified:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=[f"Added @{decorator} to {function_name}"]
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def rename_function(
        self,
        source: str,
        old_name: str,
        new_name: str
    ) -> TransformResult:
        """
        Rename a function and update all calls.
        
        Args:
            source: Original source code
            old_name: Current function name
            new_name: New function name
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            changes = []
            
            class RenameTransformer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == old_name:
                        node.name = new_name
                        changes.append(f"Renamed function definition: {old_name} -> {new_name}")
                    self.generic_visit(node)
                    return node
                
                def visit_AsyncFunctionDef(self, node):
                    return self.visit_FunctionDef(node)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name) and node.func.id == old_name:
                        node.func.id = new_name
                        changes.append(f"Updated call: {old_name}() -> {new_name}()")
                    self.generic_visit(node)
                    return node
            
            transformer = RenameTransformer()
            new_tree = transformer.visit(tree)
            
            if not changes:
                return TransformResult(
                    success=False,
                    error=f"Function '{old_name}' not found"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=changes
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def extract_function(
        self,
        source: str,
        function_name: str,
        start_line: int,
        end_line: int,
        new_function_name: str
    ) -> TransformResult:
        """
        Extract lines from a function into a new function.
        
        Args:
            source: Original source code
            function_name: Function to extract from
            start_line: Start line of code to extract (relative to function)
            end_line: End line of code to extract
            new_function_name: Name for the new function
            
        Returns:
            Transformation result with extracted function
        """
        # This is a complex refactoring - simplified version
        try:
            lines = source.split('\n')
            
            # Find the function
            tree = ast.parse(source)
            target_func = None
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        target_func = node
                        break
            
            if not target_func:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found"
                )
            
            # For now, return a placeholder - full implementation is complex
            return TransformResult(
                success=False,
                error="Extract function not fully implemented - use code generator instead"
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
    
    def add_error_handling(
        self,
        source: str,
        function_name: str,
        exception_type: str = "Exception"
    ) -> TransformResult:
        """
        Wrap function body in try-except.
        
        Args:
            source: Original source code
            function_name: Function to add error handling to
            exception_type: Type of exception to catch
            
        Returns:
            Transformation result
        """
        try:
            tree = ast.parse(source)
            changes = []
            
            class ErrorHandlingTransformer(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    if node.name == function_name:
                        # Check if already has try-except
                        if any(isinstance(stmt, ast.Try) for stmt in node.body):
                            return node
                        
                        # Create try-except wrapper
                        try_node = ast.Try(
                            body=node.body,
                            handlers=[
                                ast.ExceptHandler(
                                    type=ast.Name(id=exception_type, ctx=ast.Load()),
                                    name='e',
                                    body=[
                                        ast.Raise(
                                            exc=ast.Call(
                                                func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                                                args=[
                                                    ast.JoinedStr(values=[
                                                        ast.Constant(value=f'{function_name} failed: '),
                                                        ast.FormattedValue(
                                                            value=ast.Name(id='e', ctx=ast.Load()),
                                                            conversion=-1
                                                        )
                                                    ])
                                                ],
                                                keywords=[]
                                            ),
                                            cause=ast.Name(id='e', ctx=ast.Load())
                                        )
                                    ]
                                )
                            ],
                            orelse=[],
                            finalbody=[]
                        )
                        
                        node.body = [try_node]
                        changes.append(f"Added try-except to {function_name}")
                    
                    return node
            
            transformer = ErrorHandlingTransformer()
            new_tree = transformer.visit(tree)
            
            if not changes:
                return TransformResult(
                    success=False,
                    error=f"Function '{function_name}' not found or already has error handling"
                )
            
            new_code = astor.to_source(new_tree)
            
            return TransformResult(
                success=True,
                code=new_code,
                changes_made=changes
            )
            
        except Exception as e:
            return TransformResult(
                success=False,
                error=str(e)
            )
```

---

## Testing Requirements

**File:** `tests/test_code_patcher.py` (CREATE NEW FILE)

```python
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
```

---

## Acceptance Criteria

- [ ] `src/utils/code_patcher.py` - Safe patching with rollback
- [ ] `src/utils/code_generator.py` - LLM-based code generation
- [ ] `src/utils/code_transformer.py` - AST-based transformations
- [ ] `tests/test_code_patcher.py` - All tests passing
- [ ] Patches validate before applying
- [ ] Rollback works for failed patchsets
- [ ] Generated code passes validation

---

## Dependencies to Add

Add to `pyproject.toml`:
```toml
[tool.poetry.dependencies]
astor = "^0.8.1"  # For AST to source conversion
```

---

## File Summary

| Action | File Path |
|--------|-----------|
| REPLACE | `src/utils/code_patcher.py` |
| CREATE | `src/utils/code_generator.py` |
| CREATE | `src/utils/code_transformer.py` |
| CREATE | `tests/test_code_patcher.py` |

---

*End of Part 7*
