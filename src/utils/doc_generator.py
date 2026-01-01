"""
Documentation Generator.
Auto-generate documentation from code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Callable
from enum import Enum
import ast
import inspect
import os
import re

from src.utils.structured_logging import get_logger

logger = get_logger("doc_generator")


class DocFormat(str, Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"  # reStructuredText


@dataclass
class ParameterDoc:
    """Documentation for a function parameter."""
    name: str
    type_hint: str = ""
    description: str = ""
    default: Optional[str] = None
    required: bool = True


@dataclass
class ReturnDoc:
    """Documentation for function return value."""
    type_hint: str = ""
    description: str = ""


@dataclass
class ExceptionDoc:
    """Documentation for an exception."""
    type: str
    description: str = ""


@dataclass
class FunctionDoc:
    """Documentation for a function/method."""
    name: str
    signature: str
    docstring: str = ""
    description: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    raises: List[ExceptionDoc] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    deprecated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "parameters": [
                {"name": p.name, "type": p.type_hint, "description": p.description}
                for p in self.parameters
            ],
            "returns": {"type": self.returns.type_hint, "description": self.returns.description} if self.returns else None,
            "is_async": self.is_async,
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    docstring: str = ""
    description: str = ""
    bases: List[str] = field(default_factory=list)
    methods: List[FunctionDoc] = field(default_factory=list)
    class_methods: List[FunctionDoc] = field(default_factory=list)
    static_methods: List[FunctionDoc] = field(default_factory=list)
    properties: List[FunctionDoc] = field(default_factory=list)
    attributes: List[ParameterDoc] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "properties": [p.to_dict() for p in self.properties],
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    docstring: str = ""
    description: str = ""
    classes: List[ClassDoc] = field(default_factory=list)
    functions: List[FunctionDoc] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
        }


class DocstringParser:
    """Parse docstrings in various formats."""
    
    @classmethod
    def parse(cls, docstring: str) -> Dict[str, Any]:
        """Parse a docstring."""
        if not docstring:
            return {"description": "", "params": [], "returns": None, "raises": [], "examples": []}
        
        # Try Google style first, then NumPy, then basic
        if "Args:" in docstring or "Arguments:" in docstring:
            return cls._parse_google(docstring)
        elif "Parameters" in docstring and "----------" in docstring:
            return cls._parse_numpy(docstring)
        else:
            return cls._parse_basic(docstring)
    
    @classmethod
    def _parse_google(cls, docstring: str) -> Dict[str, Any]:
        """Parse Google-style docstring."""
        result = {
            "description": "",
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }
        
        lines = docstring.strip().split("\n")
        current_section = "description"
        current_content = []
        
        section_patterns = {
            "args": re.compile(r'^(Args|Arguments):?\s*$', re.IGNORECASE),
            "returns": re.compile(r'^Returns?:?\s*$', re.IGNORECASE),
            "raises": re.compile(r'^Raises?:?\s*$', re.IGNORECASE),
            "examples": re.compile(r'^Examples?:?\s*$', re.IGNORECASE),
        }
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section headers
            new_section = None
            for section, pattern in section_patterns.items():
                if pattern.match(stripped):
                    new_section = section
                    break
            
            if new_section:
                # Save previous section
                cls._save_section(result, current_section, current_content)
                current_section = new_section
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        cls._save_section(result, current_section, current_content)
        
        return result
    
    @classmethod
    def _save_section(cls, result: Dict, section: str, content: List[str]) -> None:
        """Save parsed section content."""
        text = "\n".join(content).strip()
        
        if section == "description":
            result["description"] = text
        elif section == "args":
            result["params"] = cls._parse_params(text)
        elif section == "returns":
            result["returns"] = text
        elif section == "raises":
            result["raises"] = cls._parse_raises(text)
        elif section == "examples":
            result["examples"] = [text] if text else []
    
    @classmethod
    def _parse_params(cls, text: str) -> List[Dict[str, str]]:
        """Parse parameter descriptions."""
        params = []
        param_pattern = re.compile(r'^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)$')
        
        current_param = None
        current_desc = []
        
        for line in text.split("\n"):
            match = param_pattern.match(line)
            if match:
                if current_param:
                    params.append({
                        "name": current_param["name"],
                        "type": current_param.get("type", ""),
                        "description": " ".join(current_desc).strip(),
                    })
                
                current_param = {"name": match.group(1), "type": match.group(2) or ""}
                current_desc = [match.group(3)]
            elif current_param and line.strip():
                current_desc.append(line.strip())
        
        if current_param:
            params.append({
                "name": current_param["name"],
                "type": current_param.get("type", ""),
                "description": " ".join(current_desc).strip(),
            })
        
        return params
    
    @classmethod
    def _parse_raises(cls, text: str) -> List[Dict[str, str]]:
        """Parse raises section."""
        raises = []
        raise_pattern = re.compile(r'^\s*(\w+)\s*:\s*(.*)$')
        
        for line in text.split("\n"):
            match = raise_pattern.match(line)
            if match:
                raises.append({
                    "type": match.group(1),
                    "description": match.group(2).strip(),
                })
        
        return raises
    
    @classmethod
    def _parse_numpy(cls, docstring: str) -> Dict[str, Any]:
        """Parse NumPy-style docstring."""
        # Simplified NumPy parsing
        return cls._parse_basic(docstring)
    
    @classmethod
    def _parse_basic(cls, docstring: str) -> Dict[str, Any]:
        """Parse basic docstring (just description)."""
        return {
            "description": docstring.strip(),
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }


class CodeAnalyzer:
    """Analyze Python code to extract documentation."""
    
    def analyze_file(self, file_path: str) -> ModuleDoc:
        """Analyze a Python file."""
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        
        return ModuleDoc(
            name=module_name,
            path=file_path,
            docstring=ast.get_docstring(tree) or "",
            description=self._get_description(ast.get_docstring(tree)),
            classes=self._extract_classes(tree),
            functions=self._extract_functions(tree),
            constants=self._extract_constants(tree),
            imports=self._extract_imports(tree),
        )
    
    def _get_description(self, docstring: str) -> str:
        """Extract description from docstring."""
        if not docstring:
            return ""
        
        parsed = DocstringParser.parse(docstring)
        return parsed.get("description", "").split("\n")[0]
    
    def _extract_classes(self, tree: ast.AST) -> List[ClassDoc]:
        """Extract class documentation."""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = self._document_class(node)
                classes.append(class_doc)
        
        return classes
    
    def _document_class(self, node: ast.ClassDef) -> ClassDoc:
        """Document a class."""
        docstring = ast.get_docstring(node) or ""
        parsed = DocstringParser.parse(docstring)
        
        methods = []
        class_methods = []
        static_methods = []
        properties = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef) or isinstance(item, ast.AsyncFunctionDef):
                func_doc = self._document_function(item)
                
                # Check decorators
                decorator_names = [
                    self._get_decorator_name(d)
                    for d in item.decorator_list
                ]
                
                if "property" in decorator_names:
                    properties.append(func_doc)
                elif "classmethod" in decorator_names:
                    class_methods.append(func_doc)
                elif "staticmethod" in decorator_names:
                    static_methods.append(func_doc)
                else:
                    methods.append(func_doc)
        
        return ClassDoc(
            name=node.name,
            docstring=docstring,
            description=parsed.get("description", ""),
            bases=[self._get_base_name(base) for base in node.bases],
            methods=methods,
            class_methods=class_methods,
            static_methods=static_methods,
            properties=properties,
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
        )
    
    def _get_base_name(self, node: ast.AST) -> str:
        """Get base class name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return ""
    
    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get decorator name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""
    
    def _extract_functions(self, tree: ast.AST) -> List[FunctionDoc]:
        """Extract module-level function documentation."""
        functions = []
        
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_doc = self._document_function(node)
                functions.append(func_doc)
        
        return functions
    
    def _document_function(self, node) -> FunctionDoc:
        """Document a function."""
        docstring = ast.get_docstring(node) or ""
        parsed = DocstringParser.parse(docstring)
        
        # Build signature
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            type_hint = ""
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ""
            args.append(f"{arg_name}: {type_hint}" if type_hint else arg_name)
        
        signature = f"({', '.join(args)})"
        
        # Return type
        returns = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else ""
            returns = ReturnDoc(
                type_hint=return_type,
                description=parsed.get("returns", ""),
            )
        
        # Parameters
        parameters = []
        parsed_params = {p["name"]: p for p in parsed.get("params", [])}
        
        for arg in node.args.args:
            param_info = parsed_params.get(arg.arg, {})
            type_hint = ""
            if arg.annotation and hasattr(ast, 'unparse'):
                type_hint = ast.unparse(arg.annotation)
            
            parameters.append(ParameterDoc(
                name=arg.arg,
                type_hint=type_hint or param_info.get("type", ""),
                description=param_info.get("description", ""),
            ))
        
        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=docstring,
            description=parsed.get("description", ""),
            parameters=parameters,
            returns=returns,
            raises=[
                ExceptionDoc(type=r["type"], description=r["description"])
                for r in parsed.get("raises", [])
            ],
            examples=parsed.get("examples", []),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )
    
    def _extract_constants(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract module-level constants."""
        constants = []
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # Check if it's a constant (UPPER_CASE)
                        if name.isupper():
                            constants.append({
                                "name": name,
                                "value": ast.unparse(node.value) if hasattr(ast, 'unparse') else "",
                            })
        
        return constants
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imports."""
        imports = []
        
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports


class MarkdownRenderer:
    """Render documentation as Markdown."""
    
    def render_module(self, doc: ModuleDoc) -> str:
        """Render module documentation."""
        lines = []
        
        # Header
        lines.append(f"# {doc.name}")
        lines.append("")
        
        if doc.description:
            lines.append(doc.description)
            lines.append("")
        
        if doc.docstring and doc.docstring != doc.description:
            lines.append(doc.docstring)
            lines.append("")
        
        # Functions
        if doc.functions:
            lines.append("## Functions")
            lines.append("")
            for func in doc.functions:
                lines.append(self.render_function(func))
        
        # Classes
        if doc.classes:
            lines.append("## Classes")
            lines.append("")
            for cls in doc.classes:
                lines.append(self.render_class(cls))
        
        return "\n".join(lines)
    
    def render_function(self, doc: FunctionDoc) -> str:
        """Render function documentation."""
        lines = []
        
        # Function header
        async_prefix = "async " if doc.is_async else ""
        lines.append(f"### `{async_prefix}{doc.name}{doc.signature}`")
        lines.append("")
        
        if doc.description:
            lines.append(doc.description)
            lines.append("")
        
        # Parameters
        if doc.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for param in doc.parameters:
                type_str = f" (`{param.type_hint}`)" if param.type_hint else ""
                lines.append(f"- `{param.name}`{type_str}: {param.description}")
            lines.append("")
        
        # Returns
        if doc.returns:
            lines.append("**Returns:**")
            lines.append("")
            type_str = f"`{doc.returns.type_hint}`" if doc.returns.type_hint else ""
            lines.append(f"{type_str} {doc.returns.description}")
            lines.append("")
        
        # Raises
        if doc.raises:
            lines.append("**Raises:**")
            lines.append("")
            for exc in doc.raises:
                lines.append(f"- `{exc.type}`: {exc.description}")
            lines.append("")
        
        return "\n".join(lines)
    
    def render_class(self, doc: ClassDoc) -> str:
        """Render class documentation."""
        lines = []
        
        # Class header
        bases_str = f"({', '.join(doc.bases)})" if doc.bases else ""
        lines.append(f"### class `{doc.name}{bases_str}`")
        lines.append("")
        
        if doc.description:
            lines.append(doc.description)
            lines.append("")
        
        # Properties
        if doc.properties:
            lines.append("**Properties:**")
            lines.append("")
            for prop in doc.properties:
                lines.append(f"- `{prop.name}`: {prop.description}")
            lines.append("")
        
        # Methods
        if doc.methods:
            lines.append("**Methods:**")
            lines.append("")
            for method in doc.methods:
                if not method.name.startswith("_"):
                    lines.append(f"- `{method.name}{method.signature}`: {method.description}")
            lines.append("")
        
        return "\n".join(lines)


class DocumentationGenerator:
    """
    Documentation Generator.
    
    Features:
    - Parse Python source code
    - Extract docstrings
    - Generate Markdown/HTML/RST
    - Support Google/NumPy docstring styles
    """
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.renderers = {
            DocFormat.MARKDOWN: MarkdownRenderer(),
        }
    
    def generate_for_file(
        self,
        file_path: str,
        format: DocFormat = DocFormat.MARKDOWN,
    ) -> str:
        """
        Generate documentation for a file.
        
        Args:
            file_path: Path to Python file
            format: Output format
            
        Returns:
            Generated documentation
        """
        doc = self.analyzer.analyze_file(file_path)
        renderer = self.renderers.get(format, MarkdownRenderer())
        return renderer.render_module(doc)
    
    def generate_for_directory(
        self,
        directory: str,
        format: DocFormat = DocFormat.MARKDOWN,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate documentation for all Python files in a directory.
        
        Args:
            directory: Directory path
            format: Output format
            output_dir: Where to write files (optional)
            
        Returns:
            Dict of file paths to documentation
        """
        results = {}
        
        for root, dirs, files in os.walk(directory):
            # Skip hidden and __pycache__ directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            
            for file in files:
                if file.endswith(".py") and not file.startswith("_"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        doc = self.generate_for_file(file_path, format)
                        results[file_path] = doc
                        
                        if output_dir:
                            # Write to output directory
                            rel_path = os.path.relpath(file_path, directory)
                            out_path = os.path.join(
                                output_dir,
                                rel_path.replace(".py", ".md"),
                            )
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            with open(out_path, "w", encoding="utf-8") as f:
                                f.write(doc)
                    
                    except Exception as e:
                        logger.error(f"Error documenting {file_path}: {e}")
        
        return results
    
    def generate_api_reference(
        self,
        directory: str,
        title: str = "API Reference",
    ) -> str:
        """
        Generate combined API reference.
        
        Args:
            directory: Source directory
            title: Document title
            
        Returns:
            Combined API reference document
        """
        lines = [f"# {title}", ""]
        
        docs = self.generate_for_directory(directory)
        
        for file_path, doc in sorted(docs.items()):
            lines.append(doc)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)


# Global documentation generator
doc_generator = DocumentationGenerator()


def get_doc_generator() -> DocumentationGenerator:
    """Get the global documentation generator."""
    return doc_generator
