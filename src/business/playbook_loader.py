"""
Playbook Loader - Parses YAML business strategies.
Provides the 'Blueprints' for AI execution.
"""

import yaml
from pathlib import Path

class PlaybookLoader:
    """
    Interface for loading strategic playbooks from the filesystem.
    """
    
    def __init__(self, playbook_dir: str = "config/playbooks"):
        self.playbook_dir = Path(playbook_dir)

    def load_playbook(self, name: str) -> dict | None:
        """Loads a specific playbook by filename (without extension)."""
        file_path = self.playbook_dir / f"{name}.yaml"
        if not file_path.exists():
            return None
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def list_playbooks(self) -> list[str]:
        """Lists all available business blueprints."""
        if not self.playbook_dir.exists():
            return []
        return [f.stem for f in self.playbook_dir.glob("*.yaml")]
