"""
Configuration Hot Reload.
Enables runtime configuration changes without restart.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import hashlib
import yaml
import json
import os

from src.utils.structured_logging import get_logger

logger = get_logger("config_reload")


class ConfigFormat(str, Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


class ReloadStrategy(str, Enum):
    """Configuration reload strategies."""
    IMMEDIATE = "immediate"  # Apply immediately
    GRACEFUL = "graceful"  # Wait for in-flight requests
    SCHEDULED = "scheduled"  # Apply at specific time


@dataclass
class ConfigSource:
    """Configuration source definition."""
    name: str
    path: Path
    format: ConfigFormat
    priority: int = 0  # Higher priority overrides lower
    watch: bool = True
    reload_strategy: ReloadStrategy = ReloadStrategy.GRACEFUL
    
    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class ConfigChange:
    """Represents a configuration change."""
    source: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_addition(self) -> bool:
        return self.old_value is None and self.new_value is not None
    
    @property
    def is_removal(self) -> bool:
        return self.old_value is not None and self.new_value is None
    
    @property
    def is_modification(self) -> bool:
        return self.old_value is not None and self.new_value is not None


@dataclass
class ConfigSnapshot:
    """Snapshot of configuration at a point in time."""
    config: Dict[str, Any]
    sources: List[str]
    hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConfigValidator:
    """Validates configuration values."""
    
    def __init__(self):
        self._validators: Dict[str, Callable[[Any], bool]] = {}
        self._type_hints: Dict[str, type] = {}
    
    def register(
        self,
        key: str,
        validator: Callable[[Any], bool],
        expected_type: Optional[type] = None,
    ) -> None:
        """Register a validator for a config key."""
        self._validators[key] = validator
        if expected_type:
            self._type_hints[key] = expected_type
    
    def validate(self, key: str, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a config value."""
        # Type check
        if key in self._type_hints:
            expected = self._type_hints[key]
            if not isinstance(value, expected):
                return False, f"Expected {expected.__name__}, got {type(value).__name__}"
        
        # Custom validator
        if key in self._validators:
            try:
                if not self._validators[key](value):
                    return False, f"Validation failed for {key}"
            except Exception as e:
                return False, f"Validator error: {str(e)}"
        
        return True, None


class ConfigWatcher:
    """Watches configuration files for changes."""
    
    def __init__(self):
        self._file_hashes: Dict[Path, str] = {}
        self._callbacks: List[Callable[[Path], None]] = []
        self._watching = False
        self._watch_task: Optional[asyncio.Task] = None
        self._poll_interval = 5.0  # seconds
    
    def add_callback(self, callback: Callable[[Path], None]) -> None:
        """Add callback for file changes."""
        self._callbacks.append(callback)
    
    def _calculate_hash(self, path: Path) -> str:
        """Calculate file content hash."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def add_file(self, path: Path) -> None:
        """Start watching a file."""
        self._file_hashes[path] = self._calculate_hash(path)
    
    def remove_file(self, path: Path) -> None:
        """Stop watching a file."""
        self._file_hashes.pop(path, None)
    
    async def start(self) -> None:
        """Start watching for changes."""
        if self._watching:
            return
        
        self._watching = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        logger.info("Config watcher started")
    
    async def stop(self) -> None:
        """Stop watching for changes."""
        self._watching = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        logger.info("Config watcher stopped")
    
    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._watching:
            try:
                for path in list(self._file_hashes.keys()):
                    current_hash = self._calculate_hash(path)
                    
                    if current_hash != self._file_hashes[path]:
                        logger.info(f"Config file changed: {path}")
                        self._file_hashes[path] = current_hash
                        
                        for callback in self._callbacks:
                            try:
                                callback(path)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watch loop error: {e}")
                await asyncio.sleep(self._poll_interval)


class HotReloadConfig:
    """
    Configuration manager with hot reload support.
    
    Features:
    - Multiple config sources with priority
    - File watching for auto-reload
    - Validation before applying changes
    - Change callbacks for dependent services
    - Rollback support
    """
    
    def __init__(
        self,
        poll_interval: float = 5.0,
        enable_watching: bool = True,
    ):
        self._sources: Dict[str, ConfigSource] = {}
        self._config: Dict[str, Any] = {}
        self._history: List[ConfigSnapshot] = []
        self._max_history = 10
        
        self._validator = ConfigValidator()
        self._watcher = ConfigWatcher()
        self._watcher._poll_interval = poll_interval
        self._enable_watching = enable_watching
        
        self._change_callbacks: List[Callable[[List[ConfigChange]], None]] = []
        self._key_callbacks: Dict[str, List[Callable[[Any, Any], None]]] = {}
        
        self._reload_lock = asyncio.Lock()
        self._initialized = False
    
    def add_source(self, source: ConfigSource) -> None:
        """Add a configuration source."""
        self._sources[source.name] = source
        
        if source.watch and self._enable_watching:
            self._watcher.add_file(source.path)
        
        logger.info(f"Added config source: {source.name} ({source.path})")
    
    def remove_source(self, name: str) -> None:
        """Remove a configuration source."""
        if name in self._sources:
            source = self._sources[name]
            self._watcher.remove_file(source.path)
            del self._sources[name]
    
    def register_validator(
        self,
        key: str,
        validator: Callable[[Any], bool],
        expected_type: Optional[type] = None,
    ) -> None:
        """Register a validator for a config key."""
        self._validator.register(key, validator, expected_type)
    
    def on_change(self, callback: Callable[[List[ConfigChange]], None]) -> None:
        """Register callback for any config changes."""
        self._change_callbacks.append(callback)
    
    def on_key_change(self, key: str, callback: Callable[[Any, Any], None]) -> None:
        """Register callback for specific key changes."""
        if key not in self._key_callbacks:
            self._key_callbacks[key] = []
        self._key_callbacks[key].append(callback)
    
    async def initialize(self) -> None:
        """Initialize and load all configurations."""
        await self.reload()
        
        if self._enable_watching:
            self._watcher.add_callback(self._on_file_changed)
            await self._watcher.start()
        
        self._initialized = True
        logger.info(f"Config initialized with {len(self._sources)} sources")
    
    async def shutdown(self) -> None:
        """Shutdown the config manager."""
        await self._watcher.stop()
        logger.info("Config manager shutdown")
    
    def _on_file_changed(self, path: Path) -> None:
        """Handle file change notification."""
        # Find source for this path
        for source in self._sources.values():
            if source.path == path:
                asyncio.create_task(self._reload_source(source))
                break
    
    async def _reload_source(self, source: ConfigSource) -> None:
        """Reload a specific source."""
        async with self._reload_lock:
            logger.info(f"Reloading config source: {source.name}")
            
            try:
                new_config = self._load_source(source)
                old_config = self._config.copy()
                
                # Merge with priority
                await self._apply_all_sources()
                
                # Detect changes
                changes = self._detect_changes(old_config, self._config)
                
                if changes:
                    self._notify_changes(changes)
                
            except Exception as e:
                logger.error(f"Failed to reload {source.name}: {e}")
    
    async def reload(self) -> List[ConfigChange]:
        """Reload all configuration sources."""
        async with self._reload_lock:
            old_config = self._config.copy()
            
            # Save snapshot for rollback
            self._save_snapshot()
            
            # Apply all sources
            await self._apply_all_sources()
            
            # Detect changes
            changes = self._detect_changes(old_config, self._config)
            
            if changes:
                self._notify_changes(changes)
            
            return changes
    
    async def _apply_all_sources(self) -> None:
        """Apply all config sources with priority."""
        merged: Dict[str, Any] = {}
        
        # Sort by priority (lower first)
        sorted_sources = sorted(
            self._sources.values(),
            key=lambda s: s.priority
        )
        
        for source in sorted_sources:
            try:
                config = self._load_source(source)
                self._deep_merge(merged, config)
            except Exception as e:
                logger.error(f"Failed to load {source.name}: {e}")
        
        self._config = merged
    
    def _load_source(self, source: ConfigSource) -> Dict[str, Any]:
        """Load configuration from a source."""
        if not source.path.exists():
            logger.warning(f"Config file not found: {source.path}")
            return {}
        
        content = source.path.read_text()
        
        if source.format == ConfigFormat.YAML:
            return yaml.safe_load(content) or {}
        elif source.format == ConfigFormat.JSON:
            return json.loads(content)
        elif source.format == ConfigFormat.ENV:
            return self._parse_env(content)
        
        return {}
    
    def _parse_env(self, content: str) -> Dict[str, Any]:
        """Parse .env format."""
        result = {}
        
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                result[key] = value
        
        return result
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _detect_changes(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
        prefix: str = "",
    ) -> List[ConfigChange]:
        """Detect changes between configs."""
        changes = []
        
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            old_val = old.get(key)
            new_val = new.get(key)
            
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                changes.extend(self._detect_changes(old_val, new_val, full_key))
            elif old_val != new_val:
                changes.append(ConfigChange(
                    source="merged",
                    key=full_key,
                    old_value=old_val,
                    new_value=new_val,
                ))
        
        return changes
    
    def _notify_changes(self, changes: List[ConfigChange]) -> None:
        """Notify callbacks of changes."""
        # General callbacks
        for callback in self._change_callbacks:
            try:
                callback(changes)
            except Exception as e:
                logger.error(f"Change callback error: {e}")
        
        # Key-specific callbacks
        for change in changes:
            if change.key in self._key_callbacks:
                for callback in self._key_callbacks[change.key]:
                    try:
                        callback(change.old_value, change.new_value)
                    except Exception as e:
                        logger.error(f"Key callback error for {change.key}: {e}")
    
    def _save_snapshot(self) -> None:
        """Save current config as snapshot."""
        snapshot = ConfigSnapshot(
            config=self._config.copy(),
            sources=list(self._sources.keys()),
            hash=hashlib.md5(
                json.dumps(self._config, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )
        
        self._history.append(snapshot)
        
        # Limit history
        while len(self._history) > self._max_history:
            self._history.pop(0)
    
    async def rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous configuration."""
        if steps >= len(self._history):
            logger.warning("Not enough history to rollback")
            return False
        
        async with self._reload_lock:
            snapshot = self._history[-(steps + 1)]
            old_config = self._config.copy()
            
            self._config = snapshot.config.copy()
            
            changes = self._detect_changes(old_config, self._config)
            if changes:
                self._notify_changes(changes)
            
            logger.info(f"Rolled back {steps} step(s) to snapshot from {snapshot.timestamp}")
            return True
    
    # Access methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = self.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 'on')
        return bool(value)
    
    def get_list(self, key: str, default: Optional[List] = None) -> List:
        """Get a list configuration value."""
        value = self.get(key)
        if value is None:
            return default or []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [v.strip() for v in value.split(',')]
        return [value]
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value at runtime."""
        # Validate first
        is_valid, error = self._validator.validate(key, value)
        if not is_valid:
            raise ValueError(f"Invalid value for {key}: {error}")
        
        old_value = self.get(key)
        
        parts = key.split('.')
        target = self._config
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
        
        # Notify
        if old_value != value:
            change = ConfigChange(
                source="runtime",
                key=key,
                old_value=old_value,
                new_value=value,
            )
            self._notify_changes([change])
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config.copy()
    
    @property
    def sources(self) -> List[str]:
        """Get list of source names."""
        return list(self._sources.keys())


# Global instance
hot_config = HotReloadConfig()


def get_hot_config() -> HotReloadConfig:
    """Get the global hot reload config instance."""
    return hot_config


async def init_config(config_dir: str = "config") -> HotReloadConfig:
    """Initialize config with default sources."""
    config_path = Path(config_dir)
    
    # Add common config sources
    if (config_path / "settings.py").exists():
        # For Python settings, we'd need different handling
        pass
    
    if (config_path / "settings.yaml").exists():
        hot_config.add_source(ConfigSource(
            name="settings",
            path=config_path / "settings.yaml",
            format=ConfigFormat.YAML,
            priority=10,
        ))
    
    if (config_path / "risk_profiles.yaml").exists():
        hot_config.add_source(ConfigSource(
            name="risk_profiles",
            path=config_path / "risk_profiles.yaml",
            format=ConfigFormat.YAML,
            priority=20,
        ))
    
    # Load playbooks
    playbooks_dir = config_path / "playbooks"
    if playbooks_dir.exists():
        for playbook in playbooks_dir.glob("*.yaml"):
            hot_config.add_source(ConfigSource(
                name=f"playbook_{playbook.stem}",
                path=playbook,
                format=ConfigFormat.YAML,
                priority=30,
            ))
    
    await hot_config.initialize()
    return hot_config
