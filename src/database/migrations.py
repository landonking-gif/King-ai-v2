"""
Database Migration Runner.
Programmatic migration management with rollback support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Awaitable
from pathlib import Path
from enum import Enum
import hashlib
import asyncio
import importlib.util
import sys

from src.utils.structured_logging import get_logger

logger = get_logger("migrations")


class MigrationStatus(str, Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationDirection(str, Enum):
    """Migration direction."""
    UP = "up"
    DOWN = "down"


@dataclass
class Migration:
    """A single database migration."""
    id: str
    name: str
    version: str
    description: str = ""
    up_sql: Optional[str] = None
    down_sql: Optional[str] = None
    up_func: Optional[Callable[..., Awaitable[None]]] = None
    down_func: Optional[Callable[..., Awaitable[None]]] = None
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.checksum:
            content = (self.up_sql or "") + (self.down_sql or "")
            self.checksum = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class MigrationRecord:
    """Record of an executed migration."""
    migration_id: str
    version: str
    name: str
    status: MigrationStatus
    checksum: str
    executed_at: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: int = 0
    error_message: Optional[str] = None
    batch: int = 1


@dataclass
class MigrationResult:
    """Result of migration execution."""
    success: bool
    migrations_run: int
    migrations_failed: int
    duration_ms: int
    errors: List[str] = field(default_factory=list)
    executed: List[str] = field(default_factory=list)


class MigrationLoader:
    """Loads migrations from various sources."""
    
    @staticmethod
    def from_sql_files(
        directory: Path,
    ) -> List[Migration]:
        """Load migrations from SQL files."""
        migrations = []
        
        if not directory.exists():
            return migrations
        
        # Find migration files
        files = sorted(directory.glob("*.sql"))
        
        for file_path in files:
            # Expected format: V001__migration_name.sql
            name = file_path.stem
            parts = name.split("__", 1)
            
            if len(parts) != 2:
                continue
            
            version = parts[0]
            desc = parts[1].replace("_", " ")
            
            content = file_path.read_text()
            
            # Split up/down if marked
            up_sql = content
            down_sql = None
            
            if "-- DOWN" in content:
                up_part, down_part = content.split("-- DOWN", 1)
                up_sql = up_part.strip()
                down_sql = down_part.strip()
            
            migrations.append(Migration(
                id=name,
                name=desc,
                version=version,
                up_sql=up_sql,
                down_sql=down_sql,
            ))
        
        return migrations
    
    @staticmethod
    def from_python_files(
        directory: Path,
    ) -> List[Migration]:
        """Load migrations from Python files."""
        migrations = []
        
        if not directory.exists():
            return migrations
        
        files = sorted(directory.glob("*.py"))
        
        for file_path in files:
            if file_path.name.startswith("_"):
                continue
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                file_path.stem, file_path
            )
            if not spec or not spec.loader:
                continue
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[file_path.stem] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"Failed to load migration {file_path}: {e}")
                continue
            
            # Extract migration info
            if hasattr(module, "upgrade") and hasattr(module, "downgrade"):
                name = file_path.stem
                parts = name.split("_", 1)
                
                version = parts[0] if parts else "000"
                desc = parts[1].replace("_", " ") if len(parts) > 1 else name
                
                migrations.append(Migration(
                    id=name,
                    name=desc,
                    version=version,
                    description=getattr(module, "__doc__", "") or "",
                    up_func=module.upgrade,
                    down_func=module.downgrade,
                    dependencies=getattr(module, "dependencies", []),
                ))
        
        return migrations


class MigrationRunner:
    """
    Executes database migrations with rollback support.
    
    Features:
    - SQL and Python migrations
    - Dependency ordering
    - Batch execution
    - Rollback support
    - Checksum verification
    - Dry run mode
    """
    
    # Table to track migrations
    MIGRATIONS_TABLE = "schema_migrations"
    
    def __init__(
        self,
        db_connection: Any = None,
        migrations_dir: Optional[Path] = None,
    ):
        self._connection = db_connection
        self._migrations_dir = migrations_dir
        self._migrations: Dict[str, Migration] = {}
        self._records: Dict[str, MigrationRecord] = {}
        self._current_batch = 1
    
    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations[migration.id] = migration
    
    def register_many(self, migrations: List[Migration]) -> None:
        """Register multiple migrations."""
        for m in migrations:
            self.register(m)
    
    async def initialize(self) -> None:
        """Initialize migration tracking table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
            id SERIAL PRIMARY KEY,
            migration_id VARCHAR(255) UNIQUE NOT NULL,
            version VARCHAR(50) NOT NULL,
            name VARCHAR(255) NOT NULL,
            status VARCHAR(50) NOT NULL,
            checksum VARCHAR(50) NOT NULL,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            execution_time_ms INTEGER DEFAULT 0,
            error_message TEXT,
            batch INTEGER DEFAULT 1
        );
        """
        
        await self._execute_sql(create_table_sql)
        await self._load_records()
        
        # Determine current batch
        if self._records:
            self._current_batch = max(r.batch for r in self._records.values()) + 1
    
    async def _load_records(self) -> None:
        """Load existing migration records from database."""
        try:
            rows = await self._query(
                f"SELECT * FROM {self.MIGRATIONS_TABLE}"
            )
            
            for row in rows:
                record = MigrationRecord(
                    migration_id=row["migration_id"],
                    version=row["version"],
                    name=row["name"],
                    status=MigrationStatus(row["status"]),
                    checksum=row["checksum"],
                    executed_at=row["executed_at"],
                    execution_time_ms=row["execution_time_ms"],
                    error_message=row.get("error_message"),
                    batch=row["batch"],
                )
                self._records[record.migration_id] = record
        except Exception:
            # Table might not exist yet
            pass
    
    async def _execute_sql(self, sql: str) -> None:
        """Execute SQL statement."""
        if self._connection:
            await self._connection.execute(sql)
        else:
            logger.debug(f"Would execute: {sql[:100]}...")
    
    async def _query(self, sql: str) -> List[Dict]:
        """Execute query and return rows."""
        if self._connection:
            return await self._connection.fetch(sql)
        return []
    
    def get_pending(self) -> List[Migration]:
        """Get migrations that haven't been run."""
        pending = []
        
        for migration in self._migrations.values():
            if migration.id not in self._records:
                pending.append(migration)
            elif self._records[migration.id].status == MigrationStatus.FAILED:
                pending.append(migration)
        
        # Sort by version
        pending.sort(key=lambda m: m.version)
        
        return pending
    
    def get_completed(self) -> List[MigrationRecord]:
        """Get completed migrations."""
        return [
            r for r in self._records.values()
            if r.status == MigrationStatus.COMPLETED
        ]
    
    async def migrate(
        self,
        target_version: Optional[str] = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Run pending migrations.
        
        Args:
            target_version: Stop at this version (None = run all)
            dry_run: Preview without executing
            
        Returns:
            Migration result
        """
        pending = self.get_pending()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations")
            return MigrationResult(
                success=True,
                migrations_run=0,
                migrations_failed=0,
                duration_ms=0,
            )
        
        logger.info(f"Running {len(pending)} pending migrations")
        
        start_time = datetime.utcnow()
        executed = []
        errors = []
        
        for migration in pending:
            if dry_run:
                logger.info(f"[DRY RUN] Would run: {migration.name}")
                executed.append(migration.id)
                continue
            
            try:
                await self._run_migration(migration, MigrationDirection.UP)
                executed.append(migration.id)
            except Exception as e:
                error_msg = f"{migration.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Migration failed: {error_msg}")
                break  # Stop on first failure
        
        duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return MigrationResult(
            success=len(errors) == 0,
            migrations_run=len(executed),
            migrations_failed=len(errors),
            duration_ms=duration,
            errors=errors,
            executed=executed,
        )
    
    async def rollback(
        self,
        steps: int = 1,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Rollback migrations.
        
        Args:
            steps: Number of batches to rollback
            dry_run: Preview without executing
            
        Returns:
            Migration result
        """
        # Get migrations from last N batches
        completed = self.get_completed()
        completed.sort(key=lambda r: (r.batch, r.version), reverse=True)
        
        if not completed:
            return MigrationResult(
                success=True,
                migrations_run=0,
                migrations_failed=0,
                duration_ms=0,
            )
        
        target_batch = completed[0].batch - steps + 1
        to_rollback = [r for r in completed if r.batch >= target_batch]
        
        logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        start_time = datetime.utcnow()
        rolled_back = []
        errors = []
        
        for record in to_rollback:
            migration = self._migrations.get(record.migration_id)
            if not migration:
                errors.append(f"Migration not found: {record.migration_id}")
                continue
            
            if dry_run:
                logger.info(f"[DRY RUN] Would rollback: {migration.name}")
                rolled_back.append(migration.id)
                continue
            
            try:
                await self._run_migration(migration, MigrationDirection.DOWN)
                rolled_back.append(migration.id)
            except Exception as e:
                error_msg = f"{migration.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Rollback failed: {error_msg}")
        
        duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return MigrationResult(
            success=len(errors) == 0,
            migrations_run=len(rolled_back),
            migrations_failed=len(errors),
            duration_ms=duration,
            errors=errors,
            executed=rolled_back,
        )
    
    async def _run_migration(
        self,
        migration: Migration,
        direction: MigrationDirection,
    ) -> None:
        """Execute a single migration."""
        start = datetime.utcnow()
        
        logger.info(f"Running migration {direction.value}: {migration.name}")
        
        try:
            if direction == MigrationDirection.UP:
                if migration.up_func:
                    await migration.up_func(self._connection)
                elif migration.up_sql:
                    await self._execute_sql(migration.up_sql)
                
                # Record success
                await self._record_migration(migration, MigrationStatus.COMPLETED, start)
            
            else:  # DOWN
                if migration.down_func:
                    await migration.down_func(self._connection)
                elif migration.down_sql:
                    await self._execute_sql(migration.down_sql)
                else:
                    raise ValueError(f"No rollback defined for {migration.name}")
                
                # Remove record
                await self._remove_record(migration.id)
        
        except Exception as e:
            if direction == MigrationDirection.UP:
                await self._record_migration(
                    migration, 
                    MigrationStatus.FAILED, 
                    start,
                    error=str(e)
                )
            raise
    
    async def _record_migration(
        self,
        migration: Migration,
        status: MigrationStatus,
        start_time: datetime,
        error: Optional[str] = None,
    ) -> None:
        """Record migration execution."""
        duration = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        record = MigrationRecord(
            migration_id=migration.id,
            version=migration.version,
            name=migration.name,
            status=status,
            checksum=migration.checksum,
            execution_time_ms=duration,
            error_message=error,
            batch=self._current_batch,
        )
        
        self._records[migration.id] = record
        
        insert_sql = f"""
        INSERT INTO {self.MIGRATIONS_TABLE}
        (migration_id, version, name, status, checksum, execution_time_ms, error_message, batch)
        VALUES ('{migration.id}', '{migration.version}', '{migration.name}', 
                '{status.value}', '{migration.checksum}', {duration}, 
                {'NULL' if not error else f"'{error}'"}, {self._current_batch})
        ON CONFLICT (migration_id) DO UPDATE
        SET status = EXCLUDED.status,
            execution_time_ms = EXCLUDED.execution_time_ms,
            error_message = EXCLUDED.error_message,
            executed_at = CURRENT_TIMESTAMP;
        """
        
        await self._execute_sql(insert_sql)
    
    async def _remove_record(self, migration_id: str) -> None:
        """Remove migration record after rollback."""
        self._records.pop(migration_id, None)
        
        delete_sql = f"""
        DELETE FROM {self.MIGRATIONS_TABLE}
        WHERE migration_id = '{migration_id}';
        """
        
        await self._execute_sql(delete_sql)
    
    def status(self) -> Dict[str, Any]:
        """Get migration status summary."""
        pending = self.get_pending()
        completed = self.get_completed()
        
        return {
            "total_migrations": len(self._migrations),
            "completed": len(completed),
            "pending": len(pending),
            "failed": sum(
                1 for r in self._records.values()
                if r.status == MigrationStatus.FAILED
            ),
            "current_batch": self._current_batch,
            "pending_migrations": [
                {"id": m.id, "name": m.name, "version": m.version}
                for m in pending
            ],
            "recent_migrations": [
                {
                    "id": r.migration_id,
                    "name": r.name,
                    "status": r.status.value,
                    "executed_at": r.executed_at.isoformat(),
                }
                for r in sorted(
                    completed, 
                    key=lambda r: r.executed_at, 
                    reverse=True
                )[:5]
            ],
        }
    
    def verify_checksums(self) -> List[str]:
        """Verify migration checksums haven't changed."""
        mismatches = []
        
        for record in self._records.values():
            migration = self._migrations.get(record.migration_id)
            if migration and migration.checksum != record.checksum:
                mismatches.append(
                    f"{record.migration_id}: expected {record.checksum}, got {migration.checksum}"
                )
        
        return mismatches


# Global runner instance
migration_runner: Optional[MigrationRunner] = None


def get_migration_runner() -> MigrationRunner:
    """Get the global migration runner instance."""
    global migration_runner
    if migration_runner is None:
        migration_runner = MigrationRunner()
    return migration_runner


async def init_migrations(
    db_connection: Any,
    migrations_dir: str = "migrations",
) -> MigrationRunner:
    """Initialize migration runner with database connection."""
    global migration_runner
    
    path = Path(migrations_dir)
    
    migration_runner = MigrationRunner(
        db_connection=db_connection,
        migrations_dir=path,
    )
    
    # Load migrations from files
    if path.exists():
        sql_migrations = MigrationLoader.from_sql_files(path / "sql")
        py_migrations = MigrationLoader.from_python_files(path / "python")
        
        migration_runner.register_many(sql_migrations)
        migration_runner.register_many(py_migrations)
    
    await migration_runner.initialize()
    
    return migration_runner
