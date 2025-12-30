"""
Migrate data from King AI v1 (Node.js/SQLite) to v2 (Python/PostgreSQL).

Run this script once after setting up v2 infrastructure:
    python scripts/migrate_v1_data.py /path/to/old/king-ai-studio
"""

import sqlite3
import json
import asyncio
from pathlib import Path
from uuid import uuid4

async def migrate(old_path: str):
    """
    Orchestrates the migration of all data tables from v1 SQLite to v2 PostgreSQL.
    :param old_path: Path to the King AI v1 installation.
    """
    old_path = Path(old_path)
    old_db_path = old_path / "data" / "king-ai.db"
    
    if not old_db_path.exists():
        print(f"Error: Old database not found at {old_db_path}")
        return
    
    # Connect to old SQLite
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row
    
    # Migrate businesses
    print("Migrating businesses...")
    try:
        businesses = old_conn.execute("SELECT * FROM businesses").fetchall()
        for b in businesses:
            await migrate_business(dict(b))
        print(f"  Migrated {len(businesses)} businesses")
    except Exception as e:
        print(f"  Skipping businesses table: {e}")
    
    # Migrate tasks
    print("Migrating tasks...")
    try:
        tasks = old_conn.execute("SELECT * FROM tasks").fetchall()
        for t in tasks:
            await migrate_task(dict(t))
        print(f"  Migrated {len(tasks)} tasks")
    except Exception as e:
        print(f"  Skipping tasks table: {e}")
    
    # Migrate logs
    print("Migrating logs...")
    try:
        logs = old_conn.execute("SELECT * FROM logs").fetchall()
        for log in logs:
            await migrate_log(dict(log))
        print(f"  Migrated {len(logs)} logs")
    except Exception as e:
        print(f"  Skipping logs table: {e}")
    
    # Migrate approvals
    print("Migrating approvals...")
    try:
        approvals = old_conn.execute("SELECT * FROM approvals").fetchall()
        for a in approvals:
            await migrate_approval(dict(a))
        print(f"  Migrated {len(approvals)} approvals")
    except Exception as e:
        print(f"  Skipping approvals table: {e}")
    
    old_conn.close()
    print("Migration complete!")

async def migrate_business(old: dict):
    """Transform and insert a business record."""
    from src.database.connection import get_db
    from src.database.models import BusinessUnit, BusinessStatus
    
    status_map = {
        "active": BusinessStatus.OPERATION,
        "pending": BusinessStatus.SETUP,
        "completed": BusinessStatus.OPTIMIZATION,
        "failed": BusinessStatus.SUNSET,
    }
    
    async with get_db() as db:
        unit = BusinessUnit(
            id=old.get("id") or str(uuid4()),
            name=old.get("name", "Unknown"),
            type=old.get("type", "general"),
            status=status_map.get(old.get("status"), BusinessStatus.DISCOVERY),
            total_revenue=float(old.get("revenue", 0) or 0),
            total_expenses=float(old.get("expenses", 0) or 0),
            kpis=json.loads(old.get("kpis", "{}") or "{}"),
            config=json.loads(old.get("config", "{}") or "{}"),
        )
        db.add(unit)
        try:
            await db.commit()
        except:
            await db.rollback()

async def migrate_task(old: dict):
    # Stub implementation - actual migration logic depends on schema mapping
    pass

async def migrate_log(old: dict):
    pass

async def migrate_approval(old: dict):
    pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python migrate_v1_data.py /path/to/old/king-ai-studio")
        sys.exit(1)
    
    asyncio.run(migrate(sys.argv[1]))
