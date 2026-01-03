#!/bin/bash
# Redis Restore Script for Mother-Harness
# Restores Redis data from a backup archive

set -e

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-mother_dev_password}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/mother-harness/redis}"

# Check if backup file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <backup-file.tar.gz> [--force]"
    echo ""
    echo "Available backups:"
    ls -lh "$BACKUP_DIR"/redis_backup_*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"
FORCE_RESTORE="$2"

# Validate backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    # Try looking in backup directory
    BACKUP_FILE="$BACKUP_DIR/$1"
    if [ ! -f "$BACKUP_FILE" ]; then
        echo "Error: Backup file not found: $1"
        exit 1
    fi
fi

echo "[Restore] =========================================="
echo "[Restore] Redis Restore Utility"
echo "[Restore] =========================================="
echo "[Restore] Backup file: $BACKUP_FILE"
echo "[Restore] Redis host: $REDIS_HOST:$REDIS_PORT"
echo ""

# Function to run Redis commands
redis_cmd() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning "$@"
}

# 1. Verify backup integrity
echo "[Restore] Verifying backup integrity..."
if ! tar -tzf "$BACKUP_FILE" > /dev/null 2>&1; then
    echo "[Restore] ERROR: Backup file is corrupted!"
    exit 1
fi
echo "[Restore] ✓ Backup file is valid"

# 2. Extract backup metadata
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
echo "[Restore] Backup extracted to temporary directory"

# Display backup info
if [ -f "$TEMP_DIR"/*_manifest.json ]; then
    echo ""
    echo "[Restore] Backup Manifest:"
    cat "$TEMP_DIR"/*_manifest.json
    echo ""
fi

# 3. Check current database status
CURRENT_SIZE=$(redis_cmd DBSIZE)
echo "[Restore] Current database size: $CURRENT_SIZE keys"

if [ "$CURRENT_SIZE" -gt 0 ] && [ "$FORCE_RESTORE" != "--force" ]; then
    echo ""
    echo "[Restore] WARNING: Current database contains $CURRENT_SIZE keys"
    echo "[Restore] This restore will OVERWRITE the existing data"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "[Restore] Restore cancelled"
        exit 0
    fi
fi

# 4. Get Redis data directory
RDB_DIR=$(redis_cmd CONFIG GET dir | tail -1)
RDB_FILE=$(redis_cmd CONFIG GET dbfilename | tail -1)
RDB_PATH="${RDB_DIR}/${RDB_FILE}"

echo "[Restore] Redis data directory: $RDB_DIR"
echo "[Restore] RDB filename: $RDB_FILE"

# 5. Create pre-restore backup
if [ "$CURRENT_SIZE" -gt 0 ]; then
    echo "[Restore] Creating pre-restore backup..."
    PRE_RESTORE_BACKUP="${BACKUP_DIR}/pre_restore_$(date +%Y%m%d_%H%M%S).rdb"
    if [ -f "$RDB_PATH" ]; then
        cp "$RDB_PATH" "$PRE_RESTORE_BACKUP"
        echo "[Restore] Pre-restore backup saved to: $PRE_RESTORE_BACKUP"
    fi
fi

# 6. Stop Redis writes (optional, commented out for safety)
# echo "[Restore] Stopping writes to Redis..."
# redis_cmd CONFIG SET save ""
# redis_cmd CONFIG SET appendonly no

# 7. Flush current database
echo "[Restore] Flushing current database..."
redis_cmd FLUSHALL
echo "[Restore] Database flushed"

# 8. Shutdown Redis for file replacement
echo "[Restore] Shutting down Redis..."
redis_cmd SHUTDOWN NOSAVE || true
sleep 2

# Wait for Redis to stop
echo "[Restore] Waiting for Redis to stop..."
for i in {1..30}; do
    if ! redis_cmd PING > /dev/null 2>&1; then
        echo "[Restore] Redis stopped"
        break
    fi
    sleep 1
done

# 9. Replace RDB file
echo "[Restore] Replacing RDB file..."
RDB_BACKUP=$(find "$TEMP_DIR" -name "*.rdb" | head -1)
if [ -f "$RDB_BACKUP" ]; then
    sudo cp "$RDB_BACKUP" "$RDB_PATH"
    sudo chown redis:redis "$RDB_PATH" 2>/dev/null || true
    echo "[Restore] RDB file replaced"
else
    echo "[Restore] ERROR: No RDB file found in backup!"
    exit 1
fi

# 10. Start Redis
echo "[Restore] Starting Redis..."
sudo systemctl start redis-server 2>/dev/null || \
    sudo service redis-server start 2>/dev/null || \
    redis-server --daemonize yes

# Wait for Redis to start
echo "[Restore] Waiting for Redis to start..."
for i in {1..30}; do
    if redis_cmd PING > /dev/null 2>&1; then
        echo "[Restore] Redis started"
        break
    fi
    sleep 1
done

# 11. Verify restore
echo "[Restore] Verifying restore..."
RESTORED_SIZE=$(redis_cmd DBSIZE)
echo "[Restore] Restored database size: $RESTORED_SIZE keys"

# 12. Re-enable persistence
echo "[Restore] Re-enabling persistence..."
redis_cmd CONFIG SET save "900 1 300 10 60 10000"
redis_cmd CONFIG SET appendonly yes
redis_cmd CONFIG REWRITE

# 13. Generate restore report
{
    echo "=== Redis Restore Report ==="
    echo "Timestamp: $(date)"
    echo "Backup File: $BACKUP_FILE"
    echo "Pre-restore Keys: $CURRENT_SIZE"
    echo "Post-restore Keys: $RESTORED_SIZE"
    echo "Redis Host: $REDIS_HOST:$REDIS_PORT"
    echo ""
    echo "=== Restored Database Info ==="
    redis_cmd INFO keyspace
    echo ""
    echo "=== Sample Keys by Prefix ==="
    for prefix in "task" "project" "approval" "library"; do
        count=$(redis_cmd KEYS "${prefix}:*" | wc -l)
        echo "${prefix}: $count keys"
    done
} > "$BACKUP_DIR/latest_restore_report.txt"

echo ""
echo "[Restore] =========================================="
echo "[Restore] ✓ Restore completed successfully"
echo "[Restore] =========================================="
echo "[Restore] Restored $RESTORED_SIZE keys"
echo "[Restore] Report: $BACKUP_DIR/latest_restore_report.txt"

if [ -n "$PRE_RESTORE_BACKUP" ]; then
    echo "[Restore] Pre-restore backup: $PRE_RESTORE_BACKUP"
fi

echo ""

# Optional: Send notification
# curl -X POST https://your-webhook-url \
#   -H "Content-Type: application/json" \
#   -d "{\"text\":\"Redis restore completed: $RESTORED_SIZE keys restored\"}"
