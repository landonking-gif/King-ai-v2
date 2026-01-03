#!/bin/bash
# Redis Backup Script for Mother-Harness
# Backs up Redis data and creates timestamped snapshots

set -e

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-mother_dev_password}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/mother-harness/redis}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="redis_backup_${BACKUP_TIMESTAMP}"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

echo "[Backup] Starting Redis backup at $(date)"
echo "[Backup] Host: $REDIS_HOST:$REDIS_PORT"
echo "[Backup] Backup directory: $BACKUP_DIR"

# Function to run Redis commands
redis_cmd() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning "$@"
}

# 1. Trigger BGSAVE for RDB snapshot
echo "[Backup] Triggering BGSAVE..."
redis_cmd BGSAVE

# Wait for BGSAVE to complete
while true; do
    LASTSAVE=$(redis_cmd LASTSAVE)
    sleep 2
    NEWSAVE=$(redis_cmd LASTSAVE)
    if [ "$NEWSAVE" != "$LASTSAVE" ]; then
        echo "[Backup] BGSAVE completed"
        break
    fi
    echo "[Backup] Waiting for BGSAVE to complete..."
done

# 2. Get Redis info
echo "[Backup] Gathering Redis info..."
redis_cmd INFO > "$BACKUP_DIR/${BACKUP_NAME}_info.txt"

# 3. Get database size statistics
echo "[Backup] Getting database statistics..."
{
    echo "=== Database Keys Count ==="
    redis_cmd DBSIZE
    echo ""
    echo "=== Memory Usage ==="
    redis_cmd INFO memory | grep used_memory_human
    echo ""
    echo "=== Keyspace Info ==="
    redis_cmd INFO keyspace
} > "$BACKUP_DIR/${BACKUP_NAME}_stats.txt"

# 4. Export critical keys to JSON (for inspection/verification)
echo "[Backup] Exporting critical keys..."
{
    echo "{"
    echo "  \"metadata\": {"
    echo "    \"backup_timestamp\": \"$BACKUP_TIMESTAMP\","
    echo "    \"redis_host\": \"$REDIS_HOST\","
    echo "    \"redis_port\": $REDIS_PORT"
    echo "  },"
    echo "  \"key_counts\": {"

    # Count keys by prefix
    for prefix in "task" "project" "approval" "library" "chunk" "memory" "cost" "budget"; do
        count=$(redis_cmd KEYS "${prefix}:*" | wc -l)
        echo "    \"${prefix}\": $count,"
    done

    echo "    \"total\": $(redis_cmd DBSIZE)"
    echo "  }"
    echo "}"
} > "$BACKUP_DIR/${BACKUP_NAME}_manifest.json"

# 5. Copy RDB file
echo "[Backup] Copying RDB snapshot..."
RDB_DIR=$(redis_cmd CONFIG GET dir | tail -1)
RDB_FILE=$(redis_cmd CONFIG GET dbfilename | tail -1)
RDB_PATH="${RDB_DIR}/${RDB_FILE}"

if [ -f "$RDB_PATH" ]; then
    cp "$RDB_PATH" "$BACKUP_DIR/${BACKUP_NAME}.rdb"
    echo "[Backup] RDB file copied to $BACKUP_DIR/${BACKUP_NAME}.rdb"
else
    echo "[Backup] WARNING: RDB file not found at $RDB_PATH"
fi

# 6. Create compressed archive
echo "[Backup] Creating compressed archive..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" \
    "${BACKUP_NAME}.rdb" \
    "${BACKUP_NAME}_info.txt" \
    "${BACKUP_NAME}_stats.txt" \
    "${BACKUP_NAME}_manifest.json" 2>/dev/null || true

# Remove uncompressed files
rm -f "${BACKUP_NAME}.rdb" \
    "${BACKUP_NAME}_info.txt" \
    "${BACKUP_NAME}_stats.txt" \
    "${BACKUP_NAME}_manifest.json"

# 7. Calculate backup size
BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)
echo "[Backup] Compressed backup size: $BACKUP_SIZE"

# 8. Clean up old backups (keep last N days)
echo "[Backup] Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "redis_backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
REMAINING_BACKUPS=$(find "$BACKUP_DIR" -name "redis_backup_*.tar.gz" -type f | wc -l)
echo "[Backup] Remaining backups: $REMAINING_BACKUPS"

# 9. Verify backup integrity
echo "[Backup] Verifying backup integrity..."
if tar -tzf "${BACKUP_NAME}.tar.gz" > /dev/null 2>&1; then
    echo "[Backup] Backup integrity verified"
else
    echo "[Backup] ERROR: Backup verification failed!"
    exit 1
fi

# 10. Generate backup report
{
    echo "=== Redis Backup Report ==="
    echo "Timestamp: $(date)"
    echo "Backup Name: ${BACKUP_NAME}.tar.gz"
    echo "Backup Size: $BACKUP_SIZE"
    echo "Backup Path: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    echo "Retention Policy: $RETENTION_DAYS days"
    echo "Total Backups: $REMAINING_BACKUPS"
    echo ""
    echo "=== Key Statistics ==="
    cat "${BACKUP_NAME}_stats.txt" 2>/dev/null || echo "Stats not available"
} > "$BACKUP_DIR/latest_backup_report.txt"

echo ""
echo "[Backup] ✓ Backup completed successfully"
echo "[Backup] Backup file: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "[Backup] Report: $BACKUP_DIR/latest_backup_report.txt"
echo ""

# Optional: Send notification (uncomment and configure as needed)
# curl -X POST https://your-webhook-url \
#   -H "Content-Type: application/json" \
#   -d "{\"text\":\"Redis backup completed: ${BACKUP_NAME}.tar.gz ($BACKUP_SIZE)\"}"
