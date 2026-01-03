# Redis Backup and Restore Strategy

## Overview

Mother-Harness uses Redis Stack as its primary data store for tasks, projects, approvals, memory, and all operational data. A robust backup and restore strategy is critical for data protection and disaster recovery.

## Backup Strategy

### Automated Daily Backups

Backups are automated using cron jobs to run daily at 2 AM:

```bash
# Add to crontab
0 2 * * * /path/to/mother-harness/scripts/backup-redis.sh >> /var/log/mother-harness/backup.log 2>&1
```

### Backup Components

Each backup includes:

1. **RDB Snapshot**: Binary snapshot of the entire Redis database
2. **Info Dump**: Redis server information and configuration
3. **Statistics**: Key counts and memory usage by prefix
4. **Manifest**: JSON metadata about the backup

### Backup Process

The [backup-redis.sh](../scripts/backup-redis.sh) script:

1. Triggers `BGSAVE` for non-blocking snapshot creation
2. Waits for background save to complete
3. Collects database statistics and metadata
4. Copies the RDB file to backup directory
5. Creates compressed archive (`.tar.gz`)
6. Cleans up backups older than retention period (default: 30 days)
7. Verifies backup integrity
8. Generates backup report

### Running Manual Backup

```bash
# Basic backup
./scripts/backup-redis.sh

# Custom backup directory
BACKUP_DIR=/mnt/backups ./scripts/backup-redis.sh

# Custom retention (keep 90 days)
RETENTION_DAYS=90 ./scripts/backup-redis.sh

# Remote Redis instance
REDIS_HOST=prod-redis.example.com \
REDIS_PASSWORD=secret \
./scripts/backup-redis.sh
```

### Backup Locations

- **Default Directory**: `/var/backups/mother-harness/redis/`
- **Filename Format**: `redis_backup_YYYYMMDD_HHMMSS.tar.gz`
- **Latest Report**: `latest_backup_report.txt`

### Backup Retention

- **Default**: 30 days
- **Configurable**: Set `RETENTION_DAYS` environment variable
- **Minimum Recommended**: 7 days
- **Production Recommended**: 90 days

### Off-Site Backups

For production, copy backups to off-site storage:

```bash
# Copy to S3
aws s3 sync /var/backups/mother-harness/redis/ \
  s3://your-bucket/redis-backups/ \
  --storage-class STANDARD_IA

# Copy to network storage
rsync -avz /var/backups/mother-harness/redis/ \
  backup-server:/backups/mother-harness/redis/

# Copy to cloud storage
rclone sync /var/backups/mother-harness/redis/ \
  remote:mother-harness-backups/redis/
```

## Restore Strategy

### Restore Process

The [restore-redis.sh](../scripts/restore-redis.sh) script:

1. Verifies backup file integrity
2. Displays backup metadata
3. Checks current database size
4. Prompts for confirmation (unless `--force` flag used)
5. Creates pre-restore backup of current state
6. Flushes current database
7. Shuts down Redis
8. Replaces RDB file
9. Restarts Redis
10. Verifies restored data
11. Re-enables persistence
12. Generates restore report

### Running a Restore

```bash
# List available backups
./scripts/restore-redis.sh

# Restore specific backup (with confirmation)
./scripts/restore-redis.sh redis_backup_20241222_020000.tar.gz

# Force restore without confirmation
./scripts/restore-redis.sh redis_backup_20241222_020000.tar.gz --force

# Restore from custom location
./scripts/restore-redis.sh /path/to/backup.tar.gz
```

### Restore Safety Features

1. **Integrity Check**: Validates backup file before proceeding
2. **Confirmation Prompt**: Requires explicit confirmation if database has data
3. **Pre-restore Backup**: Creates backup of current state before overwriting
4. **Verification**: Confirms key counts after restore

### Restore Scenarios

#### Scenario 1: Complete Data Loss

```bash
# Restore from latest backup
LATEST_BACKUP=$(ls -t /var/backups/mother-harness/redis/*.tar.gz | head -1)
./scripts/restore-redis.sh "$LATEST_BACKUP" --force
```

#### Scenario 2: Rollback to Previous State

```bash
# Find backup from specific date
BACKUP=$(ls /var/backups/mother-harness/redis/redis_backup_20241220*.tar.gz)
./scripts/restore-redis.sh "$BACKUP"
```

#### Scenario 3: Clone Production to Staging

```bash
# On production server
./scripts/backup-redis.sh
scp /var/backups/mother-harness/redis/latest.tar.gz staging:/tmp/

# On staging server
./scripts/restore-redis.sh /tmp/latest.tar.gz --force
```

## Testing Backup and Restore

### Test Backup Creation

```bash
# Create test backup
./scripts/backup-redis.sh

# Verify backup exists
ls -lh /var/backups/mother-harness/redis/

# Check backup contents
tar -tzf /var/backups/mother-harness/redis/redis_backup_*.tar.gz

# View backup report
cat /var/backups/mother-harness/redis/latest_backup_report.txt
```

### Test Restore Process

**Important**: Only test restore on non-production instances!

```bash
# 1. Create current state backup
./scripts/backup-redis.sh
CURRENT_BACKUP=$(ls -t /var/backups/mother-harness/redis/*.tar.gz | head -1)

# 2. Create some test data
redis-cli SET test_key "test_value"
redis-cli SET test_timestamp "$(date)"

# 3. Create backup with test data
./scripts/backup-redis.sh
TEST_BACKUP=$(ls -t /var/backups/mother-harness/redis/*.tar.gz | head -1)

# 4. Restore to previous state
./scripts/restore-redis.sh "$CURRENT_BACKUP" --force

# 5. Verify test keys are gone
redis-cli GET test_key  # Should return (nil)

# 6. Restore test backup
./scripts/restore-redis.sh "$TEST_BACKUP" --force

# 7. Verify test keys are back
redis-cli GET test_key  # Should return "test_value"
```

## Monitoring and Alerts

### Backup Monitoring

Monitor backup success with these checks:

```bash
# Check last backup time
ls -lh /var/backups/mother-harness/redis/redis_backup_*.tar.gz | tail -1

# Verify recent backups
find /var/backups/mother-harness/redis/ -name "*.tar.gz" -mtime -1

# Check backup size trends
du -h /var/backups/mother-harness/redis/redis_backup_*.tar.gz | tail -10
```

### Alert Conditions

Set up alerts for:

1. **Backup Failure**: No backup created in last 24 hours
2. **Backup Size Anomaly**: Backup size differs significantly from average
3. **Disk Space**: Backup directory > 80% full
4. **Integrity Issues**: Backup verification fails

### Example Monitoring Script

```bash
#!/bin/bash
# Check if backup ran in last 24 hours

BACKUP_DIR="/var/backups/mother-harness/redis"
LATEST_BACKUP=$(find "$BACKUP_DIR" -name "*.tar.gz" -mtime -1 | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "ERROR: No backup found in last 24 hours!"
    # Send alert
    curl -X POST https://alerts.example.com/webhook \
      -d '{"text":"Redis backup failed or missing"}'
    exit 1
fi

echo "✓ Backup is up to date: $LATEST_BACKUP"
```

## Disaster Recovery Plan

### RPO and RTO

- **Recovery Point Objective (RPO)**: 24 hours (daily backups)
- **Recovery Time Objective (RTO)**: 15 minutes (time to restore)

### Recovery Procedures

#### Step 1: Assess Situation

```bash
# Check if Redis is responding
redis-cli PING

# Check database size
redis-cli DBSIZE

# Check for data corruption
redis-cli --rdb /tmp/check.rdb
```

#### Step 2: Determine Recovery Point

```bash
# List available backups
ls -lh /var/backups/mother-harness/redis/

# Check backup manifests
for backup in /var/backups/mother-harness/redis/*.tar.gz; do
    echo "=== $backup ==="
    tar -xzOf "$backup" "*_manifest.json" 2>/dev/null || echo "No manifest"
    echo ""
done
```

#### Step 3: Execute Restore

```bash
# Choose backup
RESTORE_BACKUP="/var/backups/mother-harness/redis/redis_backup_20241222_020000.tar.gz"

# Execute restore
./scripts/restore-redis.sh "$RESTORE_BACKUP" --force
```

#### Step 4: Verify Recovery

```bash
# Check key counts
redis-cli DBSIZE

# Sample data verification
redis-cli KEYS "task:*" | head -5
redis-cli KEYS "project:*" | head -5

# Check application health
curl http://localhost:8000/health
```

#### Step 5: Resume Operations

```bash
# Restart dependent services
docker-compose restart orchestrator docling dashboard

# Monitor logs
docker-compose logs -f
```

## Best Practices

### Backup Best Practices

1. **Regular Testing**: Test restore process monthly
2. **Multiple Locations**: Store backups in at least 2 locations
3. **Encryption**: Encrypt backups for sensitive data
4. **Versioning**: Keep multiple backup versions
5. **Documentation**: Document restore procedures
6. **Automation**: Automate backup verification
7. **Monitoring**: Alert on backup failures

### Restore Best Practices

1. **Test Environment First**: Always test restore in non-production first
2. **Verify Integrity**: Check backup integrity before restoring
3. **Communication**: Notify team before production restores
4. **Backup Current State**: Always backup current state before restoring
5. **Staged Approach**: Consider restoring to staging, then promoting to production
6. **Post-Restore Validation**: Thoroughly verify data after restore
7. **Document Issues**: Log any problems encountered during restore

## Troubleshooting

### Backup Issues

**Problem**: BGSAVE fails with "Can't save in background: fork: Cannot allocate memory"

**Solution**:
```bash
# Increase overcommit memory
sudo sysctl vm.overcommit_memory=1
echo "vm.overcommit_memory = 1" | sudo tee -a /etc/sysctl.conf
```

**Problem**: Backup file is unexpectedly large

**Solution**:
```bash
# Check for memory fragmentation
redis-cli INFO memory | grep frag

# Analyze key distribution
redis-cli --bigkeys
```

### Restore Issues

**Problem**: Redis won't start after restore

**Solution**:
```bash
# Check Redis logs
sudo tail -f /var/log/redis/redis-server.log

# Verify RDB file
redis-check-rdb /var/lib/redis/dump.rdb

# Check file permissions
ls -l /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
```

**Problem**: Data missing after restore

**Solution**:
```bash
# Verify backup contains expected data
tar -xzOf backup.tar.gz "*.rdb" | redis-check-rdb -

# Check restore report
cat /var/backups/mother-harness/redis/latest_restore_report.txt
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server hostname |
| `REDIS_PORT` | 6379 | Redis server port |
| `REDIS_PASSWORD` | mother_dev_password | Redis authentication password |
| `BACKUP_DIR` | /var/backups/mother-harness/redis | Backup storage directory |
| `RETENTION_DAYS` | 30 | Days to keep old backups |

### Cron Schedule Examples

```bash
# Daily at 2 AM
0 2 * * * /path/to/backup-redis.sh

# Every 6 hours
0 */6 * * * /path/to/backup-redis.sh

# Weekdays at midnight
0 0 * * 1-5 /path/to/backup-redis.sh

# Monthly on first day at 1 AM
0 1 1 * * /path/to/backup-redis.sh
```

## Appendix

### Quick Reference Commands

```bash
# Create backup
./scripts/backup-redis.sh

# List backups
ls -lh /var/backups/mother-harness/redis/

# Restore backup
./scripts/restore-redis.sh backup_file.tar.gz

# Verify backup
tar -tzf backup_file.tar.gz

# Check Redis status
redis-cli PING && redis-cli DBSIZE

# Manual RDB save
redis-cli BGSAVE
```

### Related Documentation

- [Redis Persistence Documentation](https://redis.io/topics/persistence)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [Operations Guide](operations-guide.md)
