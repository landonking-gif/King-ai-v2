#!/bin/bash
# Redis ACL Setup Script
# Sets up Redis ACL users for Mother-Harness services

set -e

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-mother_dev_password}"

# Load environment if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Validate required passwords are set
REQUIRED_PASSWORDS=(
    "REDIS_ACL_ORCHESTRATOR_PASSWORD"
    "REDIS_ACL_DOCLING_PASSWORD"
    "REDIS_ACL_AGENTS_PASSWORD"
    "REDIS_ACL_DASHBOARD_PASSWORD"
)

for PASSWORD_VAR in "${REQUIRED_PASSWORDS[@]}"; do
    if [ -z "${!PASSWORD_VAR}" ] || [ "${!PASSWORD_VAR}" = "CHANGE_ME" ]; then
        echo "Error: $PASSWORD_VAR is not set or is a placeholder"
        exit 1
    fi
done

echo "Setting up Redis ACL users on ${REDIS_HOST}:${REDIS_PORT}..."

# Connect using redis-cli
REDIS_CLI="redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD}"

# Orchestrator user - full access to tasks, projects, approvals, metrics, costs
echo "Creating orchestrator user..."
$REDIS_CLI ACL SETUSER orchestrator on >"${REDIS_ACL_ORCHESTRATOR_PASSWORD}" \
    +get +set +del +keys +scan \
    +json.get +json.set +json.del +json.arrappend +json.arrindex +json.arrlen \
    +ft.search +ft.create +ft.info +ft.aggregate \
    +xadd +xread +xreadgroup +xack +xgroup +xlen +xpending \
    +hget +hset +hgetall +hincrby +hincrbyfloat +hdel +hkeys \
    +expire +ttl +exists +ping +info \
    ~task:* ~project:* ~approval:* ~model_decision:* ~cost:* ~budget:* ~retry:* \
    ~stream:activity ~metrics:* ~memory:* ~library:* ~pkm:*

# Docling user - access to libraries and document chunks
echo "Creating docling user..."
$REDIS_CLI ACL SETUSER docling on >"${REDIS_ACL_DOCLING_PASSWORD}" \
    +get +set +del +keys +scan \
    +json.get +json.set +json.del +json.arrappend \
    +ft.search +ft.create +ft.info \
    +expire +ttl +exists +ping +info \
    ~library:* ~chunk:* ~docling:*

# Agents user - access to tasks, memory, and libraries (read-heavy)
echo "Creating agents user..."
$REDIS_CLI ACL SETUSER agents on >"${REDIS_ACL_AGENTS_PASSWORD}" \
    +get +set +keys +scan \
    +json.get +json.set \
    +ft.search +ft.aggregate \
    +hget +hgetall \
    +expire +ttl +exists +ping +info \
    ~task:* ~memory:* ~library:* ~chunk:* ~pkm:*

# Dashboard user - read-only access to most data
echo "Creating dashboard user..."
$REDIS_CLI ACL SETUSER dashboard on >"${REDIS_ACL_DASHBOARD_PASSWORD}" \
    +get +keys +scan \
    +json.get \
    +ft.search +ft.aggregate \
    +hget +hgetall \
    +xread +xlen \
    +exists +ping +info \
    ~task:* ~project:* ~approval:* ~metrics:* ~cost:* ~budget:* ~stream:activity

# Verify ACL users
echo ""
echo "Verifying ACL users..."
$REDIS_CLI ACL LIST | grep -E '(orchestrator|docling|agents|dashboard)'

echo ""
echo "Redis ACL setup complete!"
echo ""
echo "Summary:"
echo "  - orchestrator: Full access to tasks, projects, metrics, streams"
echo "  - docling: Access to libraries and document processing"
echo "  - agents: Read access to tasks, memory, libraries"
echo "  - dashboard: Read-only access to all user-facing data"
