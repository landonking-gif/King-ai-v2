"""Initial schema with artifacts and provenance_logs tables

Revision ID: 20251229_0001
Revises:
Create Date: 2025-12-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20251229_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create artifacts table
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("artifact_type", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("safety_class", sa.String(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("extra_metadata", sa.JSON(), nullable=False),
        sa.Column("embedding_ref", sa.String(), nullable=True),
        sa.Column("cold_storage_ref", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifacts_artifact_type"), "artifacts", ["artifact_type"], unique=False)
    op.create_index(op.f("ix_artifacts_content_hash"), "artifacts", ["content_hash"], unique=False)
    op.create_index(op.f("ix_artifacts_session_id"), "artifacts", ["session_id"], unique=False)

    # Create provenance_logs table
    op.create_table(
        "provenance_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("artifact_id", sa.String(), nullable=False),
        sa.Column("actor_id", sa.String(), nullable=False),
        sa.Column("actor_type", sa.String(), nullable=False),
        sa.Column("inputs_hash", sa.String(), nullable=False),
        sa.Column("outputs_hash", sa.String(), nullable=False),
        sa.Column("tool_ids", sa.JSON(), nullable=False),
        sa.Column("parent_artifact_ids", sa.JSON(), nullable=False),
        sa.Column("extra_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_provenance_logs_artifact_id"),
        "provenance_logs",
        ["artifact_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index(op.f("ix_provenance_logs_artifact_id"), table_name="provenance_logs")
    op.drop_table("provenance_logs")

    op.drop_index(op.f("ix_artifacts_session_id"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_content_hash"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_artifact_type"), table_name="artifacts")
    op.drop_table("artifacts")
