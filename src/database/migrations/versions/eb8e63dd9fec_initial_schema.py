"""Initial schema

Revision ID: eb8e63dd9fec
Revises: 
Create Date: 2025-12-28 22:48:29.038307

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eb8e63dd9fec'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- business_units ---
    op.create_table(
        'business_units',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.Enum('DISCOVERY', 'VALIDATION', 'SETUP', 'OPERATION', 'OPTIMIZATION', 'REPLICATION', 'SUNSET', name='businessstatus'), nullable=True),
        sa.Column('playbook_id', sa.String(length=50), nullable=True),
        sa.Column('total_revenue', sa.Float(), nullable=True),
        sa.Column('total_expenses', sa.Float(), nullable=True),
        sa.Column('kpis', sa.JSON(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # --- tasks ---
    op.create_table(
        'tasks',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('business_id', sa.String(length=36), sa.ForeignKey('business_units.id'), nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True),
        sa.Column('agent', sa.String(length=50), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=True),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('requires_approval', sa.Boolean(), nullable=True),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('approved_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # --- evolution_proposals ---
    op.create_table(
        'evolution_proposals',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('type', sa.String(length=20), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('proposed_changes', sa.JSON(), nullable=False),
        sa.Column('expected_impact', sa.Text(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'APPROVED', 'REJECTED', 'APPLIED', 'ROLLED_BACK', name='evolutionstatus'), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # --- conversation_messages ---
    op.create_table(
        'conversation_messages',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('role', sa.String(length=20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # --- logs ---
    op.create_table(
        'logs',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('business_id', sa.String(length=36), sa.ForeignKey('business_units.id'), nullable=True),
        sa.Column('level', sa.String(length=10), nullable=False),
        sa.Column('module', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('logs')
    op.drop_table('conversation_messages')
    op.drop_table('evolution_proposals')
    op.drop_table('tasks')
    op.drop_table('business_units')
    # Optional: drop enums if using Postgres
    # sa.Enum(name='businessstatus').drop(op.get_bind(), checkfirst=False)
    # sa.Enum(name='evolutionstatus').drop(op.get_bind(), checkfirst=False)
