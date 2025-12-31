"""Add audit trail and evolution history tables

Revision ID: 004_audit_evolution
Revises: 
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '004_audit_evolution'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create audit_trail and evolution_history tables."""
    
    # Audit trail table
    op.create_table(
        'audit_trail',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('event_type', sa.String(100), nullable=False, index=True),
        sa.Column('severity', sa.String(20), nullable=False, index=True),
        sa.Column('timestamp', sa.DateTime, nullable=False, index=True),
        sa.Column('actor_id', sa.String(36), nullable=True, index=True),
        sa.Column('actor_type', sa.String(20), nullable=False, default='system'),
        sa.Column('resource_type', sa.String(50), nullable=False, index=True),
        sa.Column('resource_id', sa.String(100), nullable=False),
        sa.Column('action', sa.String(500), nullable=False),
        sa.Column('old_value', JSONB, nullable=True),
        sa.Column('new_value', JSONB, nullable=True),
        sa.Column('metadata', JSONB, nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('request_id', sa.String(36), nullable=True),
        sa.Column('business_id', sa.String(36), nullable=True, index=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
    )
    
    # Create indexes for common queries
    op.create_index(
        'ix_audit_trail_timestamp_event_type',
        'audit_trail',
        ['timestamp', 'event_type']
    )
    op.create_index(
        'ix_audit_trail_business_timestamp',
        'audit_trail',
        ['business_id', 'timestamp']
    )
    
    # Evolution history table
    op.create_table(
        'evolution_history',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('proposal_id', sa.String(36), nullable=False, index=True),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('event_data', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=False, default=sa.func.now()),
        sa.UniqueConstraint('proposal_id', 'event_type', name='uq_evolution_proposal_event')
    )
    
    # Create index for querying by proposal
    op.create_index(
        'ix_evolution_history_proposal_created',
        'evolution_history',
        ['proposal_id', 'created_at']
    )


def downgrade() -> None:
    """Drop audit and evolution tables."""
    op.drop_table('evolution_history')
    op.drop_table('audit_trail')
