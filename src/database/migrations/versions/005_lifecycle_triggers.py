"""Add database triggers for lifecycle audit logging

Revision ID: 005_lifecycle_triggers
Revises: 004_audit_evolution
Create Date: 2025-12-31 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005_lifecycle_triggers'
down_revision = '004_audit_evolution'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add PostgreSQL triggers for:
    1. Automatic updated_at timestamp
    2. Lifecycle status change audit logging
    3. Business unit KPI change tracking
    4. Evolution proposal state tracking
    """
    
    # Create lifecycle_audit table for tracking all status changes
    op.create_table(
        'lifecycle_audit',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('business_id', sa.String(36), sa.ForeignKey('business_units.id'), nullable=False),
        sa.Column('previous_status', sa.String(50), nullable=True),
        sa.Column('new_status', sa.String(50), nullable=False),
        sa.Column('changed_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('changed_by', sa.String(100), nullable=True),
        sa.Column('reason', sa.Text, nullable=True),
        sa.Column('metadata', sa.JSON, default={}),
    )
    
    # Create index for faster queries
    op.create_index('idx_lifecycle_audit_business', 'lifecycle_audit', ['business_id'])
    op.create_index('idx_lifecycle_audit_changed_at', 'lifecycle_audit', ['changed_at'])
    
    # Create kpi_history table for tracking KPI changes over time
    op.create_table(
        'kpi_history',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('business_id', sa.String(36), sa.ForeignKey('business_units.id'), nullable=False),
        sa.Column('kpi_name', sa.String(100), nullable=False),
        sa.Column('old_value', sa.Float, nullable=True),
        sa.Column('new_value', sa.Float, nullable=False),
        sa.Column('changed_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('source', sa.String(50), nullable=True),  # 'agent', 'api', 'sync', etc.
    )
    
    # Create indexes
    op.create_index('idx_kpi_history_business', 'kpi_history', ['business_id'])
    op.create_index('idx_kpi_history_kpi_name', 'kpi_history', ['kpi_name'])
    op.create_index('idx_kpi_history_changed_at', 'kpi_history', ['changed_at'])
    
    # Create evolution_audit table
    op.create_table(
        'evolution_audit',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('proposal_id', sa.String(36), sa.ForeignKey('evolution_proposals.id'), nullable=False),
        sa.Column('previous_status', sa.String(50), nullable=True),
        sa.Column('new_status', sa.String(50), nullable=False),
        sa.Column('changed_at', sa.DateTime, server_default=sa.func.now(), nullable=False),
        sa.Column('changed_by', sa.String(100), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
    )
    
    op.create_index('idx_evolution_audit_proposal', 'evolution_audit', ['proposal_id'])
    
    # ==========================================================================
    # PostgreSQL Trigger Functions and Triggers
    # ==========================================================================
    
    # 1. Function to auto-update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Apply to business_units table
    op.execute("""
        DROP TRIGGER IF EXISTS update_business_units_updated_at ON business_units;
        CREATE TRIGGER update_business_units_updated_at
            BEFORE UPDATE ON business_units
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    # Apply to tasks table
    op.execute("""
        DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;
        CREATE TRIGGER update_tasks_updated_at
            BEFORE UPDATE ON tasks
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    """)
    
    # 2. Function to audit business status changes
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_business_status_change()
        RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.status IS DISTINCT FROM NEW.status THEN
                INSERT INTO lifecycle_audit (
                    id,
                    business_id,
                    previous_status,
                    new_status,
                    changed_at,
                    changed_by,
                    reason,
                    metadata
                ) VALUES (
                    gen_random_uuid()::text,
                    NEW.id,
                    OLD.status::text,
                    NEW.status::text,
                    NOW(),
                    COALESCE(current_setting('app.current_user', true), 'system'),
                    COALESCE(current_setting('app.change_reason', true), NULL),
                    jsonb_build_object(
                        'old_revenue', OLD.total_revenue,
                        'new_revenue', NEW.total_revenue,
                        'old_expenses', OLD.total_expenses,
                        'new_expenses', NEW.total_expenses
                    )
                );
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS audit_business_status ON business_units;
        CREATE TRIGGER audit_business_status
            AFTER UPDATE ON business_units
            FOR EACH ROW
            EXECUTE FUNCTION audit_business_status_change();
    """)
    
    # 3. Function to validate lifecycle transitions
    op.execute("""
        CREATE OR REPLACE FUNCTION validate_lifecycle_transition()
        RETURNS TRIGGER AS $$
        DECLARE
            valid_transition BOOLEAN := FALSE;
        BEGIN
            -- Skip validation if status hasn't changed
            IF OLD.status = NEW.status THEN
                RETURN NEW;
            END IF;
            
            -- Define valid transitions
            -- DISCOVERY -> VALIDATION -> SETUP -> OPERATION -> OPTIMIZATION -> REPLICATION
            -- Any state -> SUNSET (always allowed)
            
            CASE OLD.status::text
                WHEN 'discovery' THEN
                    valid_transition := NEW.status::text IN ('validation', 'sunset');
                WHEN 'validation' THEN
                    valid_transition := NEW.status::text IN ('setup', 'discovery', 'sunset');
                WHEN 'setup' THEN
                    valid_transition := NEW.status::text IN ('operation', 'sunset');
                WHEN 'operation' THEN
                    valid_transition := NEW.status::text IN ('optimization', 'sunset');
                WHEN 'optimization' THEN
                    valid_transition := NEW.status::text IN ('replication', 'operation', 'sunset');
                WHEN 'replication' THEN
                    valid_transition := NEW.status::text IN ('sunset');
                WHEN 'sunset' THEN
                    -- Cannot transition from sunset (terminal state)
                    valid_transition := FALSE;
                ELSE
                    valid_transition := TRUE;
            END CASE;
            
            -- Allow if using force flag
            IF current_setting('app.force_transition', true) = 'true' THEN
                valid_transition := TRUE;
            END IF;
            
            IF NOT valid_transition THEN
                RAISE EXCEPTION 'Invalid lifecycle transition from % to %', 
                    OLD.status::text, NEW.status::text
                    USING ERRCODE = 'check_violation';
            END IF;
            
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS validate_lifecycle ON business_units;
        CREATE TRIGGER validate_lifecycle
            BEFORE UPDATE ON business_units
            FOR EACH ROW
            EXECUTE FUNCTION validate_lifecycle_transition();
    """)
    
    # 4. Function to audit evolution proposal status changes
    op.execute("""
        CREATE OR REPLACE FUNCTION audit_evolution_status_change()
        RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.status IS DISTINCT FROM NEW.status THEN
                INSERT INTO evolution_audit (
                    id,
                    proposal_id,
                    previous_status,
                    new_status,
                    changed_at,
                    changed_by,
                    notes
                ) VALUES (
                    gen_random_uuid()::text,
                    NEW.id,
                    OLD.status::text,
                    NEW.status::text,
                    NOW(),
                    COALESCE(current_setting('app.current_user', true), 'system'),
                    COALESCE(current_setting('app.approval_notes', true), NULL)
                );
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS audit_evolution_status ON evolution_proposals;
        CREATE TRIGGER audit_evolution_status
            AFTER UPDATE ON evolution_proposals
            FOR EACH ROW
            EXECUTE FUNCTION audit_evolution_status_change();
    """)
    
    # 5. Function to track significant revenue/expense changes
    op.execute("""
        CREATE OR REPLACE FUNCTION track_financial_changes()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Track revenue changes > 10%
            IF OLD.total_revenue > 0 AND 
               ABS(NEW.total_revenue - OLD.total_revenue) / OLD.total_revenue > 0.10 THEN
                INSERT INTO kpi_history (
                    id, business_id, kpi_name, old_value, new_value, changed_at, source
                ) VALUES (
                    gen_random_uuid()::text,
                    NEW.id,
                    'total_revenue',
                    OLD.total_revenue,
                    NEW.total_revenue,
                    NOW(),
                    COALESCE(current_setting('app.change_source', true), 'system')
                );
            END IF;
            
            -- Track expense changes > 10%
            IF OLD.total_expenses > 0 AND 
               ABS(NEW.total_expenses - OLD.total_expenses) / OLD.total_expenses > 0.10 THEN
                INSERT INTO kpi_history (
                    id, business_id, kpi_name, old_value, new_value, changed_at, source
                ) VALUES (
                    gen_random_uuid()::text,
                    NEW.id,
                    'total_expenses',
                    OLD.total_expenses,
                    NEW.total_expenses,
                    NOW(),
                    COALESCE(current_setting('app.change_source', true), 'system')
                );
            END IF;
            
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS track_financials ON business_units;
        CREATE TRIGGER track_financials
            AFTER UPDATE ON business_units
            FOR EACH ROW
            EXECUTE FUNCTION track_financial_changes();
    """)
    
    # 6. Add helper function for getting lifecycle history
    op.execute("""
        CREATE OR REPLACE FUNCTION get_lifecycle_history(p_business_id TEXT)
        RETURNS TABLE (
            status TEXT,
            changed_at TIMESTAMP,
            duration_hours NUMERIC,
            changed_by TEXT,
            reason TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                la.new_status,
                la.changed_at,
                EXTRACT(EPOCH FROM (
                    LEAD(la.changed_at) OVER (ORDER BY la.changed_at) - la.changed_at
                )) / 3600 as duration_hours,
                la.changed_by,
                la.reason
            FROM lifecycle_audit la
            WHERE la.business_id = p_business_id
            ORDER BY la.changed_at;
        END;
        $$ language 'plpgsql';
    """)
    
    # 7. Add view for current lifecycle status with metrics
    op.execute("""
        CREATE OR REPLACE VIEW business_lifecycle_view AS
        SELECT 
            b.id,
            b.name,
            b.type,
            b.status::text as current_status,
            b.total_revenue,
            b.total_expenses,
            b.total_revenue - b.total_expenses as net_profit,
            b.created_at,
            b.updated_at,
            (SELECT COUNT(*) FROM lifecycle_audit la WHERE la.business_id = b.id) as transition_count,
            (SELECT la.changed_at FROM lifecycle_audit la 
             WHERE la.business_id = b.id ORDER BY la.changed_at DESC LIMIT 1) as last_transition_at,
            EXTRACT(EPOCH FROM (NOW() - b.created_at)) / 86400 as days_since_created,
            CASE 
                WHEN b.total_expenses > 0 THEN 
                    (b.total_revenue - b.total_expenses) / b.total_expenses * 100
                ELSE 0 
            END as roi_percent
        FROM business_units b;
    """)


def downgrade() -> None:
    """Remove all triggers and audit tables."""
    
    # Drop view
    op.execute("DROP VIEW IF EXISTS business_lifecycle_view;")
    
    # Drop helper function
    op.execute("DROP FUNCTION IF EXISTS get_lifecycle_history(TEXT);")
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS track_financials ON business_units;")
    op.execute("DROP TRIGGER IF EXISTS audit_evolution_status ON evolution_proposals;")
    op.execute("DROP TRIGGER IF EXISTS validate_lifecycle ON business_units;")
    op.execute("DROP TRIGGER IF EXISTS audit_business_status ON business_units;")
    op.execute("DROP TRIGGER IF EXISTS update_tasks_updated_at ON tasks;")
    op.execute("DROP TRIGGER IF EXISTS update_business_units_updated_at ON business_units;")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS track_financial_changes();")
    op.execute("DROP FUNCTION IF EXISTS audit_evolution_status_change();")
    op.execute("DROP FUNCTION IF EXISTS validate_lifecycle_transition();")
    op.execute("DROP FUNCTION IF EXISTS audit_business_status_change();")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column();")
    
    # Drop indexes
    op.drop_index('idx_evolution_audit_proposal', table_name='evolution_audit')
    op.drop_index('idx_kpi_history_changed_at', table_name='kpi_history')
    op.drop_index('idx_kpi_history_kpi_name', table_name='kpi_history')
    op.drop_index('idx_kpi_history_business', table_name='kpi_history')
    op.drop_index('idx_lifecycle_audit_changed_at', table_name='lifecycle_audit')
    op.drop_index('idx_lifecycle_audit_business', table_name='lifecycle_audit')
    
    # Drop tables
    op.drop_table('evolution_audit')
    op.drop_table('kpi_history')
    op.drop_table('lifecycle_audit')
