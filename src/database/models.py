from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Enum, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

# Base class for all SQLAlchemy models
Base = declarative_base()

class BusinessStatus(enum.Enum):
    """Enumeration of the possible lifecycle stages of a business unit."""
    DISCOVERY = "discovery"       # Market research phase
    VALIDATION = "validation"     # Concept testing phase
    SETUP = "setup"              # Initial operations setup
    OPERATION = "operation"      # Active and generating data
    OPTIMIZATION = "optimization" # Refining for better ROI
    REPLICATION = "replication"  # Scaling the model
    SUNSET = "sunset"           # Winding down operations

class EvolutionStatus(enum.Enum):
    """Tracks the state of self-modification proposals."""
    PENDING = "pending"     # Waiting for human review
    APPROVED = "approved"   # User accepted the change
    REJECTED = "rejected"   # User declined the change
    APPLIED = "applied"    # System successfully modified itself
    ROLLED_BACK = "rolled_back" # Modification caused error and was reverted

class BusinessUnit(Base):
    """
    Represents an independent business venture managed by the Master AI.
    Each unit tracks its own financials, KPIs, and operational state.
    """
    __tablename__ = "business_units"
    
    id: str = Column(String(36), primary_key=True)
    name: str = Column(String(255), nullable=False)
    type: str = Column(String(50), nullable=False)  # e.g., dropshipping, saas, content_site
    status: BusinessStatus = Column(Enum(BusinessStatus), default=BusinessStatus.DISCOVERY)
    playbook_id: str = Column(String(50), nullable=True) # Ref to specific strategy files
    
    # Financials
    total_revenue: float = Column(Float, default=0.0)
    total_expenses: float = Column(Float, default=0.0)
    
    # Key Performance Indicators stored as a flexible JSON blob
    kpis: dict = Column(JSON, default={})
    
    # Metadata and operational configuration
    config: dict = Column(JSON, default={})
    created_at: datetime = Column(DateTime, server_default=func.now())
    updated_at: datetime = Column(DateTime, onupdate=func.now())
    
    # Relationships to child entities
    tasks = relationship("Task", back_populates="business")
    logs = relationship("Log", back_populates="business")

class Task(Base):
    """
    A specific unit of work delegated to an agent within a business unit.
    """
    __tablename__ = "tasks"
    
    id: str = Column(String(36), primary_key=True)
    business_id: str = Column(String(36), ForeignKey("business_units.id"), nullable=True)
    
    name: str = Column(String(255), nullable=False)
    description: str = Column(Text, nullable=True)
    type: str = Column(String(50), nullable=False)  # research, setup, content_generation, etc.
    status: str = Column(String(20), default="pending")  # pending, running, completed, failed
    
    # Execution details
    agent: str = Column(String(50), nullable=True)  # The specialized sub-agent handling this task
    input_data: dict = Column(JSON, default={})    # Parameters passed to the agent
    output_data: dict = Column(JSON, default={})   # Results returned by the agent
    
    # Approval gating for risk management
    requires_approval: bool = Column(Integer, default=False)
    approved_at: datetime = Column(DateTime, nullable=True)
    approved_by: str = Column(String(100), nullable=True)
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    completed_at: datetime = Column(DateTime, nullable=True)
    
    # Links back to the parent business unit
    business = relationship("BusinessUnit", back_populates="tasks")

class EvolutionProposal(Base):
    """
    Proposals from the Master AI to modify its own source code or architecture.
    """
    __tablename__ = "evolution_proposals"
    
    id: str = Column(String(36), primary_key=True)
    type: str = Column(String(20), nullable=False)  # code_mod, ml_retrain, arch_update
    
    description: str = Column(Text, nullable=False) # Natural language description of the change
    rationale: str = Column(Text, nullable=False)   # Why this change is beneficial
    proposed_changes: dict = Column(JSON, nullable=False)  # Map of file paths to diffs/content
    expected_impact: str = Column(Text, nullable=True)  # Predicted improvement in metrics
    confidence_score: float = Column(Float, default=0.0) # AI's self-assessed confidence
    
    status: EvolutionStatus = Column(Enum(EvolutionStatus), default=EvolutionStatus.PENDING)
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    reviewed_at: datetime = Column(DateTime, nullable=True)
    applied_at: datetime = Column(DateTime, nullable=True)

class ConversationMessage(Base):
    """
    Persists user/AI conversation history for context building.
    """
    __tablename__ = "conversation_messages"
    
    id: str = Column(String(36), primary_key=True)
    role: str = Column(String(20), nullable=False)  # user, assistant, system
    content: str = Column(Text, nullable=False)
    # Renamed msg_metadata to avoid conflict with Base.metadata
    msg_metadata: dict = Column('metadata', JSON, default={})
    created_at: datetime = Column(DateTime, server_default=func.now())

class Log(Base):
    """
    Detailed operational logs for auditing and debugging the autonomous system.
    """
    __tablename__ = "logs"
    
    id: str = Column(String(36), primary_key=True)
    business_id: str = Column(String(36), ForeignKey("business_units.id"), nullable=True)
    
    level: str = Column(String(10), nullable=False)  # info, warning, error
    module: str = Column(String(50), nullable=False) # Which part of the system generated this
    message: str = Column(Text, nullable=False)
    log_data: dict = Column('data', JSON, default={}) # Raw data context for the log event
    
    created_at: datetime = Column(DateTime, server_default=func.now())
    
    business = relationship("BusinessUnit", back_populates="logs")
