"""
Legal API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator

from src.agents.legal import LegalAgent
from src.legal.templates import DocumentType, ComplianceFramework, VALID_CONTRACT_STATUSES
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/legal", tags=["legal"])

_agent: Optional[LegalAgent] = None


def get_agent() -> LegalAgent:
    global _agent
    if _agent is None:
        _agent = LegalAgent()
    return _agent


class GenerateDocRequest(BaseModel):
    document_type: str
    variables: dict
    custom_sections: Optional[dict] = None


class CustomDocRequest(BaseModel):
    document_type: str
    description: str
    key_terms: list[str]
    parties: Optional[list[dict]] = None


class ComplianceRequest(BaseModel):
    business_id: str
    frameworks: list[str]
    website_url: Optional[str] = None
    data_practices: Optional[dict] = None


class ContractRequest(BaseModel):
    title: str
    document_type: str
    parties: list[dict]
    terms: dict = Field(default_factory=dict)


class ContractStatusUpdate(BaseModel):
    status: str
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_CONTRACT_STATUSES:
            raise ValueError(f"Invalid status. Must be one of: {VALID_CONTRACT_STATUSES}")
        return v


class RiskRequest(BaseModel):
    business_type: str
    operations: list[str]
    jurisdictions: list[str] = ["US"]


@router.post("/documents/generate")
async def generate_document(req: GenerateDocRequest, agent: LegalAgent = Depends(get_agent)):
    """Generate a legal document from template."""
    try:
        doc_type = DocumentType(req.document_type)
    except ValueError:
        raise HTTPException(400, f"Invalid document type: {req.document_type}")
    
    result = await agent.generate_document(doc_type, req.variables, req.custom_sections)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result["output"]


@router.post("/documents/custom")
async def generate_custom_document(req: CustomDocRequest, agent: LegalAgent = Depends(get_agent)):
    """Generate a custom legal document using AI."""
    result = await agent.generate_custom_document(
        req.document_type, req.description, req.key_terms, req.parties
    )
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result["output"]


@router.get("/documents/{document_id}")
async def get_document(document_id: str, agent: LegalAgent = Depends(get_agent)):
    """Get a generated document."""
    result = await agent.get_document(document_id)
    if not result["success"]:
        raise HTTPException(404, result["error"])
    return result["output"]


@router.post("/compliance/check")
async def check_compliance(req: ComplianceRequest, agent: LegalAgent = Depends(get_agent)):
    """Check compliance against frameworks."""
    try:
        frameworks = [ComplianceFramework(f) for f in req.frameworks]
    except ValueError as e:
        raise HTTPException(400, f"Invalid framework: {e}")
    
    result = await agent.check_compliance(
        req.business_id, frameworks, req.website_url, req.data_practices
    )
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result["output"]


@router.post("/contracts")
async def create_contract(req: ContractRequest, agent: LegalAgent = Depends(get_agent)):
    """Create a new contract."""
    try:
        doc_type = DocumentType(req.document_type)
    except ValueError:
        raise HTTPException(400, f"Invalid document type: {req.document_type}")
    
    result = await agent.create_contract(req.title, doc_type, req.parties, req.terms)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result["output"]


@router.get("/contracts/{contract_id}")
async def get_contract(contract_id: str, agent: LegalAgent = Depends(get_agent)):
    """Get contract details."""
    result = await agent.get_contract(contract_id)
    if not result["success"]:
        raise HTTPException(404, result["error"])
    return result["output"]


@router.patch("/contracts/{contract_id}/status")
async def update_contract_status(
    contract_id: str, status_update: ContractStatusUpdate, agent: LegalAgent = Depends(get_agent)
):
    """Update contract status."""
    result = await agent.update_contract_status(contract_id, status_update.status)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return {"status": "updated"}


@router.get("/contracts/expiring")
async def get_expiring_contracts(days: int = 30, agent: LegalAgent = Depends(get_agent)):
    """Get contracts expiring soon."""
    result = await agent.get_expiring_contracts(days)
    return result["output"]


@router.post("/risk/assess")
async def assess_risk(req: RiskRequest, agent: LegalAgent = Depends(get_agent)):
    """Assess legal risks."""
    result = await agent.assess_risk(req.business_type, req.operations, req.jurisdictions)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result["output"]
