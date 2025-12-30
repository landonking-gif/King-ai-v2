# Implementation Plan Part 18: Legal Sub-Agent

| Field | Value |
|-------|-------|
| Module | Legal Document & Compliance Sub-Agent |
| Priority | Medium |
| Estimated Effort | 4-5 hours |
| Dependencies | Part 3 (Database), Part 11 (Code Generator) |

---

## 1. Scope

This module implements legal document generation and compliance monitoring:

- **Document Templates** - Privacy policies, terms of service, contracts
- **Document Generator** - AI-powered legal document creation
- **Compliance Checker** - GDPR, CCPA, accessibility compliance
- **Contract Management** - Track contracts, renewals, signatures
- **Risk Assessment** - Legal risk analysis for business operations

---

## 2. Tasks

### Task 18.1: Legal Templates

**File: `src/legal/templates.py`**

```python
"""
Legal Document Templates.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Optional


class DocumentType(Enum):
    PRIVACY_POLICY = "privacy_policy"
    TERMS_OF_SERVICE = "terms_of_service"
    COOKIE_POLICY = "cookie_policy"
    REFUND_POLICY = "refund_policy"
    EULA = "eula"
    NDA = "nda"
    CONTRACTOR_AGREEMENT = "contractor_agreement"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    SLA = "sla"


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOC2 = "soc2"
    WCAG = "wcag"


@dataclass
class LegalDocument:
    id: str
    document_type: DocumentType
    title: str
    content: str
    version: str
    effective_date: date
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)


@dataclass
class Contract:
    id: str
    title: str
    parties: list[dict]
    document_type: DocumentType
    content: str
    status: str = "draft"  # draft, pending_signature, active, expired, terminated
    effective_date: Optional[date] = None
    expiration_date: Optional[date] = None
    renewal_terms: Optional[str] = None
    signed_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ComplianceCheck:
    framework: ComplianceFramework
    passed: bool
    score: float  # 0-100
    issues: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)


# Template strings for common documents
TEMPLATES = {
    DocumentType.PRIVACY_POLICY: """
# Privacy Policy

**Effective Date:** {effective_date}

## Introduction

{company_name} ("we," "our," or "us") respects your privacy and is committed to protecting your personal data. This privacy policy explains how we collect, use, and safeguard your information when you visit {website_url}.

## Information We Collect

### Information You Provide
- Contact information (name, email, phone)
- Account credentials
- Payment information
- Communications with us

### Automatically Collected Information
- IP address and device information
- Browser type and settings
- Usage data and analytics
- Cookies and similar technologies

## How We Use Your Information

We use collected information to:
- Provide and maintain our services
- Process transactions
- Send communications
- Improve our services
- Comply with legal obligations

## Data Sharing

We may share your information with:
- Service providers
- Legal authorities when required
- Business partners with your consent

## Your Rights

{rights_section}

## Data Retention

We retain your data for {retention_period} or as required by law.

## Contact Us

For privacy inquiries: {contact_email}

{company_name}
{company_address}
""",

    DocumentType.TERMS_OF_SERVICE: """
# Terms of Service

**Last Updated:** {effective_date}

## Agreement to Terms

By accessing {website_url}, you agree to these Terms of Service and our Privacy Policy.

## Use of Services

### Eligibility
You must be at least {min_age} years old to use our services.

### Account Responsibilities
- Maintain accurate account information
- Keep credentials confidential
- Notify us of unauthorized access

### Prohibited Uses
You may not:
- Violate any laws or regulations
- Infringe intellectual property rights
- Transmit harmful code or content
- Interfere with service operation

## Intellectual Property

All content and materials are owned by {company_name} or its licensors.

## Payment Terms

{payment_terms}

## Limitation of Liability

{liability_section}

## Termination

We may terminate access for violations of these terms.

## Governing Law

These terms are governed by the laws of {jurisdiction}.

## Contact

{company_name}
{contact_email}
""",

    DocumentType.REFUND_POLICY: """
# Refund Policy

**Effective Date:** {effective_date}

## Refund Eligibility

{refund_eligibility}

## Refund Process

1. Contact us at {contact_email}
2. Provide order details
3. Allow {processing_days} business days for processing

## Non-Refundable Items

{non_refundable_items}

## Contact

{company_name}
{contact_email}
""",

    DocumentType.NDA: """
# Non-Disclosure Agreement

This Non-Disclosure Agreement ("Agreement") is entered into as of {effective_date}.

## Parties

**Disclosing Party:** {disclosing_party}
**Receiving Party:** {receiving_party}

## Confidential Information

"Confidential Information" includes all non-public information disclosed by either party.

## Obligations

The Receiving Party agrees to:
- Maintain confidentiality
- Use information only for permitted purposes
- Not disclose to third parties without consent

## Term

This Agreement remains in effect for {term_years} years from the effective date.

## Exceptions

Obligations do not apply to information that:
- Is publicly available
- Was known prior to disclosure
- Is independently developed
- Is required by law to disclose

## Signatures

_________________________
{disclosing_party}
Date: _______________

_________________________
{receiving_party}
Date: _______________
""",
}
```

---

### Task 18.2: Legal Agent

**File: `src/agents/legal.py`**

```python
"""
Legal Agent - Document generation, compliance, contracts.
"""
import uuid
from dataclasses import asdict
from datetime import datetime, date, timedelta
from typing import Any, Optional
from src.agents.base import BaseAgent, AgentCapability, AgentResult
from src.legal.templates import (
    DocumentType, ComplianceFramework, LegalDocument, Contract,
    ComplianceCheck, TEMPLATES
)
from src.utils.ollama_client import OllamaClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LegalAgent(BaseAgent):
    """Agent for legal document generation and compliance."""

    def __init__(self, ollama_client: OllamaClient = None):
        super().__init__(
            name="Legal Agent",
            capabilities=[AgentCapability.CONTENT, AgentCapability.ANALYSIS]
        )
        self.llm = ollama_client
        self._documents: dict[str, LegalDocument] = {}
        self._contracts: dict[str, Contract] = {}

    async def generate_document(
        self,
        document_type: DocumentType,
        variables: dict,
        custom_sections: dict = None,
    ) -> AgentResult:
        """Generate a legal document from template."""
        try:
            template = TEMPLATES.get(document_type)
            if not template:
                return AgentResult(success=False, message=f"No template for {document_type}")

            # Set defaults
            variables.setdefault("effective_date", date.today().isoformat())
            variables.setdefault("min_age", "18")
            variables.setdefault("retention_period", "3 years")
            variables.setdefault("processing_days", "5-7")

            # Add compliance-specific sections
            if variables.get("gdpr_compliant"):
                variables["rights_section"] = self._gdpr_rights_section()
            else:
                variables["rights_section"] = self._standard_rights_section()

            # Format template
            try:
                content = template.format(**variables)
            except KeyError as e:
                return AgentResult(success=False, message=f"Missing variable: {e}")

            # Add custom sections
            if custom_sections:
                for section_name, section_content in custom_sections.items():
                    content += f"\n\n## {section_name}\n\n{section_content}"

            doc = LegalDocument(
                id=str(uuid.uuid4()),
                document_type=document_type,
                title=f"{document_type.value.replace('_', ' ').title()}",
                content=content,
                version="1.0",
                effective_date=date.fromisoformat(variables["effective_date"]),
                variables=variables,
            )
            self._documents[doc.id] = doc

            return AgentResult(
                success=True,
                message="Document generated",
                data={
                    "document_id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "version": doc.version,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def generate_custom_document(
        self,
        document_type: str,
        description: str,
        key_terms: list[str],
        parties: list[dict] = None,
    ) -> AgentResult:
        """Generate a custom legal document using AI."""
        if not self.llm:
            return AgentResult(success=False, message="LLM not configured")

        try:
            prompt = f"""Generate a professional legal document with the following specifications:

Document Type: {document_type}
Description: {description}
Key Terms to Include: {', '.join(key_terms)}
Parties: {parties if parties else 'General/Standard'}

Requirements:
- Use clear, professional legal language
- Include standard legal clauses (definitions, term, termination, governing law)
- Add appropriate sections for the document type
- Include signature blocks if applicable

Generate the complete document in markdown format."""

            content = await self.llm.generate(prompt)

            doc = LegalDocument(
                id=str(uuid.uuid4()),
                document_type=DocumentType.CONTRACTOR_AGREEMENT,  # Default
                title=document_type,
                content=content,
                version="1.0",
                effective_date=date.today(),
            )
            self._documents[doc.id] = doc

            return AgentResult(
                success=True,
                data={"document_id": doc.id, "title": doc.title, "content": content}
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def check_compliance(
        self,
        business_id: str,
        frameworks: list[ComplianceFramework],
        website_url: str = None,
        data_practices: dict = None,
    ) -> AgentResult:
        """Check compliance against specified frameworks."""
        try:
            results = []
            
            for framework in frameworks:
                check = await self._check_framework(framework, website_url, data_practices)
                results.append({
                    "framework": framework.value,
                    "passed": check.passed,
                    "score": check.score,
                    "issues": check.issues,
                    "recommendations": check.recommendations,
                })

            overall_passed = all(r["passed"] for r in results)
            avg_score = sum(r["score"] for r in results) / len(results) if results else 0

            return AgentResult(
                success=True,
                data={
                    "overall_passed": overall_passed,
                    "average_score": round(avg_score, 1),
                    "frameworks": results,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def _check_framework(
        self,
        framework: ComplianceFramework,
        website_url: str = None,
        data_practices: dict = None,
    ) -> ComplianceCheck:
        """Check compliance for a specific framework."""
        issues = []
        recommendations = []
        score = 100.0

        data_practices = data_practices or {}

        if framework == ComplianceFramework.GDPR:
            # GDPR checks
            if not data_practices.get("has_privacy_policy"):
                issues.append({"severity": "high", "issue": "Missing privacy policy"})
                score -= 20
            if not data_practices.get("has_cookie_consent"):
                issues.append({"severity": "high", "issue": "No cookie consent mechanism"})
                score -= 15
            if not data_practices.get("data_deletion_process"):
                issues.append({"severity": "medium", "issue": "No data deletion process"})
                score -= 10
                recommendations.append("Implement user data deletion request handling")
            if not data_practices.get("dpo_appointed"):
                recommendations.append("Consider appointing a Data Protection Officer")

        elif framework == ComplianceFramework.CCPA:
            # CCPA checks
            if not data_practices.get("has_privacy_policy"):
                issues.append({"severity": "high", "issue": "Missing privacy policy"})
                score -= 20
            if not data_practices.get("opt_out_mechanism"):
                issues.append({"severity": "high", "issue": "No opt-out mechanism for data sale"})
                score -= 15
            if not data_practices.get("data_inventory"):
                recommendations.append("Create comprehensive data inventory")

        elif framework == ComplianceFramework.WCAG:
            # Accessibility checks
            if not data_practices.get("alt_text_images"):
                issues.append({"severity": "medium", "issue": "Missing alt text on images"})
                score -= 10
            if not data_practices.get("keyboard_navigation"):
                issues.append({"severity": "high", "issue": "Keyboard navigation issues"})
                score -= 15
            if not data_practices.get("color_contrast"):
                issues.append({"severity": "medium", "issue": "Color contrast issues"})
                score -= 10

        elif framework == ComplianceFramework.PCI_DSS:
            # PCI DSS checks
            if not data_practices.get("encrypted_storage"):
                issues.append({"severity": "critical", "issue": "Card data not encrypted"})
                score -= 30
            if not data_practices.get("secure_transmission"):
                issues.append({"severity": "critical", "issue": "Insecure data transmission"})
                score -= 25

        return ComplianceCheck(
            framework=framework,
            passed=score >= 70,
            score=max(0, score),
            issues=issues,
            recommendations=recommendations,
        )

    async def create_contract(
        self,
        title: str,
        document_type: DocumentType,
        parties: list[dict],
        terms: dict,
    ) -> AgentResult:
        """Create a new contract."""
        try:
            variables = {**terms}
            for i, party in enumerate(parties):
                variables[f"party_{i+1}_name"] = party.get("name", "")
                variables[f"party_{i+1}_address"] = party.get("address", "")

            # Generate content
            doc_result = await self.generate_document(document_type, variables)
            if not doc_result.success:
                return doc_result

            contract = Contract(
                id=str(uuid.uuid4()),
                title=title,
                parties=parties,
                document_type=document_type,
                content=doc_result.data["content"],
                effective_date=date.fromisoformat(terms.get("effective_date", date.today().isoformat())),
                expiration_date=date.fromisoformat(terms["expiration_date"]) if terms.get("expiration_date") else None,
            )
            self._contracts[contract.id] = contract

            return AgentResult(
                success=True,
                data={
                    "contract_id": contract.id,
                    "title": contract.title,
                    "status": contract.status,
                    "parties": contract.parties,
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    async def get_contract(self, contract_id: str) -> AgentResult:
        """Get contract details."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return AgentResult(success=False, message="Contract not found")

        return AgentResult(
            success=True,
            data={
                "id": contract.id,
                "title": contract.title,
                "status": contract.status,
                "parties": contract.parties,
                "content": contract.content,
                "effective_date": contract.effective_date.isoformat() if contract.effective_date else None,
                "expiration_date": contract.expiration_date.isoformat() if contract.expiration_date else None,
            }
        )

    async def update_contract_status(
        self, contract_id: str, status: str
    ) -> AgentResult:
        """Update contract status."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return AgentResult(success=False, message="Contract not found")

        valid_statuses = ["draft", "pending_signature", "active", "expired", "terminated"]
        if status not in valid_statuses:
            return AgentResult(success=False, message=f"Invalid status. Use: {valid_statuses}")

        contract.status = status
        if status == "active":
            contract.signed_at = datetime.utcnow()

        return AgentResult(success=True, message=f"Contract status updated to {status}")

    async def get_expiring_contracts(self, days: int = 30) -> AgentResult:
        """Get contracts expiring within specified days."""
        expiring = []
        threshold = date.today() + timedelta(days=days)

        for contract in self._contracts.values():
            if contract.expiration_date and contract.status == "active":
                if contract.expiration_date <= threshold:
                    expiring.append({
                        "id": contract.id,
                        "title": contract.title,
                        "expiration_date": contract.expiration_date.isoformat(),
                        "days_remaining": (contract.expiration_date - date.today()).days,
                    })

        expiring.sort(key=lambda x: x["days_remaining"])

        return AgentResult(success=True, data={"contracts": expiring, "count": len(expiring)})

    async def assess_risk(
        self,
        business_type: str,
        operations: list[str],
        jurisdictions: list[str],
    ) -> AgentResult:
        """Assess legal risks for business operations."""
        try:
            risks = []
            
            # Data handling risks
            if "data_collection" in operations:
                risks.append({
                    "area": "Data Privacy",
                    "level": "medium",
                    "description": "Collecting user data requires privacy policy and consent mechanisms",
                    "mitigations": ["Implement privacy policy", "Add consent forms", "Document data flows"],
                })

            # E-commerce risks
            if "online_sales" in operations:
                risks.append({
                    "area": "Consumer Protection",
                    "level": "medium",
                    "description": "Online sales require clear terms, refund policies",
                    "mitigations": ["Terms of service", "Refund policy", "Consumer disclosures"],
                })

            # International operations
            if len(jurisdictions) > 1 or "EU" in jurisdictions:
                risks.append({
                    "area": "International Compliance",
                    "level": "high",
                    "description": "Multi-jurisdiction operations require varied compliance",
                    "mitigations": ["GDPR compliance for EU", "Local legal counsel", "Data localization review"],
                })

            # IP risks
            if "content_creation" in operations:
                risks.append({
                    "area": "Intellectual Property",
                    "level": "low",
                    "description": "Content creation requires IP protection and licensing",
                    "mitigations": ["Copyright notices", "Content licensing terms", "Trademark registration"],
                })

            overall_level = "high" if any(r["level"] == "high" for r in risks) else "medium" if any(r["level"] == "medium" for r in risks) else "low"

            return AgentResult(
                success=True,
                data={
                    "overall_risk_level": overall_level,
                    "risks": risks,
                    "recommendations": self._get_risk_recommendations(risks),
                }
            )
        except Exception as e:
            return AgentResult(success=False, message=str(e))

    def _gdpr_rights_section(self) -> str:
        return """Under GDPR, you have the right to:
- Access your personal data
- Rectify inaccurate data
- Erase your data ("right to be forgotten")
- Restrict processing
- Data portability
- Object to processing
- Lodge a complaint with a supervisory authority"""

    def _standard_rights_section(self) -> str:
        return """You have the right to:
- Access your personal data
- Request correction of inaccurate data
- Request deletion of your data
- Opt-out of marketing communications"""

    def _get_risk_recommendations(self, risks: list) -> list[str]:
        recommendations = []
        if any(r["area"] == "Data Privacy" for r in risks):
            recommendations.append("Conduct privacy impact assessment")
        if any(r["level"] == "high" for r in risks):
            recommendations.append("Consult with legal counsel")
        recommendations.append("Review and update policies annually")
        return recommendations

    async def execute(self, task: str, context: dict) -> AgentResult:
        action = context.get("action", "")

        if action == "generate":
            return await self.generate_document(
                DocumentType(context["document_type"]),
                context.get("variables", {}),
            )
        elif action == "compliance":
            frameworks = [ComplianceFramework(f) for f in context.get("frameworks", [])]
            return await self.check_compliance(
                context.get("business_id", ""),
                frameworks,
                context.get("website_url"),
                context.get("data_practices"),
            )
        elif action == "create_contract":
            return await self.create_contract(
                context["title"],
                DocumentType(context["document_type"]),
                context["parties"],
                context.get("terms", {}),
            )
        elif action == "risk_assessment":
            return await self.assess_risk(
                context["business_type"],
                context.get("operations", []),
                context.get("jurisdictions", ["US"]),
            )

        return AgentResult(success=False, message=f"Unknown action: {action}")
```

---

### Task 18.3: Legal API Routes

**File: `src/api/routes/legal.py`**

```python
"""
Legal API Routes.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.agents.legal import LegalAgent
from src.legal.templates import DocumentType, ComplianceFramework
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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/documents/custom")
async def generate_custom_document(req: CustomDocRequest, agent: LegalAgent = Depends(get_agent)):
    """Generate a custom legal document using AI."""
    result = await agent.generate_custom_document(
        req.document_type, req.description, req.key_terms, req.parties
    )
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/documents/{document_id}")
async def get_document(document_id: str, agent: LegalAgent = Depends(get_agent)):
    """Get a generated document."""
    doc = agent._documents.get(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "id": doc.id,
        "title": doc.title,
        "content": doc.content,
        "version": doc.version,
        "effective_date": doc.effective_date.isoformat(),
    }


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
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.post("/contracts")
async def create_contract(req: ContractRequest, agent: LegalAgent = Depends(get_agent)):
    """Create a new contract."""
    try:
        doc_type = DocumentType(req.document_type)
    except ValueError:
        raise HTTPException(400, f"Invalid document type: {req.document_type}")
    
    result = await agent.create_contract(req.title, doc_type, req.parties, req.terms)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data


@router.get("/contracts/{contract_id}")
async def get_contract(contract_id: str, agent: LegalAgent = Depends(get_agent)):
    """Get contract details."""
    result = await agent.get_contract(contract_id)
    if not result.success:
        raise HTTPException(404, result.message)
    return result.data


@router.patch("/contracts/{contract_id}/status")
async def update_contract_status(
    contract_id: str, status: str, agent: LegalAgent = Depends(get_agent)
):
    """Update contract status."""
    result = await agent.update_contract_status(contract_id, status)
    if not result.success:
        raise HTTPException(400, result.message)
    return {"status": "updated"}


@router.get("/contracts/expiring")
async def get_expiring_contracts(days: int = 30, agent: LegalAgent = Depends(get_agent)):
    """Get contracts expiring soon."""
    result = await agent.get_expiring_contracts(days)
    return result.data


@router.post("/risk/assess")
async def assess_risk(req: RiskRequest, agent: LegalAgent = Depends(get_agent)):
    """Assess legal risks."""
    result = await agent.assess_risk(req.business_type, req.operations, req.jurisdictions)
    if not result.success:
        raise HTTPException(400, result.message)
    return result.data
```

---

### Task 18.4: Tests

**File: `tests/test_legal.py`**

```python
"""Tests for Legal Agent."""
import pytest
from datetime import date
from src.legal.templates import DocumentType, ComplianceFramework, LegalDocument, Contract
from src.agents.legal import LegalAgent


@pytest.fixture
def legal_agent():
    return LegalAgent()


class TestLegalDocument:
    def test_create_document(self):
        doc = LegalDocument(
            id="doc_1",
            document_type=DocumentType.PRIVACY_POLICY,
            title="Privacy Policy",
            content="Test content",
            version="1.0",
            effective_date=date.today(),
        )
        assert doc.id == "doc_1"
        assert doc.document_type == DocumentType.PRIVACY_POLICY


class TestLegalAgent:
    @pytest.mark.asyncio
    async def test_generate_privacy_policy(self, legal_agent):
        result = await legal_agent.generate_document(
            DocumentType.PRIVACY_POLICY,
            {
                "company_name": "Test Corp",
                "website_url": "https://test.com",
                "contact_email": "legal@test.com",
                "company_address": "123 Test St",
            }
        )
        assert result.success
        assert "document_id" in result.data
        assert "Test Corp" in result.data["content"]

    @pytest.mark.asyncio
    async def test_generate_terms_of_service(self, legal_agent):
        result = await legal_agent.generate_document(
            DocumentType.TERMS_OF_SERVICE,
            {
                "company_name": "Test Corp",
                "website_url": "https://test.com",
                "contact_email": "legal@test.com",
                "jurisdiction": "Delaware, USA",
                "payment_terms": "Payment due upon receipt",
                "liability_section": "Limited to fees paid",
            }
        )
        assert result.success

    @pytest.mark.asyncio
    async def test_check_gdpr_compliance(self, legal_agent):
        result = await legal_agent.check_compliance(
            "biz_1",
            [ComplianceFramework.GDPR],
            data_practices={
                "has_privacy_policy": True,
                "has_cookie_consent": True,
                "data_deletion_process": False,
            }
        )
        assert result.success
        assert "frameworks" in result.data
        gdpr = result.data["frameworks"][0]
        assert gdpr["framework"] == "gdpr"

    @pytest.mark.asyncio
    async def test_create_contract(self, legal_agent):
        result = await legal_agent.create_contract(
            title="Service Agreement",
            document_type=DocumentType.NDA,
            parties=[
                {"name": "Company A", "address": "123 A St"},
                {"name": "Company B", "address": "456 B St"},
            ],
            terms={
                "effective_date": date.today().isoformat(),
                "term_years": "2",
                "disclosing_party": "Company A",
                "receiving_party": "Company B",
            }
        )
        assert result.success
        assert "contract_id" in result.data

    @pytest.mark.asyncio
    async def test_assess_risk(self, legal_agent):
        result = await legal_agent.assess_risk(
            business_type="e-commerce",
            operations=["online_sales", "data_collection"],
            jurisdictions=["US", "EU"],
        )
        assert result.success
        assert "risks" in result.data
        assert result.data["overall_risk_level"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_missing_template_variable(self, legal_agent):
        result = await legal_agent.generate_document(
            DocumentType.PRIVACY_POLICY,
            {}  # Missing required variables
        )
        assert not result.success
        assert "Missing variable" in result.message
```

---

## 3. Acceptance Criteria

| Criteria | Validation |
|----------|------------|
| Documents generated | Privacy policy, ToS, contracts created |
| Templates work | Variables substituted correctly |
| Compliance checks | GDPR, CCPA, WCAG evaluated |
| Contracts managed | Create, update status, track expiration |
| Risk assessed | Business operations analyzed |
| API functional | All endpoints operational |

---

## 4. File Summary

| File | Purpose |
|------|---------|
| `src/legal/templates.py` | Document templates and models |
| `src/agents/legal.py` | Legal agent implementation |
| `src/api/routes/legal.py` | REST API endpoints |
| `tests/test_legal.py` | Unit tests |
