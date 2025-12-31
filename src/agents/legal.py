"""
Legal Agent - Document generation, compliance, contracts.
"""
import uuid
from dataclasses import asdict
from datetime import datetime, date, timedelta
from typing import Any, Optional

from src.agents.base import SubAgent
from src.legal.templates import (
    DocumentType, ComplianceFramework, LegalDocument, Contract, Party,
    ComplianceCheck, TEMPLATES, VALID_CONTRACT_STATUSES
)
from src.utils.metrics import TASKS_EXECUTED


class LegalAgent(SubAgent):
    """Agent for legal document generation and compliance."""

    name = "legal"
    description = "Ensures business compliance and analyzes legal documents."

    def __init__(self):
        super().__init__()
        self._documents: dict[str, LegalDocument] = {}
        self._contracts: dict[str, Contract] = {}

    async def generate_document(
        self,
        document_type: DocumentType,
        variables: dict,
        custom_sections: dict = None,
    ) -> dict:
        """Generate a legal document from template."""
        try:
            template = TEMPLATES.get(document_type)
            if not template:
                return {"success": False, "error": f"No template for {document_type}"}

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
                return {"success": False, "error": f"Missing variable: {e}"}

            # Parse and validate effective_date
            try:
                effective_date = date.fromisoformat(variables["effective_date"])
            except (ValueError, KeyError) as e:
                return {"success": False, "error": f"Invalid effective_date format. Use YYYY-MM-DD: {e}"}

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
                effective_date=effective_date,
                variables=variables,
            )
            self._documents[doc.id] = doc

            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": {
                    "document_id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "version": doc.version,
                }
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

    async def generate_custom_document(
        self,
        document_type: str,
        description: str,
        key_terms: list[str],
        parties: list[dict] = None,
    ) -> dict:
        """Generate a custom legal document using AI."""
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

            content = await self._ask_llm(prompt)

            # Try to map document_type string to enum, default to CONTRACTOR_AGREEMENT
            doc_type_enum = DocumentType.CONTRACTOR_AGREEMENT
            try:
                doc_type_enum = DocumentType(document_type.lower().replace(' ', '_'))
            except ValueError:
                # Use default if mapping fails
                pass

            doc = LegalDocument(
                id=str(uuid.uuid4()),
                document_type=doc_type_enum,
                title=document_type,
                content=content,
                version="1.0",
                effective_date=date.today(),
            )
            self._documents[doc.id] = doc

            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": {"document_id": doc.id, "title": doc.title, "content": content}
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

    async def check_compliance(
        self,
        business_id: str,
        frameworks: list[ComplianceFramework],
        website_url: str = None,
        data_practices: dict = None,
    ) -> dict:
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

            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": {
                    "overall_passed": overall_passed,
                    "average_score": round(avg_score, 1),
                    "frameworks": results,
                }
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

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
    ) -> dict:
        """Create a new contract."""
        try:
            variables = {**terms}
            
            # Convert party dicts to Party objects
            party_objects = []
            for i, party_dict in enumerate(parties):
                party = Party(
                    name=party_dict.get("name", ""),
                    address=party_dict.get("address", ""),
                    role=party_dict.get("role"),
                    email=party_dict.get("email"),
                    phone=party_dict.get("phone")
                )
                party_objects.append(party)
                # Also set variables for template
                variables[f"party_{i+1}_name"] = party.name
                variables[f"party_{i+1}_address"] = party.address

            # Generate content
            doc_result = await self.generate_document(document_type, variables)
            if not doc_result["success"]:
                return doc_result

            # Parse dates with error handling
            try:
                effective_date = date.fromisoformat(terms.get("effective_date", date.today().isoformat()))
            except ValueError as e:
                return {"success": False, "error": f"Invalid effective_date format: {e}"}
            
            expiration_date = None
            if terms.get("expiration_date"):
                try:
                    expiration_date = date.fromisoformat(terms["expiration_date"])
                except ValueError as e:
                    return {"success": False, "error": f"Invalid expiration_date format: {e}"}

            contract = Contract(
                id=str(uuid.uuid4()),
                title=title,
                parties=party_objects,
                document_type=document_type,
                content=doc_result["output"]["content"],
                effective_date=effective_date,
                expiration_date=expiration_date,
            )
            self._contracts[contract.id] = contract

            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": {
                    "contract_id": contract.id,
                    "title": contract.title,
                    "status": contract.status,
                    "parties": [{"name": p.name, "address": p.address, "role": p.role} for p in contract.parties],
                }
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

    async def get_contract(self, contract_id: str) -> dict:
        """Get contract details."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return {"success": False, "error": "Contract not found"}

        return {
            "success": True,
            "output": {
                "id": contract.id,
                "title": contract.title,
                "status": contract.status,
                "parties": [{"name": p.name, "address": p.address, "role": p.role} for p in contract.parties],
                "content": contract.content,
                "effective_date": contract.effective_date.isoformat() if contract.effective_date else None,
                "expiration_date": contract.expiration_date.isoformat() if contract.expiration_date else None,
            }
        }

    async def get_document(self, document_id: str) -> dict:
        """Get document details."""
        doc = self._documents.get(document_id)
        if not doc:
            return {"success": False, "error": "Document not found"}
        
        return {
            "success": True,
            "output": {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "version": doc.version,
                "effective_date": doc.effective_date.isoformat(),
            }
        }

    async def update_contract_status(
        self, contract_id: str, status: str
    ) -> dict:
        """Update contract status."""
        contract = self._contracts.get(contract_id)
        if not contract:
            return {"success": False, "error": "Contract not found"}

        if status not in VALID_CONTRACT_STATUSES:
            return {"success": False, "error": f"Invalid status. Use: {VALID_CONTRACT_STATUSES}"}

        contract.status = status
        if status == "active":
            contract.signed_at = datetime.utcnow()

        TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
        return {"success": True, "output": f"Contract status updated to {status}"}

    async def get_expiring_contracts(self, days: int = 30) -> dict:
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

        return {"success": True, "output": {"contracts": expiring, "count": len(expiring)}}

    async def assess_risk(
        self,
        business_type: str,
        operations: list[str],
        jurisdictions: list[str],
    ) -> dict:
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

            # Determine overall risk level
            risk_levels = [r["level"] for r in risks]
            if "high" in risk_levels:
                overall_level = "high"
            elif "medium" in risk_levels:
                overall_level = "medium"
            else:
                overall_level = "low"

            TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
            return {
                "success": True,
                "output": {
                    "overall_risk_level": overall_level,
                    "risks": risks,
                    "recommendations": self._get_risk_recommendations(risks),
                }
            }
        except Exception as e:
            TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
            return {"success": False, "error": str(e)}

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

    async def execute(self, task: dict) -> dict:
        """Execute legal agent task based on action type."""
        action = task.get("action", "")
        context = task.get("input", {})

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
        else:
            # Fallback to legacy behavior for backward compatibility
            description = task.get("description", "Legal check")
            input_data = task.get("input_data", {})

            # Heuristic: route legacy "compliance" tasks to deterministic checks
            # so tests and offline execution don't require an LLM.
            desc_lower = (description or "").lower()
            if "gdpr" in desc_lower:
                data_practices = {
                    "has_privacy_policy": bool(input_data.get("privacy_policy")) or bool(input_data.get("has_privacy_policy")),
                    "has_cookie_consent": bool(input_data.get("cookie_consent")) or bool(input_data.get("has_cookie_consent")),
                    "data_deletion_process": bool(input_data.get("data_deletion_process")),
                    "dpo_appointed": bool(input_data.get("dpo_appointed")),
                }
                return await self.check_compliance(
                    business_id=input_data.get("business_id", ""),
                    frameworks=[ComplianceFramework.GDPR],
                    website_url=input_data.get("website_url"),
                    data_practices=data_practices,
                )

            if "ccpa" in desc_lower:
                data_practices = {
                    "has_privacy_policy": bool(input_data.get("privacy_policy")) or bool(input_data.get("has_privacy_policy")),
                    "opt_out_mechanism": bool(input_data.get("opt_out_mechanism")),
                    "data_inventory": bool(input_data.get("data_inventory")),
                }
                return await self.check_compliance(
                    business_id=input_data.get("business_id", ""),
                    frameworks=[ComplianceFramework.CCPA],
                    website_url=input_data.get("website_url"),
                    data_practices=data_practices,
                )
            
            prompt = f"""
            ### TASK: LEGAL & COMPLIANCE
            {description}
            
            ### DATA / DOCUMENT:
            {input_data}
            
            ### INSTRUCTION:
            Evaluate for risk, compliance issues, and legal obligations.
            Provide a concise summary of concerns and recommended mitigations.
            """
            
            try:
                result = await self._ask_llm(prompt)
                TASKS_EXECUTED.labels(agent=self.name, status="success").inc()
                return {
                    "success": True, 
                    "output": result, 
                    "metadata": {"type": "legal_check"}
                }
            except Exception as e:
                TASKS_EXECUTED.labels(agent=self.name, status="failed").inc()
                return {"success": False, "error": str(e)}
