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
        assert result["success"]
        assert "document_id" in result["output"]
        assert "Test Corp" in result["output"]["content"]

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
        assert result["success"]

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
        assert result["success"]
        assert "frameworks" in result["output"]
        gdpr = result["output"]["frameworks"][0]
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
        assert result["success"]
        assert "contract_id" in result["output"]

    @pytest.mark.asyncio
    async def test_assess_risk(self, legal_agent):
        result = await legal_agent.assess_risk(
            business_type="e-commerce",
            operations=["online_sales", "data_collection"],
            jurisdictions=["US", "EU"],
        )
        assert result["success"]
        assert "risks" in result["output"]
        assert result["output"]["overall_risk_level"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_missing_template_variable(self, legal_agent):
        result = await legal_agent.generate_document(
            DocumentType.PRIVACY_POLICY,
            {}  # Missing required variables
        )
        assert not result["success"]
        assert "Missing variable" in result["error"]

    @pytest.mark.asyncio
    async def test_get_contract(self, legal_agent):
        # Create a contract first
        create_result = await legal_agent.create_contract(
            title="Test Contract",
            document_type=DocumentType.NDA,
            parties=[
                {"name": "Party A", "address": "123 A St"},
                {"name": "Party B", "address": "456 B St"},
            ],
            terms={
                "effective_date": date.today().isoformat(),
                "term_years": "1",
                "disclosing_party": "Party A",
                "receiving_party": "Party B",
            }
        )
        assert create_result["success"]
        contract_id = create_result["output"]["contract_id"]

        # Get the contract
        get_result = await legal_agent.get_contract(contract_id)
        assert get_result["success"]
        assert get_result["output"]["id"] == contract_id

    @pytest.mark.asyncio
    async def test_update_contract_status(self, legal_agent):
        # Create a contract first
        create_result = await legal_agent.create_contract(
            title="Test Contract",
            document_type=DocumentType.NDA,
            parties=[
                {"name": "Party A", "address": "123 A St"},
                {"name": "Party B", "address": "456 B St"},
            ],
            terms={
                "effective_date": date.today().isoformat(),
                "term_years": "1",
                "disclosing_party": "Party A",
                "receiving_party": "Party B",
            }
        )
        contract_id = create_result["output"]["contract_id"]

        # Update status
        update_result = await legal_agent.update_contract_status(contract_id, "active")
        assert update_result["success"]

    @pytest.mark.asyncio
    async def test_get_expiring_contracts(self, legal_agent):
        result = await legal_agent.get_expiring_contracts(days=30)
        assert result["success"]
        assert "contracts" in result["output"]
        assert "count" in result["output"]

    @pytest.mark.asyncio
    async def test_compliance_multiple_frameworks(self, legal_agent):
        result = await legal_agent.check_compliance(
            "biz_1",
            [ComplianceFramework.GDPR, ComplianceFramework.CCPA],
            data_practices={
                "has_privacy_policy": True,
                "has_cookie_consent": True,
                "opt_out_mechanism": False,
            }
        )
        assert result["success"]
        assert len(result["output"]["frameworks"]) == 2

    @pytest.mark.asyncio
    async def test_compliance_wcag(self, legal_agent):
        result = await legal_agent.check_compliance(
            "biz_1",
            [ComplianceFramework.WCAG],
            data_practices={
                "alt_text_images": False,
                "keyboard_navigation": True,
                "color_contrast": True,
            }
        )
        assert result["success"]
        wcag = result["output"]["frameworks"][0]
        assert wcag["framework"] == "wcag"
        assert len(wcag["issues"]) > 0

    @pytest.mark.asyncio
    async def test_legacy_execute_format(self, legal_agent):
        """Test backward compatibility with legacy execute format."""
        task = {
            "description": "Check for GDPR compliance",
            "input_data": {"privacy_policy": "exists"}
        }
        result = await legal_agent.execute(task)
        assert result["success"]
        assert "output" in result
