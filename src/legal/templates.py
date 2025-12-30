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
