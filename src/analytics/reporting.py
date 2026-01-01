"""
Automated Reporting.
Scheduled report generation and distribution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
import uuid
import json
import asyncio

from src.utils.structured_logging import get_logger

logger = get_logger("reporting")


class ReportFormat(str, Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    PDF = "pdf"


class ReportFrequency(str, Enum):
    """Report generation frequency."""
    REALTIME = "realtime"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class DeliveryMethod(str, Enum):
    """Report delivery methods."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    STORAGE = "storage"
    API = "api"


class ReportStatus(str, Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    DELIVERED = "delivered"


@dataclass
class ReportSection:
    """A section of a report."""
    id: str
    title: str
    content: Any
    section_type: str = "data"  # data, chart, table, text
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportRecipient:
    """A report recipient."""
    id: str
    name: str
    email: Optional[str] = None
    slack_channel: Optional[str] = None
    webhook_url: Optional[str] = None
    delivery_method: DeliveryMethod = DeliveryMethod.EMAIL


@dataclass
class ReportTemplate:
    """Template for generating reports."""
    id: str
    name: str
    description: str = ""
    sections: List[str] = field(default_factory=list)  # Section IDs
    format: ReportFormat = ReportFormat.JSON
    data_sources: Dict[str, Callable] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScheduledReport:
    """A scheduled report configuration."""
    id: str
    name: str
    template_id: str
    frequency: ReportFrequency
    recipients: List[ReportRecipient] = field(default_factory=list)
    
    # Schedule
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    
    # Configuration
    enabled: bool = True
    time_of_day: str = "09:00"  # HH:MM
    day_of_week: int = 0  # 0 = Monday
    day_of_month: int = 1
    
    # Filters and parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneratedReport:
    """A generated report instance."""
    id: str
    template_id: str
    schedule_id: Optional[str]
    name: str
    status: ReportStatus = ReportStatus.PENDING
    format: ReportFormat = ReportFormat.JSON
    
    # Content
    sections: List[ReportSection] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    rendered_content: Optional[str] = None
    
    # Metadata
    generated_at: Optional[datetime] = None
    generation_time_ms: int = 0
    file_size_bytes: int = 0
    
    # Delivery
    delivered_to: List[str] = field(default_factory=list)
    delivery_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "format": self.format.value,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "generation_time_ms": self.generation_time_ms,
            "section_count": len(self.sections),
        }


class ReportBuilder:
    """Build report content."""
    
    def __init__(self, report: GeneratedReport):
        self.report = report
        self.sections: List[ReportSection] = []
    
    def add_section(
        self,
        title: str,
        content: Any,
        section_type: str = "data",
        metadata: Dict[str, Any] = None,
    ) -> "ReportBuilder":
        """Add a section to the report."""
        section = ReportSection(
            id=str(uuid.uuid4())[:8],
            title=title,
            content=content,
            section_type=section_type,
            order=len(self.sections),
            metadata=metadata or {},
        )
        self.sections.append(section)
        return self
    
    def add_text(self, title: str, text: str) -> "ReportBuilder":
        """Add a text section."""
        return self.add_section(title, text, "text")
    
    def add_table(
        self,
        title: str,
        headers: List[str],
        rows: List[List[Any]],
    ) -> "ReportBuilder":
        """Add a table section."""
        return self.add_section(
            title,
            {"headers": headers, "rows": rows},
            "table",
        )
    
    def add_metric(
        self,
        title: str,
        value: Union[int, float],
        unit: str = "",
        change: Optional[float] = None,
    ) -> "ReportBuilder":
        """Add a metric section."""
        return self.add_section(
            title,
            {"value": value, "unit": unit, "change": change},
            "metric",
        )
    
    def add_chart(
        self,
        title: str,
        chart_type: str,
        data: Dict[str, Any],
    ) -> "ReportBuilder":
        """Add a chart section."""
        return self.add_section(
            title,
            {"chart_type": chart_type, "data": data},
            "chart",
        )
    
    def build(self) -> GeneratedReport:
        """Build the report."""
        self.report.sections = self.sections
        return self.report


class ReportRenderer:
    """Render reports to various formats."""
    
    def render(
        self,
        report: GeneratedReport,
        format: ReportFormat,
    ) -> str:
        """Render report to specified format."""
        if format == ReportFormat.JSON:
            return self._render_json(report)
        elif format == ReportFormat.HTML:
            return self._render_html(report)
        elif format == ReportFormat.MARKDOWN:
            return self._render_markdown(report)
        elif format == ReportFormat.CSV:
            return self._render_csv(report)
        else:
            return self._render_json(report)
    
    def _render_json(self, report: GeneratedReport) -> str:
        """Render as JSON."""
        data = {
            "report_id": report.id,
            "name": report.name,
            "generated_at": report.generated_at.isoformat() if report.generated_at else None,
            "sections": [
                {
                    "title": s.title,
                    "type": s.section_type,
                    "content": s.content,
                }
                for s in report.sections
            ],
            "data": report.data,
        }
        return json.dumps(data, indent=2, default=str)
    
    def _render_html(self, report: GeneratedReport) -> str:
        """Render as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{report.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
        .change-positive {{ color: #28a745; }}
        .change-negative {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        tr:nth-child(even) {{ background-color: #fafafa; }}
    </style>
</head>
<body>
    <h1>{report.name}</h1>
    <p>Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S') if report.generated_at else 'N/A'}</p>
"""
        
        for section in report.sections:
            html += f'    <div class="section">\n'
            html += f'        <h2>{section.title}</h2>\n'
            
            if section.section_type == "text":
                html += f'        <p>{section.content}</p>\n'
            
            elif section.section_type == "metric":
                value = section.content.get("value", 0)
                unit = section.content.get("unit", "")
                change = section.content.get("change")
                html += f'        <p class="metric">{value:,} {unit}</p>\n'
                if change is not None:
                    change_class = "change-positive" if change >= 0 else "change-negative"
                    html += f'        <p class="{change_class}">{change:+.1f}%</p>\n'
            
            elif section.section_type == "table":
                headers = section.content.get("headers", [])
                rows = section.content.get("rows", [])
                html += '        <table>\n'
                html += '            <tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>\n'
                for row in rows:
                    html += '            <tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>\n'
                html += '        </table>\n'
            
            html += '    </div>\n'
        
        html += """</body>
</html>"""
        return html
    
    def _render_markdown(self, report: GeneratedReport) -> str:
        """Render as Markdown."""
        md = f"# {report.name}\n\n"
        md += f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S') if report.generated_at else 'N/A'}*\n\n"
        
        for section in report.sections:
            md += f"## {section.title}\n\n"
            
            if section.section_type == "text":
                md += f"{section.content}\n\n"
            
            elif section.section_type == "metric":
                value = section.content.get("value", 0)
                unit = section.content.get("unit", "")
                change = section.content.get("change")
                md += f"**{value:,}** {unit}\n\n"
                if change is not None:
                    md += f"Change: {change:+.1f}%\n\n"
            
            elif section.section_type == "table":
                headers = section.content.get("headers", [])
                rows = section.content.get("rows", [])
                md += "| " + " | ".join(str(h) for h in headers) + " |\n"
                md += "| " + " | ".join("---" for _ in headers) + " |\n"
                for row in rows:
                    md += "| " + " | ".join(str(c) for c in row) + " |\n"
                md += "\n"
        
        return md
    
    def _render_csv(self, report: GeneratedReport) -> str:
        """Render table sections as CSV."""
        csv_lines = []
        
        for section in report.sections:
            if section.section_type == "table":
                headers = section.content.get("headers", [])
                rows = section.content.get("rows", [])
                csv_lines.append(",".join(f'"{h}"' for h in headers))
                for row in rows:
                    csv_lines.append(",".join(f'"{c}"' for c in row))
                csv_lines.append("")
        
        return "\n".join(csv_lines)


class AutomatedReporting:
    """
    Automated Reporting System.
    
    Features:
    - Report templates
    - Scheduled report generation
    - Multiple output formats
    - Multi-channel delivery
    - Custom data sources
    """
    
    def __init__(self):
        self.templates: Dict[str, ReportTemplate] = {}
        self.schedules: Dict[str, ScheduledReport] = {}
        self.reports: Dict[str, GeneratedReport] = {}
        self.data_sources: Dict[str, Callable] = {}
        self.renderer = ReportRenderer()
        
        self._setup_default_templates()
    
    def _setup_default_templates(self) -> None:
        """Set up default report templates."""
        # Daily summary template
        self.create_template(
            name="Daily Summary",
            description="Daily business summary report",
            sections=["overview", "sales", "traffic", "alerts"],
            format=ReportFormat.HTML,
        )
        
        # Weekly analytics template
        self.create_template(
            name="Weekly Analytics",
            description="Weekly analytics and insights report",
            sections=["performance", "trends", "recommendations"],
            format=ReportFormat.MARKDOWN,
        )
    
    def register_data_source(
        self,
        name: str,
        source: Callable[..., Any],
    ) -> None:
        """
        Register a data source for reports.
        
        Args:
            name: Data source name
            source: Callable that returns data
        """
        self.data_sources[name] = source
        logger.info(f"Registered data source: {name}")
    
    def create_template(
        self,
        name: str,
        description: str = "",
        sections: List[str] = None,
        format: ReportFormat = ReportFormat.JSON,
        filters: Dict[str, Any] = None,
    ) -> ReportTemplate:
        """
        Create a report template.
        
        Args:
            name: Template name
            description: Template description
            sections: Section IDs to include
            format: Output format
            filters: Default filters
            
        Returns:
            Created template
        """
        template_id = str(uuid.uuid4())[:8]
        
        template = ReportTemplate(
            id=template_id,
            name=name,
            description=description,
            sections=sections or [],
            format=format,
            filters=filters or {},
        )
        
        self.templates[template_id] = template
        logger.info(f"Created report template: {name}")
        
        return template
    
    def schedule_report(
        self,
        template_id: str,
        name: str,
        frequency: ReportFrequency,
        recipients: List[Dict[str, Any]] = None,
        time_of_day: str = "09:00",
        parameters: Dict[str, Any] = None,
    ) -> ScheduledReport:
        """
        Schedule a recurring report.
        
        Args:
            template_id: Template to use
            name: Schedule name
            frequency: Generation frequency
            recipients: Report recipients
            time_of_day: Time to generate (HH:MM)
            parameters: Report parameters
            
        Returns:
            Scheduled report configuration
        """
        schedule_id = str(uuid.uuid4())[:8]
        
        # Create recipients
        recipient_list = []
        for r in (recipients or []):
            recipient = ReportRecipient(
                id=str(uuid.uuid4())[:8],
                name=r.get("name", ""),
                email=r.get("email"),
                slack_channel=r.get("slack_channel"),
                webhook_url=r.get("webhook_url"),
                delivery_method=DeliveryMethod(r.get("delivery_method", "email")),
            )
            recipient_list.append(recipient)
        
        schedule = ScheduledReport(
            id=schedule_id,
            name=name,
            template_id=template_id,
            frequency=frequency,
            recipients=recipient_list,
            time_of_day=time_of_day,
            parameters=parameters or {},
            next_run=self._calculate_next_run(frequency, time_of_day),
        )
        
        self.schedules[schedule_id] = schedule
        logger.info(f"Scheduled report: {name}", extra={"frequency": frequency.value})
        
        return schedule
    
    def _calculate_next_run(
        self,
        frequency: ReportFrequency,
        time_of_day: str,
    ) -> datetime:
        """Calculate next run time for a schedule."""
        now = datetime.utcnow()
        hour, minute = map(int, time_of_day.split(":"))
        
        if frequency == ReportFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0) + timedelta(hours=1)
        
        elif frequency == ReportFrequency.DAILY:
            next_run = now.replace(hour=hour, minute=minute, second=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        
        elif frequency == ReportFrequency.WEEKLY:
            next_run = now.replace(hour=hour, minute=minute, second=0)
            days_ahead = (7 - now.weekday()) % 7
            if days_ahead == 0 and next_run <= now:
                days_ahead = 7
            next_run += timedelta(days=days_ahead)
        
        elif frequency == ReportFrequency.MONTHLY:
            next_run = now.replace(day=1, hour=hour, minute=minute, second=0)
            if next_run.month == 12:
                next_run = next_run.replace(year=next_run.year + 1, month=1)
            else:
                next_run = next_run.replace(month=next_run.month + 1)
        
        else:
            next_run = now + timedelta(hours=1)
        
        return next_run
    
    async def generate_report(
        self,
        template_id: str,
        parameters: Dict[str, Any] = None,
        format: ReportFormat = None,
    ) -> GeneratedReport:
        """
        Generate a report from a template.
        
        Args:
            template_id: Template to use
            parameters: Report parameters
            format: Override output format
            
        Returns:
            Generated report
        """
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        report_id = str(uuid.uuid4())[:8]
        start_time = datetime.utcnow()
        
        report = GeneratedReport(
            id=report_id,
            template_id=template_id,
            schedule_id=None,
            name=f"{template.name} - {start_time.strftime('%Y-%m-%d')}",
            status=ReportStatus.GENERATING,
            format=format or template.format,
        )
        
        try:
            # Build report
            builder = ReportBuilder(report)
            
            # Collect data from sources
            for section_id in template.sections:
                if section_id in self.data_sources:
                    source = self.data_sources[section_id]
                    try:
                        data = source(**(parameters or {}))
                        builder.add_section(
                            title=section_id.replace("_", " ").title(),
                            content=data,
                            section_type="data",
                        )
                    except Exception as e:
                        logger.error(f"Failed to get data from {section_id}: {e}")
                        builder.add_text(
                            section_id.replace("_", " ").title(),
                            f"Error: {str(e)}",
                        )
            
            report = builder.build()
            
            # Render report
            report.rendered_content = self.renderer.render(report, report.format)
            report.file_size_bytes = len(report.rendered_content.encode("utf-8"))
            
            report.status = ReportStatus.COMPLETED
            report.generated_at = datetime.utcnow()
            report.generation_time_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report.status = ReportStatus.FAILED
        
        self.reports[report_id] = report
        
        logger.info(
            f"Generated report: {report.name}",
            extra={
                "report_id": report_id,
                "status": report.status.value,
                "generation_time_ms": report.generation_time_ms,
            },
        )
        
        return report
    
    async def deliver_report(
        self,
        report_id: str,
        recipients: List[ReportRecipient] = None,
    ) -> Dict[str, Any]:
        """
        Deliver a report to recipients.
        
        Args:
            report_id: Report to deliver
            recipients: Override recipients
            
        Returns:
            Delivery results
        """
        report = self.reports.get(report_id)
        if not report:
            return {"error": "Report not found"}
        
        if report.status != ReportStatus.COMPLETED:
            return {"error": f"Report not ready: {report.status.value}"}
        
        recipients = recipients or []
        results = {"delivered": [], "failed": []}
        
        for recipient in recipients:
            try:
                success = await self._deliver_to_recipient(report, recipient)
                if success:
                    results["delivered"].append(recipient.name)
                    report.delivered_to.append(recipient.name)
                else:
                    results["failed"].append(recipient.name)
                    report.delivery_errors.append(f"Failed: {recipient.name}")
            except Exception as e:
                results["failed"].append(recipient.name)
                report.delivery_errors.append(f"{recipient.name}: {str(e)}")
        
        if results["delivered"]:
            report.status = ReportStatus.DELIVERED
        
        return results
    
    async def _deliver_to_recipient(
        self,
        report: GeneratedReport,
        recipient: ReportRecipient,
    ) -> bool:
        """Deliver report to a single recipient."""
        # Placeholder for actual delivery implementation
        if recipient.delivery_method == DeliveryMethod.EMAIL:
            logger.info(f"Would send email to {recipient.email}")
            # await send_email(recipient.email, report)
        
        elif recipient.delivery_method == DeliveryMethod.SLACK:
            logger.info(f"Would post to Slack channel {recipient.slack_channel}")
            # await post_to_slack(recipient.slack_channel, report)
        
        elif recipient.delivery_method == DeliveryMethod.WEBHOOK:
            logger.info(f"Would POST to webhook {recipient.webhook_url}")
            # await post_to_webhook(recipient.webhook_url, report)
        
        return True
    
    async def run_scheduled_reports(self) -> List[GeneratedReport]:
        """
        Run all due scheduled reports.
        
        Returns:
            List of generated reports
        """
        now = datetime.utcnow()
        generated = []
        
        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue
            
            if schedule.next_run and schedule.next_run <= now:
                try:
                    report = await self.generate_report(
                        template_id=schedule.template_id,
                        parameters=schedule.parameters,
                    )
                    
                    report.schedule_id = schedule.id
                    
                    # Deliver to recipients
                    if schedule.recipients:
                        await self.deliver_report(report.id, schedule.recipients)
                    
                    generated.append(report)
                    
                    # Update schedule
                    schedule.last_run = now
                    schedule.next_run = self._calculate_next_run(
                        schedule.frequency,
                        schedule.time_of_day,
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to run scheduled report: {e}")
        
        return generated
    
    def get_report(self, report_id: str) -> Optional[GeneratedReport]:
        """Get a generated report."""
        return self.reports.get(report_id)
    
    def list_schedules(self) -> List[Dict[str, Any]]:
        """List all scheduled reports."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "frequency": s.frequency.value,
                "enabled": s.enabled,
                "next_run": s.next_run.isoformat() if s.next_run else None,
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "recipient_count": len(s.recipients),
            }
            for s in self.schedules.values()
        ]
    
    def list_reports(
        self,
        limit: int = 20,
        status: Optional[ReportStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List generated reports."""
        reports = list(self.reports.values())
        
        if status:
            reports = [r for r in reports if r.status == status]
        
        # Sort by generation time
        reports.sort(
            key=lambda r: r.generated_at or datetime.min,
            reverse=True,
        )
        
        return [r.to_dict() for r in reports[:limit]]


# Global reporting system instance
reporting_system = AutomatedReporting()


def get_reporting_system() -> AutomatedReporting:
    """Get the global reporting system."""
    return reporting_system
