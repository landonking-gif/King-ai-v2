"""
Notification Service - Email, SMS, and webhook notifications for approval workflows.
Integrates with approval manager hooks for real-time alerts.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, List, Dict, Callable
import aiohttp

from src.utils.structured_logging import get_logger
from config.settings import settings

logger = get_logger("notification")


class NotificationChannel(str, Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationRecipient:
    """A notification recipient."""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    discord_webhook: Optional[str] = None
    preferences: Dict[str, bool] = field(default_factory=lambda: {
        "email": True,
        "sms": False,
        "webhook": False,
        "slack": False,
        "discord": False,
    })


@dataclass
class Notification:
    """A notification to be sent."""
    id: str
    subject: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[NotificationRecipient] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
        }


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send a notification to a recipient."""
        pass


class EmailProvider(NotificationProvider):
    """Email notification provider using SMTP or SES."""
    
    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = 587,
        smtp_user: str = None,
        smtp_password: str = None,
        from_email: str = "noreply@king-ai.com",
        use_ses: bool = False,
        aws_region: str = "us-east-1",
    ):
        self.smtp_host = smtp_host or settings.smtp_host
        self.smtp_port = smtp_port or settings.smtp_port
        self.smtp_user = smtp_user or settings.smtp_user
        self.smtp_password = smtp_password or settings.smtp_password
        self.from_email = from_email or settings.smtp_from_email
        self.use_ses = use_ses or settings.use_ses
        self.aws_region = aws_region or settings.aws_region
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send email notification."""
        if not recipient.email:
            logger.warning(f"No email address for recipient {recipient.id}")
            return False
        
        try:
            if self.use_ses:
                return await self._send_ses(notification, recipient)
            else:
                return await self._send_smtp(notification, recipient)
        except Exception as e:
            logger.error(f"Email send failed: {e}", recipient=recipient.email)
            return False
    
    async def _send_smtp(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send via SMTP."""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        if not self.smtp_host:
            logger.warning("SMTP not configured, simulating email send")
            logger.info(f"Would send email to {recipient.email}: {notification.subject}")
            return True
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = notification.subject
        msg['From'] = self.from_email
        msg['To'] = recipient.email
        
        # Add priority header
        if notification.priority == NotificationPriority.URGENT:
            msg['X-Priority'] = '1'
        elif notification.priority == NotificationPriority.HIGH:
            msg['X-Priority'] = '2'
        
        # Create HTML and plain text versions
        text_part = MIMEText(notification.body, 'plain')
        html_body = self._format_html(notification)
        html_part = MIMEText(html_body, 'html')
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp_sync, msg)
        
        logger.info(f"Email sent to {recipient.email}", subject=notification.subject)
        return True
    
    def _send_smtp_sync(self, msg):
        """Synchronous SMTP send."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
    
    async def _send_ses(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send via AWS SES."""
        try:
            import boto3
            
            client = boto3.client('ses', region_name=self.aws_region)
            
            response = client.send_email(
                Source=self.from_email,
                Destination={'ToAddresses': [recipient.email]},
                Message={
                    'Subject': {'Data': notification.subject},
                    'Body': {
                        'Text': {'Data': notification.body},
                        'Html': {'Data': self._format_html(notification)},
                    }
                }
            )
            
            logger.info(f"SES email sent", message_id=response['MessageId'])
            return True
        except Exception as e:
            logger.error(f"SES send failed: {e}")
            return False
    
    def _format_html(self, notification: Notification) -> str:
        """Format notification as HTML email."""
        priority_color = {
            NotificationPriority.LOW: "#6c757d",
            NotificationPriority.NORMAL: "#0d6efd",
            NotificationPriority.HIGH: "#ffc107",
            NotificationPriority.URGENT: "#dc3545",
        }
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #1a1a2e; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; }}
                .priority {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-size: 12px; }}
                .footer {{ background: #e9ecef; padding: 15px; text-align: center; font-size: 12px; border-radius: 0 0 8px 8px; }}
                .action-btn {{ display: inline-block; padding: 10px 20px; background: #7c3aed; color: white; text-decoration: none; border-radius: 4px; margin: 10px 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤴 King AI</h1>
                    <span class="priority" style="background: {priority_color[notification.priority]}; color: white;">
                        {notification.priority.value.upper()}
                    </span>
                </div>
                <div class="content">
                    <h2>{notification.subject}</h2>
                    <p>{notification.body.replace(chr(10), '<br>')}</p>
                    {self._format_metadata(notification.metadata)}
                </div>
                <div class="footer">
                    <p>This is an automated message from King AI v2</p>
                    <p>© 2025 King AI - Autonomous Business Empire</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata as HTML table."""
        if not metadata:
            return ""
        
        rows = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in metadata.items())
        return f"<table style='width: 100%; margin-top: 15px; border-collapse: collapse;'>{rows}</table>"


class SMSProvider(NotificationProvider):
    """SMS notification provider using Twilio or AWS SNS."""
    
    def __init__(
        self,
        twilio_sid: str = None,
        twilio_token: str = None,
        twilio_from: str = None,
        use_sns: bool = False,
        aws_region: str = "us-east-1",
    ):
        self.twilio_sid = twilio_sid or settings.twilio_sid
        self.twilio_token = twilio_token or settings.twilio_token
        self.twilio_from = twilio_from or settings.twilio_from
        self.use_sns = use_sns
        self.aws_region = aws_region
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send SMS notification."""
        if not recipient.phone:
            logger.warning(f"No phone number for recipient {recipient.id}")
            return False
        
        try:
            if self.use_sns:
                return await self._send_sns(notification, recipient)
            else:
                return await self._send_twilio(notification, recipient)
        except Exception as e:
            logger.error(f"SMS send failed: {e}", recipient=recipient.phone)
            return False
    
    async def _send_twilio(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send via Twilio."""
        if not all([self.twilio_sid, self.twilio_token, self.twilio_from]):
            logger.warning("Twilio not configured, simulating SMS send")
            logger.info(f"Would send SMS to {recipient.phone}: {notification.subject}")
            return True
        
        async with aiohttp.ClientSession() as session:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
            auth = aiohttp.BasicAuth(self.twilio_sid, self.twilio_token)
            
            # SMS has character limit, truncate body
            body = f"{notification.subject}: {notification.body}"[:160]
            
            async with session.post(url, auth=auth, data={
                "To": recipient.phone,
                "From": self.twilio_from,
                "Body": body,
            }) as response:
                if response.status in [200, 201]:
                    logger.info(f"SMS sent to {recipient.phone}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Twilio error: {error}")
                    return False
    
    async def _send_sns(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send via AWS SNS."""
        try:
            import boto3
            
            client = boto3.client('sns', region_name=self.aws_region)
            
            body = f"{notification.subject}: {notification.body}"[:160]
            
            response = client.publish(
                PhoneNumber=recipient.phone,
                Message=body,
            )
            
            logger.info(f"SNS SMS sent", message_id=response['MessageId'])
            return True
        except Exception as e:
            logger.error(f"SNS send failed: {e}")
            return False


class WebhookProvider(NotificationProvider):
    """Webhook notification provider."""
    
    def __init__(self, timeout: int = 30, retry_count: int = 3):
        self.timeout = timeout
        self.retry_count = retry_count
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send webhook notification."""
        if not recipient.webhook_url:
            logger.warning(f"No webhook URL for recipient {recipient.id}")
            return False
        
        payload = {
            "event": "notification",
            "data": notification.to_dict(),
            "recipient": {
                "id": recipient.id,
                "name": recipient.name,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        recipient.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status < 400:
                            logger.info(f"Webhook sent to {recipient.webhook_url}")
                            return True
                        else:
                            logger.warning(f"Webhook returned {response.status}")
            except Exception as e:
                logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Webhook failed after {self.retry_count} attempts")
        return False


class SlackProvider(NotificationProvider):
    """Slack notification provider."""
    
    def __init__(self, bot_token: str = None):
        self.bot_token = bot_token or getattr(settings, 'slack_bot_token', None)
    
    async def send(self, notification: Notification, recipient: NotificationRecipient) -> bool:
        """Send Slack notification."""
        channel = recipient.slack_channel
        if not channel:
            logger.warning(f"No Slack channel for recipient {recipient.id}")
            return False
        
        if not self.bot_token:
            logger.warning("Slack not configured, simulating send")
            logger.info(f"Would send Slack to {channel}: {notification.subject}")
            return True
        
        # Build Slack block message
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🤴 {notification.subject}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": notification.body}
            },
        ]
        
        # Add metadata as fields
        if notification.metadata:
            fields = [
                {"type": "mrkdwn", "text": f"*{k}*\n{v}"}
                for k, v in list(notification.metadata.items())[:10]
            ]
            blocks.append({"type": "section", "fields": fields})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    json={
                        "channel": channel,
                        "blocks": blocks,
                        "text": notification.subject,  # Fallback text
                    },
                ) as response:
                    data = await response.json()
                    if data.get("ok"):
                        logger.info(f"Slack message sent to {channel}")
                        return True
                    else:
                        logger.error(f"Slack error: {data.get('error')}")
                        return False
        except Exception as e:
            logger.error(f"Slack send failed: {e}")
            return False


class NotificationService:
    """
    Centralized notification service.
    Manages multiple providers and routes notifications appropriately.
    """
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.notification_history: List[Notification] = []
        self._hooks: Dict[str, List[Callable]] = {
            "notification_sent": [],
            "notification_failed": [],
        }
        
        # Initialize default providers
        self._init_default_providers()
    
    def _init_default_providers(self):
        """Initialize default notification providers."""
        self.providers[NotificationChannel.EMAIL] = EmailProvider()
        self.providers[NotificationChannel.SMS] = SMSProvider()
        self.providers[NotificationChannel.WEBHOOK] = WebhookProvider()
        self.providers[NotificationChannel.SLACK] = SlackProvider()
    
    def register_provider(self, channel: NotificationChannel, provider: NotificationProvider):
        """Register a custom notification provider."""
        self.providers[channel] = provider
    
    def register_recipient(self, recipient: NotificationRecipient):
        """Register a notification recipient."""
        self.recipients[recipient.id] = recipient
    
    def register_hook(self, event: str, callback: Callable):
        """Register a callback for notification events."""
        if event in self._hooks:
            self._hooks[event].append(callback)
    
    async def send(
        self,
        notification: Notification,
        recipients: List[str] = None,
    ) -> Dict[str, bool]:
        """
        Send a notification to specified recipients.
        
        Args:
            notification: The notification to send
            recipients: List of recipient IDs (uses notification.recipients if not specified)
            
        Returns:
            Dict mapping recipient IDs to success/failure
        """
        results = {}
        
        # Determine recipients
        target_recipients = []
        if recipients:
            target_recipients = [self.recipients.get(r) for r in recipients if r in self.recipients]
        elif notification.recipients:
            target_recipients = notification.recipients
        
        if not target_recipients:
            logger.warning("No recipients for notification")
            return {}
        
        # Send to each recipient via their preferred channels
        for recipient in target_recipients:
            recipient_results = []
            
            for channel in notification.channels:
                # Check recipient preference
                if not recipient.preferences.get(channel.value, False):
                    continue
                
                provider = self.providers.get(channel)
                if not provider:
                    logger.warning(f"No provider for channel {channel}")
                    continue
                
                success = await provider.send(notification, recipient)
                recipient_results.append(success)
            
            # Mark success if at least one channel succeeded
            results[recipient.id] = any(recipient_results) if recipient_results else False
        
        # Update notification status
        notification.sent_at = datetime.utcnow()
        self.notification_history.append(notification)
        
        # Trigger hooks
        for callback in self._hooks["notification_sent"]:
            try:
                await callback(notification, results)
            except Exception as e:
                logger.error(f"Notification hook error: {e}")
        
        return results
    
    async def notify_approval_request(
        self,
        request_id: str,
        title: str,
        description: str,
        risk_level: str,
        amount: float = None,
        action_url: str = None,
    ) -> Dict[str, bool]:
        """Send notification for a new approval request."""
        import uuid
        
        metadata = {
            "Request ID": request_id,
            "Risk Level": risk_level,
        }
        if amount is not None:
            metadata["Amount"] = f"${amount:,.2f}"
        if action_url:
            metadata["Action URL"] = action_url
        
        notification = Notification(
            id=str(uuid.uuid4()),
            subject=f"🔔 Approval Required: {title}",
            body=f"{description}\n\nPlease review and take action on this request.",
            priority=NotificationPriority.HIGH if risk_level in ["high", "critical"] else NotificationPriority.NORMAL,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            metadata=metadata,
        )
        
        # Get all admin recipients
        admin_recipients = [r.id for r in self.recipients.values()]
        
        return await self.send(notification, admin_recipients)
    
    async def notify_approval_decision(
        self,
        request_id: str,
        title: str,
        decision: str,  # "approved" or "rejected"
        reviewed_by: str,
        notes: str = None,
    ) -> Dict[str, bool]:
        """Send notification for an approval decision."""
        import uuid
        
        emoji = "✅" if decision == "approved" else "❌"
        
        notification = Notification(
            id=str(uuid.uuid4()),
            subject=f"{emoji} Request {decision.title()}: {title}",
            body=f"The approval request has been {decision} by {reviewed_by}." + 
                 (f"\n\nNotes: {notes}" if notes else ""),
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
            metadata={
                "Request ID": request_id,
                "Decision": decision.title(),
                "Reviewed By": reviewed_by,
            },
        )
        
        admin_recipients = [r.id for r in self.recipients.values()]
        return await self.send(notification, admin_recipients)
    
    async def notify_evolution_proposal(
        self,
        proposal_id: str,
        proposal_type: str,
        description: str,
        confidence_score: float,
    ) -> Dict[str, bool]:
        """Send notification for a new evolution proposal."""
        import uuid
        
        notification = Notification(
            id=str(uuid.uuid4()),
            subject=f"🧬 Evolution Proposal: {proposal_type}",
            body=f"{description}\n\nThis proposal requires human review before execution.",
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            metadata={
                "Proposal ID": proposal_id,
                "Type": proposal_type,
                "Confidence Score": f"{confidence_score:.1%}",
            },
        )
        
        admin_recipients = [r.id for r in self.recipients.values()]
        return await self.send(notification, admin_recipients)
    
    async def notify_circuit_breaker(
        self,
        circuit_name: str,
        state: str,
        failure_count: int,
    ) -> Dict[str, bool]:
        """Send notification for circuit breaker state changes."""
        import uuid
        
        priority = NotificationPriority.URGENT if state == "open" else NotificationPriority.NORMAL
        
        notification = Notification(
            id=str(uuid.uuid4()),
            subject=f"⚡ Circuit Breaker Alert: {circuit_name}",
            body=f"Circuit breaker '{circuit_name}' is now {state.upper()}.\n\nFailure count: {failure_count}",
            priority=priority,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
            metadata={
                "Circuit": circuit_name,
                "State": state,
                "Failures": str(failure_count),
            },
        )
        
        admin_recipients = [r.id for r in self.recipients.values()]
        return await self.send(notification, admin_recipients)


# Singleton instance
notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Get the notification service singleton."""
    return notification_service
