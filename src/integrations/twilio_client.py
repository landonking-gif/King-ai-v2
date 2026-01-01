"""
Twilio Notification Integration.

Provides SMS and voice notifications for critical approvals and alerts.
Complements email notifications for urgent, on-the-go oversight.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx

from src.utils.structured_logging import get_logger
from src.utils.circuit_breaker import CircuitBreaker

logger = get_logger("twilio_client")

# Twilio circuit breaker
twilio_circuit = CircuitBreaker(
    "twilio",
    failure_threshold=5,
    timeout=30.0,
    success_threshold=3
)


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationType(str, Enum):
    """Types of notifications."""
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_EXPIRING = "approval_expiring"
    EVOLUTION_PROPOSED = "evolution_proposed"
    INTEGRATION_DOWN = "integration_down"
    PAYMENT_FAILED = "payment_failed"
    BUSINESS_ALERT = "business_alert"
    SYSTEM_ALERT = "system_alert"


@dataclass
class NotificationResult:
    """Result of sending a notification."""
    success: bool
    message_sid: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class TwilioClient:
    """
    Twilio client for SMS and voice notifications.
    
    Features:
    - SMS notifications for approvals
    - Voice calls for critical alerts
    - Message status tracking
    - Rate limiting to prevent spam
    """
    
    API_URL = "https://api.twilio.com/2010-04-01"
    
    def __init__(
        self,
        account_sid: str = None,
        auth_token: str = None,
        from_number: str = None
    ):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        
        self._configured = all([
            self.account_sid,
            self.auth_token,
            self.from_number
        ])
        
        if not self._configured:
            logger.warning("Twilio not configured - notifications disabled")
        
        self._recent_messages: List[Dict[str, Any]] = []
        self._rate_limit: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Minimum seconds between messages to same number
    
    @property
    def is_configured(self) -> bool:
        """Check if Twilio is properly configured."""
        return self._configured
    
    def _check_rate_limit(self, to_number: str) -> bool:
        """Check if we can send to this number (rate limiting)."""
        last_sent = self._rate_limit.get(to_number)
        if last_sent:
            elapsed = (datetime.utcnow() - last_sent).total_seconds()
            if elapsed < self._rate_limit_seconds:
                logger.warning(
                    f"Rate limited: {self._rate_limit_seconds - elapsed:.0f}s remaining",
                    to_number=to_number[-4:]  # Log only last 4 digits
                )
                return False
        return True
    
    def _update_rate_limit(self, to_number: str):
        """Update rate limit timestamp for a number."""
        self._rate_limit[to_number] = datetime.utcnow()
    
    @twilio_circuit.protect
    async def send_sms(
        self,
        to_number: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM
    ) -> NotificationResult:
        """
        Send an SMS notification.
        
        Args:
            to_number: Recipient phone number (E.164 format: +1234567890)
            message: Message body (max 1600 chars)
            priority: Message priority (affects rate limiting)
            
        Returns:
            NotificationResult with success status
        """
        if not self._configured:
            return NotificationResult(
                success=False,
                error="Twilio not configured"
            )
        
        # Skip rate limit for critical messages
        if priority != NotificationPriority.CRITICAL:
            if not self._check_rate_limit(to_number):
                return NotificationResult(
                    success=False,
                    error="Rate limited"
                )
        
        # Truncate message if too long
        if len(message) > 1600:
            message = message[:1597] + "..."
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.API_URL}/Accounts/{self.account_sid}/Messages.json",
                    auth=(self.account_sid, self.auth_token),
                    data={
                        "From": self.from_number,
                        "To": to_number,
                        "Body": message
                    }
                )
                
                if response.status_code == 201:
                    data = response.json()
                    self._update_rate_limit(to_number)
                    
                    result = NotificationResult(
                        success=True,
                        message_sid=data.get("sid")
                    )
                    
                    logger.info(
                        "SMS sent successfully",
                        message_sid=result.message_sid,
                        to_number=to_number[-4:]
                    )
                    
                    self._recent_messages.append({
                        "type": "sms",
                        "sid": result.message_sid,
                        "to": to_number,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    return result
                else:
                    error = response.json().get("message", "Unknown error")
                    logger.error(f"Twilio API error: {error}")
                    return NotificationResult(
                        success=False,
                        error=error
                    )
                    
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return NotificationResult(
                success=False,
                error=str(e)
            )
    
    @twilio_circuit.protect
    async def make_call(
        self,
        to_number: str,
        twiml_message: str
    ) -> NotificationResult:
        """
        Make a voice call with a spoken message.
        
        Args:
            to_number: Recipient phone number
            twiml_message: Message to speak (will be wrapped in TwiML)
            
        Returns:
            NotificationResult with call SID
        """
        if not self._configured:
            return NotificationResult(
                success=False,
                error="Twilio not configured"
            )
        
        # Wrap message in TwiML
        twiml = f"""
        <Response>
            <Say voice="alice" language="en-US">
                {twiml_message}
            </Say>
            <Pause length="1"/>
            <Say voice="alice" language="en-US">
                Press any key to acknowledge this alert.
            </Say>
            <Gather numDigits="1" timeout="10">
                <Say>Waiting for acknowledgment.</Say>
            </Gather>
        </Response>
        """
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.API_URL}/Accounts/{self.account_sid}/Calls.json",
                    auth=(self.account_sid, self.auth_token),
                    data={
                        "From": self.from_number,
                        "To": to_number,
                        "Twiml": twiml
                    }
                )
                
                if response.status_code == 201:
                    data = response.json()
                    
                    result = NotificationResult(
                        success=True,
                        message_sid=data.get("sid")
                    )
                    
                    logger.info(
                        "Voice call initiated",
                        call_sid=result.message_sid,
                        to_number=to_number[-4:]
                    )
                    
                    return result
                else:
                    error = response.json().get("message", "Unknown error")
                    logger.error(f"Twilio call error: {error}")
                    return NotificationResult(
                        success=False,
                        error=error
                    )
                    
        except Exception as e:
            logger.error(f"Failed to make call: {e}")
            return NotificationResult(
                success=False,
                error=str(e)
            )
    
    async def send_approval_notification(
        self,
        to_number: str,
        approval_id: str,
        description: str,
        amount: float = None,
        expires_in_hours: float = None
    ) -> NotificationResult:
        """
        Send an approval request notification.
        
        Args:
            to_number: Recipient phone number
            approval_id: The approval request ID
            description: Brief description of what needs approval
            amount: Optional dollar amount involved
            expires_in_hours: Hours until the approval expires
        """
        message = f"🔔 King AI Approval Required\n\n"
        message += f"Action: {description}\n"
        
        if amount:
            message += f"Amount: ${amount:,.2f}\n"
        
        if expires_in_hours:
            message += f"Expires: {expires_in_hours:.1f} hours\n"
        
        message += f"\nID: {approval_id[:8]}\n"
        message += "Reply YES to approve, NO to reject."
        
        return await self.send_sms(
            to_number=to_number,
            message=message,
            priority=NotificationPriority.HIGH
        )
    
    async def send_critical_alert(
        self,
        to_number: str,
        alert_type: NotificationType,
        message: str,
        use_voice: bool = False
    ) -> NotificationResult:
        """
        Send a critical system alert.
        
        Args:
            to_number: Recipient phone number
            alert_type: Type of alert
            message: Alert message
            use_voice: If True, make a voice call instead of SMS
        """
        prefix = {
            NotificationType.INTEGRATION_DOWN: "🚨 INTEGRATION DOWN",
            NotificationType.PAYMENT_FAILED: "💳 PAYMENT FAILED",
            NotificationType.BUSINESS_ALERT: "📊 BUSINESS ALERT",
            NotificationType.SYSTEM_ALERT: "⚠️ SYSTEM ALERT",
        }.get(alert_type, "⚠️ ALERT")
        
        full_message = f"{prefix}\n\n{message}"
        
        if use_voice:
            spoken = f"King AI Alert. {alert_type.value.replace('_', ' ')}. {message}"
            return await self.make_call(to_number, spoken)
        else:
            return await self.send_sms(
                to_number=to_number,
                message=full_message,
                priority=NotificationPriority.CRITICAL
            )
    
    async def send_evolution_notification(
        self,
        to_number: str,
        evolution_id: str,
        description: str,
        confidence: float
    ) -> NotificationResult:
        """
        Notify about a new evolution proposal.
        """
        message = f"🧬 King AI Evolution Proposal\n\n"
        message += f"{description}\n\n"
        message += f"Confidence: {confidence*100:.0f}%\n"
        message += f"ID: {evolution_id[:8]}\n"
        message += "Review in dashboard to approve/reject."
        
        return await self.send_sms(
            to_number=to_number,
            message=message,
            priority=NotificationPriority.MEDIUM
        )
    
    def get_recent_messages(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent sent messages."""
        return self._recent_messages[-limit:]


# Global client instance
twilio_client = TwilioClient()


class NotificationManager:
    """
    Unified notification manager that routes to appropriate channels.
    
    Supports multiple notification backends:
    - Twilio (SMS/Voice)
    - Email (SMTP/SES)
    - WebSocket (real-time dashboard)
    """
    
    def __init__(self):
        self.twilio = twilio_client
        self._notification_preferences: Dict[str, Dict] = {}
    
    def set_preferences(
        self,
        user_id: str,
        phone: str = None,
        email: str = None,
        sms_enabled: bool = True,
        voice_enabled: bool = False,
        email_enabled: bool = True
    ):
        """Set notification preferences for a user."""
        self._notification_preferences[user_id] = {
            "phone": phone,
            "email": email,
            "sms_enabled": sms_enabled,
            "voice_enabled": voice_enabled,
            "email_enabled": email_enabled
        }
    
    async def notify(
        self,
        user_id: str,
        notification_type: NotificationType,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, NotificationResult]:
        """
        Send notification through all configured channels.
        
        Returns:
            Dict mapping channel name to result
        """
        results = {}
        prefs = self._notification_preferences.get(user_id, {})
        
        # SMS notification
        if prefs.get("sms_enabled") and prefs.get("phone"):
            results["sms"] = await self.twilio.send_sms(
                to_number=prefs["phone"],
                message=message,
                priority=priority
            )
        
        # Voice call for critical
        if (
            prefs.get("voice_enabled") 
            and prefs.get("phone")
            and priority == NotificationPriority.CRITICAL
        ):
            results["voice"] = await self.twilio.make_call(
                to_number=prefs["phone"],
                twiml_message=message
            )
        
        # Email notification (placeholder - would integrate with existing email service)
        if prefs.get("email_enabled") and prefs.get("email"):
            # TODO: Integrate with src/services/notification.py email service
            results["email"] = NotificationResult(
                success=True,
                message_sid="email_placeholder"
            )
        
        return results


# Global notification manager
notification_manager = NotificationManager()
