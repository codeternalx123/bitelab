"""
PHASE 8: Professional Consultations & Appointments
===================================================

Implements professional consultation platform:
- Verified dietitian/coach profiles
- Appointment booking system
- Secure payment processing (20% platform fee)
- Private 1-on-1 chat sessions
- Video consultation support
- Consultation history and notes
- Rating and review system
- Revenue sharing (80% professional, 20% platform)

Core Features:
- Professional verification and credentialing
- Availability calendar management
- Multi-tier pricing (text, voice, video)
- Secure payment with escrow
- Professional-client messaging
- Consultation notes and recommendations
- Follow-up scheduling
- Analytics and earnings tracking

Architecture:
    User → Browse Professionals → Book Appointment → Pay (80/20 split)
    → Secure Chat/Video Session → Consultation Notes → Review
    → Platform earns 20% commission
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta, time
import json
from collections import defaultdict
import hashlib
import secrets
import re
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfessionalType(Enum):
    """Types of health professionals"""
    REGISTERED_DIETITIAN = "registered_dietitian"
    NUTRITIONIST = "nutritionist"
    HEALTH_COACH = "health_coach"
    FITNESS_COACH = "fitness_coach"
    WELLNESS_COACH = "wellness_coach"
    SPORTS_NUTRITIONIST = "sports_nutritionist"


class ConsultationType(Enum):
    """Types of consultations"""
    TEXT_CHAT = "text_chat"
    VOICE_CALL = "voice_call"
    VIDEO_CALL = "video_call"
    IN_PERSON = "in_person"


class AppointmentStatus(Enum):
    """Status of appointments"""
    PENDING_PAYMENT = "pending_payment"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    REFUNDED = "refunded"


class PaymentStatus(Enum):
    """Payment status"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


@dataclass
class ProfessionalCredentials:
    """
    Professional credentials and verification
    """
    credential_type: str  # RD, RDN, LD, CPT, etc.
    credential_number: str
    issuing_organization: str
    issue_date: datetime
    expiry_date: Optional[datetime] = None
    
    # Verification
    is_verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    # Documents
    document_urls: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if credential is currently valid"""
        if not self.is_verified:
            return False
        if self.expiry_date and self.expiry_date < datetime.now():
            return False
        return True


@dataclass
class ProfessionalProfile:
    """
    Professional's public profile
    
    Visible to users browsing for consultations
    """
    professional_id: str
    user_id: str  # Internal user ID
    
    # Professional info
    professional_type: ProfessionalType
    display_name: str  # Real name for professionals
    title: str  # e.g., "Registered Dietitian, MS"
    bio: str
    
    # Credentials
    credentials: List[ProfessionalCredentials] = field(default_factory=list)
    
    # Specializations
    specializations: List[str] = field(default_factory=list)
    # e.g., ['diabetes', 'sports_nutrition', 'weight_loss', 'pregnancy']
    
    # Experience
    years_experience: int = 0
    languages: List[str] = field(default_factory=list)
    
    # Media
    profile_image_url: Optional[str] = None
    
    # Availability
    is_accepting_clients: bool = True
    consultation_types: List[ConsultationType] = field(default_factory=list)
    
    # Pricing (in USD)
    price_text_chat: Decimal = Decimal("50.00")  # 30-min text session
    price_voice_call: Decimal = Decimal("75.00")  # 30-min voice
    price_video_call: Decimal = Decimal("100.00")  # 30-min video
    
    # Stats
    total_consultations: int = 0
    average_rating: float = 0.0
    review_count: int = 0
    
    # Metadata
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def get_badge(self) -> str:
        """Get verification badge"""
        if any(c.is_valid() for c in self.credentials):
            return "⚕️"
        return ""
    
    def get_price(self, consultation_type: ConsultationType) -> Decimal:
        """Get price for consultation type"""
        if consultation_type == ConsultationType.TEXT_CHAT:
            return self.price_text_chat
        elif consultation_type == ConsultationType.VOICE_CALL:
            return self.price_voice_call
        elif consultation_type == ConsultationType.VIDEO_CALL:
            return self.price_video_call
        return Decimal("0.00")


@dataclass
class AvailabilitySlot:
    """Time slot for professional availability"""
    slot_id: str
    professional_id: str
    
    # Time
    date: datetime
    start_time: time
    end_time: time
    duration_minutes: int = 30
    
    # Status
    is_available: bool = True
    is_booked: bool = False
    booked_by: Optional[str] = None
    
    # Consultation type allowed
    allowed_types: List[ConsultationType] = field(default_factory=list)


@dataclass
class Appointment:
    """
    Scheduled appointment between user and professional
    """
    appointment_id: str
    
    # Parties
    user_id: str
    professional_id: str
    
    # Details
    consultation_type: ConsultationType
    scheduled_date: datetime
    start_time: time
    end_time: time
    duration_minutes: int
    
    # Status
    status: AppointmentStatus = AppointmentStatus.PENDING_PAYMENT
    
    # Pricing (all in USD)
    price: Decimal = Decimal("0.00")
    platform_fee: Decimal = Decimal("0.00")  # 20%
    professional_earnings: Decimal = Decimal("0.00")  # 80%
    
    # Payment
    payment_id: Optional[str] = None
    payment_status: PaymentStatus = PaymentStatus.PENDING
    
    # Notes
    user_notes: str = ""  # What user wants to discuss
    professional_notes: str = ""  # Professional's notes after consultation
    
    # Communication
    chat_session_id: Optional[str] = None
    video_room_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    def calculate_fees(self):
        """Calculate platform fee and professional earnings"""
        self.platform_fee = self.price * Decimal("0.20")  # 20%
        self.professional_earnings = self.price * Decimal("0.80")  # 80%
    
    def can_start(self) -> bool:
        """Check if appointment can start"""
        return (
            self.status == AppointmentStatus.CONFIRMED and
            self.payment_status == PaymentStatus.CAPTURED
        )


@dataclass
class ConsultationMessage:
    """Message in a consultation chat session"""
    message_id: str
    session_id: str
    
    sender_id: str
    sender_role: str  # "user" or "professional"
    
    # Content
    text: str
    attachments: List[str] = field(default_factory=list)  # URLs
    
    # Metadata
    sent_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    
    def is_read(self) -> bool:
        return self.read_at is not None


@dataclass
class ConsultationSession:
    """
    Chat session for a consultation
    
    Secure messaging between user and professional
    """
    session_id: str
    appointment_id: str
    
    user_id: str
    professional_id: str
    
    # Messages
    messages: List[ConsultationMessage] = field(default_factory=list)
    
    # Status
    is_active: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Session notes
    professional_summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def add_message(
        self,
        sender_id: str,
        sender_role: str,
        text: str,
        attachments: Optional[List[str]] = None
    ) -> ConsultationMessage:
        """Add message to session"""
        message = ConsultationMessage(
            message_id=self._generate_message_id(),
            session_id=self.session_id,
            sender_id=sender_id,
            sender_role=sender_role,
            text=text,
            attachments=attachments or []
        )
        
        self.messages.append(message)
        return message
    
    def _generate_message_id(self) -> str:
        return f"msg_{secrets.token_urlsafe(12)}"


@dataclass
class ConsultationReview:
    """User review of a consultation"""
    review_id: str
    appointment_id: str
    
    user_id: str
    professional_id: str
    
    # Rating (1-5)
    rating: int
    
    # Feedback
    review_text: str
    
    # Aspects
    communication_rating: int = 5
    expertise_rating: int = 5
    helpfulness_rating: int = 5
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    is_verified_consultation: bool = True


@dataclass
class PaymentTransaction:
    """
    Payment transaction for appointment
    
    80/20 revenue split:
    - 80% to professional
    - 20% to platform
    """
    transaction_id: str
    appointment_id: str
    
    # Parties
    payer_user_id: str
    professional_id: str
    
    # Amounts (in USD)
    total_amount: Decimal
    platform_fee: Decimal  # 20%
    professional_amount: Decimal  # 80%
    
    # Payment details
    payment_method: str  # "card", "mpesa", "paypal", etc.
    payment_status: PaymentStatus = PaymentStatus.PENDING
    
    # Payment gateway info
    gateway_transaction_id: Optional[str] = None
    gateway_response: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    authorized_at: Optional[datetime] = None
    captured_at: Optional[datetime] = None
    refunded_at: Optional[datetime] = None
    
    # Escrow (hold payment until consultation complete)
    is_escrowed: bool = True
    released_at: Optional[datetime] = None


class ProfessionalVerificationService:
    """
    Verifies professional credentials
    
    Ensures only qualified professionals can offer consultations
    """
    
    def __init__(self):
        # In production, this would integrate with:
        # - Commission on Dietetic Registration (CDR)
        # - State licensing boards
        # - Professional organizations
        self.verified_professionals: Set[str] = set()
        logger.info("ProfessionalVerificationService initialized")
    
    def submit_for_verification(
        self,
        professional_id: str,
        credentials: List[ProfessionalCredentials],
        documents: List[str]
    ) -> bool:
        """
        Submit professional for verification
        
        In production, this would:
        1. Upload credential documents
        2. Verify with issuing organizations
        3. Background checks
        4. Review by admin team
        """
        logger.info(f"Verification submitted for {professional_id}")
        
        # Mock verification (instant approval for demo)
        for credential in credentials:
            credential.is_verified = True
            credential.verified_by = "system"
            credential.verified_at = datetime.now()
        
        self.verified_professionals.add(professional_id)
        
        return True
    
    def is_verified(self, professional_id: str) -> bool:
        """Check if professional is verified"""
        return professional_id in self.verified_professionals


class AvailabilityManager:
    """
    Manages professional availability and scheduling
    """
    
    def __init__(self):
        self.slots: Dict[str, AvailabilitySlot] = {}
        self.professional_slots: Dict[str, List[str]] = defaultdict(list)
        logger.info("AvailabilityManager initialized")
    
    def set_recurring_availability(
        self,
        professional_id: str,
        day_of_week: int,  # 0=Monday, 6=Sunday
        start_time: time,
        end_time: time,
        duration_minutes: int = 30,
        weeks_ahead: int = 4
    ) -> List[AvailabilitySlot]:
        """
        Set recurring availability for a professional
        
        E.g., "Every Monday 9am-5pm for next 4 weeks"
        """
        slots = []
        
        # Start from next occurrence of day_of_week
        today = datetime.now().date()
        days_ahead = (day_of_week - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next week
        
        next_date = today + timedelta(days=days_ahead)
        
        # Create slots for each week
        for week in range(weeks_ahead):
            date = next_date + timedelta(weeks=week)
            
            # Create slots for the day
            current_time = start_time
            while current_time < end_time:
                slot_end = (
                    datetime.combine(datetime.today(), current_time) +
                    timedelta(minutes=duration_minutes)
                ).time()
                
                if slot_end > end_time:
                    break
                
                slot = AvailabilitySlot(
                    slot_id=self._generate_slot_id(),
                    professional_id=professional_id,
                    date=datetime.combine(date, current_time),
                    start_time=current_time,
                    end_time=slot_end,
                    duration_minutes=duration_minutes,
                    allowed_types=[
                        ConsultationType.TEXT_CHAT,
                        ConsultationType.VOICE_CALL,
                        ConsultationType.VIDEO_CALL
                    ]
                )
                
                self.slots[slot.slot_id] = slot
                self.professional_slots[professional_id].append(slot.slot_id)
                slots.append(slot)
                
                # Move to next slot
                current_time = slot_end
        
        logger.info(f"Created {len(slots)} availability slots for {professional_id}")
        
        return slots
    
    def get_available_slots(
        self,
        professional_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[AvailabilitySlot]:
        """Get available slots for professional in date range"""
        available = []
        
        for slot_id in self.professional_slots.get(professional_id, []):
            slot = self.slots.get(slot_id)
            if not slot:
                continue
            
            if (slot.is_available and
                not slot.is_booked and
                start_date <= slot.date <= end_date):
                available.append(slot)
        
        # Sort by date/time
        available.sort(key=lambda s: (s.date, s.start_time))
        
        return available
    
    def book_slot(
        self,
        slot_id: str,
        user_id: str
    ) -> bool:
        """Book a time slot"""
        slot = self.slots.get(slot_id)
        if not slot or not slot.is_available or slot.is_booked:
            return False
        
        slot.is_booked = True
        slot.booked_by = user_id
        
        return True
    
    def release_slot(self, slot_id: str) -> bool:
        """Release a booked slot (e.g., on cancellation)"""
        slot = self.slots.get(slot_id)
        if not slot:
            return False
        
        slot.is_booked = False
        slot.booked_by = None
        
        return True
    
    def _generate_slot_id(self) -> str:
        return f"slot_{secrets.token_urlsafe(12)}"


class PaymentProcessor:
    """
    Processes payments with 80/20 revenue split
    
    Integrates with payment gateways:
    - Credit/Debit cards (Stripe)
    - M-Pesa
    - PayPal
    - Apple Pay / Google Pay
    """
    
    def __init__(self):
        self.transactions: Dict[str, PaymentTransaction] = {}
        self.escrow_balance: Dict[str, Decimal] = defaultdict(Decimal)
        
        # Platform revenue tracking
        self.platform_revenue: Decimal = Decimal("0.00")
        
        logger.info("PaymentProcessor initialized")
    
    def create_payment(
        self,
        appointment: Appointment,
        payment_method: str
    ) -> PaymentTransaction:
        """
        Create payment transaction for appointment
        
        Args:
            appointment: Appointment to pay for
            payment_method: Payment method
            
        Returns:
            PaymentTransaction
        """
        # Calculate split
        appointment.calculate_fees()
        
        transaction = PaymentTransaction(
            transaction_id=self._generate_transaction_id(),
            appointment_id=appointment.appointment_id,
            payer_user_id=appointment.user_id,
            professional_id=appointment.professional_id,
            total_amount=appointment.price,
            platform_fee=appointment.platform_fee,
            professional_amount=appointment.professional_earnings,
            payment_method=payment_method
        )
        
        self.transactions[transaction.transaction_id] = transaction
        
        return transaction
    
    def authorize_payment(
        self,
        transaction_id: str
    ) -> bool:
        """
        Authorize payment (verify funds available)
        
        In production, this calls payment gateway
        """
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        # Mock authorization
        transaction.payment_status = PaymentStatus.AUTHORIZED
        transaction.authorized_at = datetime.now()
        transaction.gateway_transaction_id = f"gw_{secrets.token_urlsafe(16)}"
        
        logger.info(f"Payment authorized: {transaction_id}")
        
        return True
    
    def capture_payment(
        self,
        transaction_id: str
    ) -> bool:
        """
        Capture payment (actually charge)
        
        Puts funds in escrow until consultation completes
        """
        transaction = self.transactions.get(transaction_id)
        if not transaction or transaction.payment_status != PaymentStatus.AUTHORIZED:
            return False
        
        # Capture payment
        transaction.payment_status = PaymentStatus.CAPTURED
        transaction.captured_at = datetime.now()
        
        # Put in escrow
        self.escrow_balance[transaction.professional_id] += transaction.professional_amount
        self.platform_revenue += transaction.platform_fee
        
        logger.info(
            f"Payment captured: {transaction_id} "
            f"(Platform: ${transaction.platform_fee}, "
            f"Professional: ${transaction.professional_amount})"
        )
        
        return True
    
    def release_escrow(
        self,
        transaction_id: str
    ) -> bool:
        """
        Release escrowed funds to professional after consultation
        """
        transaction = self.transactions.get(transaction_id)
        if not transaction or transaction.payment_status != PaymentStatus.CAPTURED:
            return False
        
        if not transaction.is_escrowed:
            return False
        
        # Release from escrow
        transaction.released_at = datetime.now()
        
        logger.info(
            f"Escrow released: {transaction_id} "
            f"(${transaction.professional_amount} to professional)"
        )
        
        return True
    
    def refund_payment(
        self,
        transaction_id: str,
        reason: str
    ) -> bool:
        """Refund payment (e.g., cancellation)"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False
        
        # Refund
        transaction.payment_status = PaymentStatus.REFUNDED
        transaction.refunded_at = datetime.now()
        
        # Remove from escrow
        if transaction.is_escrowed:
            self.escrow_balance[transaction.professional_id] -= transaction.professional_amount
            self.platform_revenue -= transaction.platform_fee
        
        logger.info(f"Payment refunded: {transaction_id} (Reason: {reason})")
        
        return True
    
    def get_professional_earnings(
        self,
        professional_id: str
    ) -> Dict[str, Decimal]:
        """Get professional's earnings summary"""
        total_earned = Decimal("0.00")
        in_escrow = self.escrow_balance.get(professional_id, Decimal("0.00"))
        available = Decimal("0.00")
        
        for transaction in self.transactions.values():
            if transaction.professional_id != professional_id:
                continue
            
            if transaction.payment_status == PaymentStatus.CAPTURED:
                total_earned += transaction.professional_amount
                
                if transaction.released_at:
                    available += transaction.professional_amount
        
        return {
            'total_earned': total_earned,
            'in_escrow': in_escrow,
            'available_for_payout': available
        }
    
    def _generate_transaction_id(self) -> str:
        return f"txn_{secrets.token_urlsafe(16)}"


class ConsultationManager:
    """
    Manages consultations, appointments, and sessions
    """
    
    def __init__(
        self,
        availability_manager: AvailabilityManager,
        payment_processor: PaymentProcessor
    ):
        self.availability_manager = availability_manager
        self.payment_processor = payment_processor
        
        # Storage
        self.appointments: Dict[str, Appointment] = {}
        self.sessions: Dict[str, ConsultationSession] = {}
        self.reviews: Dict[str, ConsultationReview] = {}
        
        # Indexes
        self.user_appointments: Dict[str, List[str]] = defaultdict(list)
        self.professional_appointments: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("ConsultationManager initialized")
    
    def book_appointment(
        self,
        user_id: str,
        professional_id: str,
        slot_id: str,
        consultation_type: ConsultationType,
        price: Decimal,
        user_notes: str = "",
        payment_method: str = "card"
    ) -> Optional[Appointment]:
        """
        Book appointment with professional
        
        Process:
        1. Reserve time slot
        2. Create appointment
        3. Process payment
        4. Confirm appointment
        """
        # Reserve slot
        if not self.availability_manager.book_slot(slot_id, user_id):
            logger.warning(f"Failed to book slot: {slot_id}")
            return None
        
        slot = self.availability_manager.slots[slot_id]
        
        # Create appointment
        appointment = Appointment(
            appointment_id=self._generate_appointment_id(),
            user_id=user_id,
            professional_id=professional_id,
            consultation_type=consultation_type,
            scheduled_date=slot.date,
            start_time=slot.start_time,
            end_time=slot.end_time,
            duration_minutes=slot.duration_minutes,
            price=price,
            user_notes=user_notes
        )
        
        # Calculate fees
        appointment.calculate_fees()
        
        # Create payment
        transaction = self.payment_processor.create_payment(
            appointment,
            payment_method
        )
        
        appointment.payment_id = transaction.transaction_id
        
        # Authorize payment
        if not self.payment_processor.authorize_payment(transaction.transaction_id):
            # Release slot
            self.availability_manager.release_slot(slot_id)
            return None
        
        # Capture payment
        if not self.payment_processor.capture_payment(transaction.transaction_id):
            # Release slot
            self.availability_manager.release_slot(slot_id)
            return None
        
        # Confirm appointment
        appointment.status = AppointmentStatus.CONFIRMED
        appointment.payment_status = PaymentStatus.CAPTURED
        appointment.confirmed_at = datetime.now()
        
        # Store
        self.appointments[appointment.appointment_id] = appointment
        self.user_appointments[user_id].append(appointment.appointment_id)
        self.professional_appointments[professional_id].append(appointment.appointment_id)
        
        logger.info(
            f"Appointment booked: {appointment.appointment_id} "
            f"(User: {user_id}, Professional: {professional_id}, "
            f"Price: ${price})"
        )
        
        return appointment
    
    def start_consultation(
        self,
        appointment_id: str
    ) -> Optional[ConsultationSession]:
        """Start consultation session"""
        appointment = self.appointments.get(appointment_id)
        if not appointment or not appointment.can_start():
            return None
        
        # Create session
        session = ConsultationSession(
            session_id=self._generate_session_id(),
            appointment_id=appointment_id,
            user_id=appointment.user_id,
            professional_id=appointment.professional_id,
            is_active=True,
            started_at=datetime.now()
        )
        
        # Update appointment
        appointment.status = AppointmentStatus.IN_PROGRESS
        appointment.started_at = datetime.now()
        appointment.chat_session_id = session.session_id
        
        self.sessions[session.session_id] = session
        
        logger.info(f"Consultation started: {appointment_id}")
        
        return session
    
    def complete_consultation(
        self,
        appointment_id: str,
        professional_summary: str,
        recommendations: List[str]
    ) -> bool:
        """Complete consultation and release payment"""
        appointment = self.appointments.get(appointment_id)
        if not appointment:
            return False
        
        # Update appointment
        appointment.status = AppointmentStatus.COMPLETED
        appointment.completed_at = datetime.now()
        
        # Update session
        if appointment.chat_session_id:
            session = self.sessions.get(appointment.chat_session_id)
            if session:
                session.is_active = False
                session.ended_at = datetime.now()
                session.professional_summary = professional_summary
                session.recommendations = recommendations
        
        # Release payment from escrow
        if appointment.payment_id:
            self.payment_processor.release_escrow(appointment.payment_id)
        
        logger.info(f"Consultation completed: {appointment_id}")
        
        return True
    
    def cancel_appointment(
        self,
        appointment_id: str,
        reason: str,
        refund: bool = True
    ) -> bool:
        """Cancel appointment"""
        appointment = self.appointments.get(appointment_id)
        if not appointment:
            return False
        
        # Update status
        appointment.status = AppointmentStatus.CANCELLED
        appointment.cancelled_at = datetime.now()
        
        # Refund if requested
        if refund and appointment.payment_id:
            self.payment_processor.refund_payment(
                appointment.payment_id,
                reason
            )
        
        logger.info(f"Appointment cancelled: {appointment_id} (Reason: {reason})")
        
        return True
    
    def add_review(
        self,
        appointment_id: str,
        rating: int,
        review_text: str,
        communication_rating: int = 5,
        expertise_rating: int = 5,
        helpfulness_rating: int = 5
    ) -> Optional[ConsultationReview]:
        """Add review for completed consultation"""
        appointment = self.appointments.get(appointment_id)
        if not appointment or appointment.status != AppointmentStatus.COMPLETED:
            return None
        
        review = ConsultationReview(
            review_id=self._generate_review_id(),
            appointment_id=appointment_id,
            user_id=appointment.user_id,
            professional_id=appointment.professional_id,
            rating=max(1, min(5, rating)),
            review_text=review_text,
            communication_rating=max(1, min(5, communication_rating)),
            expertise_rating=max(1, min(5, expertise_rating)),
            helpfulness_rating=max(1, min(5, helpfulness_rating))
        )
        
        self.reviews[review.review_id] = review
        
        logger.info(f"Review added: {review.review_id} (Rating: {rating}/5)")
        
        return review
    
    def _generate_appointment_id(self) -> str:
        return f"appt_{secrets.token_urlsafe(16)}"
    
    def _generate_session_id(self) -> str:
        return f"session_{secrets.token_urlsafe(16)}"
    
    def _generate_review_id(self) -> str:
        return f"review_{secrets.token_urlsafe(12)}"


class ProfessionalMarketplace:
    """
    Main marketplace for professional consultations
    
    Handles:
    - Professional onboarding
    - User browsing and booking
    - Payment processing with 80/20 split
    - Consultation management
    - Reviews and ratings
    """
    
    def __init__(self):
        self.verification_service = ProfessionalVerificationService()
        self.availability_manager = AvailabilityManager()
        self.payment_processor = PaymentProcessor()
        self.consultation_manager = ConsultationManager(
            self.availability_manager,
            self.payment_processor
        )
        
        # Storage
        self.professionals: Dict[str, ProfessionalProfile] = {}
        
        logger.info("ProfessionalMarketplace initialized")
    
    def onboard_professional(
        self,
        user_id: str,
        professional_type: ProfessionalType,
        display_name: str,
        title: str,
        bio: str,
        credentials: List[ProfessionalCredentials],
        specializations: List[str],
        years_experience: int,
        languages: List[str],
        price_text: Decimal = Decimal("50.00"),
        price_voice: Decimal = Decimal("75.00"),
        price_video: Decimal = Decimal("100.00")
    ) -> Optional[ProfessionalProfile]:
        """
        Onboard a new professional
        
        Process:
        1. Create profile
        2. Submit for verification
        3. Set availability
        """
        professional_id = f"pro_{secrets.token_urlsafe(16)}"
        
        # Create profile
        profile = ProfessionalProfile(
            professional_id=professional_id,
            user_id=user_id,
            professional_type=professional_type,
            display_name=display_name,
            title=title,
            bio=bio,
            credentials=credentials,
            specializations=specializations,
            years_experience=years_experience,
            languages=languages,
            price_text_chat=price_text,
            price_voice_call=price_voice,
            price_video_call=price_video,
            consultation_types=[
                ConsultationType.TEXT_CHAT,
                ConsultationType.VOICE_CALL,
                ConsultationType.VIDEO_CALL
            ]
        )
        
        # Submit for verification
        self.verification_service.submit_for_verification(
            professional_id,
            credentials,
            []  # Document URLs
        )
        
        # Store profile
        self.professionals[professional_id] = profile
        
        logger.info(f"Professional onboarded: {professional_id} ({display_name})")
        
        return profile
    
    def search_professionals(
        self,
        specializations: Optional[List[str]] = None,
        min_rating: float = 0.0,
        max_price: Optional[Decimal] = None,
        consultation_type: Optional[ConsultationType] = None,
        languages: Optional[List[str]] = None
    ) -> List[ProfessionalProfile]:
        """
        Search for professionals
        
        Filter by:
        - Specializations
        - Rating
        - Price
        - Consultation type
        - Languages
        """
        results = []
        
        for profile in self.professionals.values():
            # Skip if not accepting clients
            if not profile.is_accepting_clients:
                continue
            
            # Specialization filter
            if specializations:
                if not any(s in profile.specializations for s in specializations):
                    continue
            
            # Rating filter
            if profile.average_rating < min_rating:
                continue
            
            # Price filter
            if max_price and consultation_type:
                price = profile.get_price(consultation_type)
                if price > max_price:
                    continue
            
            # Consultation type filter
            if consultation_type:
                if consultation_type not in profile.consultation_types:
                    continue
            
            # Language filter
            if languages:
                if not any(lang in profile.languages for lang in languages):
                    continue
            
            results.append(profile)
        
        # Sort by rating and review count
        results.sort(
            key=lambda p: (p.average_rating, p.review_count),
            reverse=True
        )
        
        return results
    
    def get_professional_availability(
        self,
        professional_id: str,
        days_ahead: int = 14
    ) -> List[AvailabilitySlot]:
        """Get available slots for professional"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        
        return self.availability_manager.get_available_slots(
            professional_id,
            start_date,
            end_date
        )
    
    def update_professional_stats(
        self,
        professional_id: str
    ):
        """Update professional's stats from reviews"""
        profile = self.professionals.get(professional_id)
        if not profile:
            return
        
        # Get all reviews
        reviews = [
            r for r in self.consultation_manager.reviews.values()
            if r.professional_id == professional_id
        ]
        
        if reviews:
            profile.review_count = len(reviews)
            profile.average_rating = sum(r.rating for r in reviews) / len(reviews)
        
        # Update consultation count
        appointments = [
            a for a in self.consultation_manager.appointments.values()
            if (a.professional_id == professional_id and
                a.status == AppointmentStatus.COMPLETED)
        ]
        profile.total_consultations = len(appointments)


if __name__ == "__main__":
    logger.info("Testing Phase 8: Professional Consultations")
    
    # Initialize marketplace
    marketplace = ProfessionalMarketplace()
    
    # Test 1: Onboard professional
    logger.info(f"\n{'='*60}")
    logger.info("Test 1: Onboarding Professional")
    
    credentials = [
        ProfessionalCredentials(
            credential_type="RDN",
            credential_number="123456",
            issuing_organization="Commission on Dietetic Registration",
            issue_date=datetime(2020, 1, 1),
            expiry_date=datetime(2025, 12, 31)
        )
    ]
    
    professional = marketplace.onboard_professional(
        user_id="user_dietitian_001",
        professional_type=ProfessionalType.REGISTERED_DIETITIAN,
        display_name="Dr. Sarah Johnson",
        title="Registered Dietitian, MS, RDN",
        bio="15 years experience in diabetes management and sports nutrition",
        credentials=credentials,
        specializations=['diabetes', 'sports_nutrition', 'weight_loss'],
        years_experience=15,
        languages=['English', 'Spanish'],
        price_text=Decimal("60.00"),
        price_voice=Decimal("85.00"),
        price_video=Decimal("120.00")
    )
    
    logger.info(f"\n✅ Professional Onboarded:")
    logger.info(f"   Name: {professional.display_name}")
    logger.info(f"   Title: {professional.title}")
    logger.info(f"   Badge: {professional.get_badge()}")
    logger.info(f"   Specializations: {professional.specializations}")
    logger.info(f"   Pricing:")
    logger.info(f"     Text Chat: ${professional.price_text_chat}")
    logger.info(f"     Voice Call: ${professional.price_voice_call}")
    logger.info(f"     Video Call: ${professional.price_video_call}")
    
    # Test 2: Set availability
    logger.info(f"\n{'='*60}")
    logger.info("Test 2: Setting Professional Availability")
    
    # Monday 9am-5pm
    slots = marketplace.availability_manager.set_recurring_availability(
        professional_id=professional.professional_id,
        day_of_week=0,  # Monday
        start_time=time(9, 0),
        end_time=time(17, 0),
        duration_minutes=30,
        weeks_ahead=4
    )
    
    logger.info(f"\n✅ Created {len(slots)} time slots")
    logger.info(f"   Sample slots:")
    for slot in slots[:3]:
        logger.info(f"     {slot.date.date()} {slot.start_time}-{slot.end_time}")
    
    # Test 3: Search professionals
    logger.info(f"\n{'='*60}")
    logger.info("Test 3: Searching Professionals")
    
    results = marketplace.search_professionals(
        specializations=['diabetes'],
        consultation_type=ConsultationType.VIDEO_CALL,
        max_price=Decimal("150.00")
    )
    
    logger.info(f"\n✅ Found {len(results)} professionals")
    for prof in results:
        logger.info(f"\n   {prof.get_badge()} {prof.display_name}")
        logger.info(f"      {prof.title}")
        logger.info(f"      Video: ${prof.price_video_call}")
        logger.info(f"      ⭐ {prof.average_rating:.1f} ({prof.review_count} reviews)")
    
    # Test 4: Book appointment
    logger.info(f"\n{'='*60}")
    logger.info("Test 4: Booking Appointment")
    
    user_id = "user_patient_001"
    
    # Get available slots
    available_slots = marketplace.get_professional_availability(
        professional.professional_id,
        days_ahead=14
    )
    
    logger.info(f"\n{len(available_slots)} slots available")
    
    if available_slots:
        first_slot = available_slots[0]
        
        appointment = marketplace.consultation_manager.book_appointment(
            user_id=user_id,
            professional_id=professional.professional_id,
            slot_id=first_slot.slot_id,
            consultation_type=ConsultationType.VIDEO_CALL,
            price=professional.price_video_call,
            user_notes="Need help managing my Type 2 diabetes through diet",
            payment_method="card"
        )
        
        if appointment:
            logger.info(f"\n✅ Appointment Booked:")
            logger.info(f"   Appointment ID: {appointment.appointment_id}")
            logger.info(f"   Date: {appointment.scheduled_date.date()}")
            logger.info(f"   Time: {appointment.start_time}-{appointment.end_time}")
            logger.info(f"   Type: {appointment.consultation_type.value}")
            logger.info(f"   Price: ${appointment.price}")
            logger.info(f"   Platform Fee (20%): ${appointment.platform_fee}")
            logger.info(f"   Professional Earnings (80%): ${appointment.professional_earnings}")
            logger.info(f"   Payment Status: {appointment.payment_status.value}")
            
            # Test 5: Start consultation
            logger.info(f"\n{'='*60}")
            logger.info("Test 5: Starting Consultation")
            
            session = marketplace.consultation_manager.start_consultation(
                appointment.appointment_id
            )
            
            if session:
                logger.info(f"\n✅ Consultation Started:")
                logger.info(f"   Session ID: {session.session_id}")
                
                # Add some messages
                session.add_message(
                    sender_id=user_id,
                    sender_role="user",
                    text="Hi Dr. Johnson, thanks for meeting with me!"
                )
                
                session.add_message(
                    sender_id=professional.professional_id,
                    sender_role="professional",
                    text="Hello! I'm happy to help. Tell me about your current diet."
                )
                
                logger.info(f"   Messages exchanged: {len(session.messages)}")
                
                # Test 6: Complete consultation
                logger.info(f"\n{'='*60}")
                logger.info("Test 6: Completing Consultation")
                
                success = marketplace.consultation_manager.complete_consultation(
                    appointment.appointment_id,
                    professional_summary="Patient is managing T2D well. Recommended reducing refined carbs and increasing fiber intake.",
                    recommendations=[
                        "Reduce refined carbohydrates to <100g/day",
                        "Increase fiber to 30g/day",
                        "Add 30 minutes walking after meals",
                        "Monitor blood sugar before and 2hrs after meals",
                        "Follow-up in 4 weeks"
                    ]
                )
                
                if success:
                    logger.info(f"\n✅ Consultation Completed")
                    logger.info(f"   Summary: {session.professional_summary}")
                    logger.info(f"   Recommendations: {len(session.recommendations)}")
                    
                    # Test 7: Add review
                    logger.info(f"\n{'='*60}")
                    logger.info("Test 7: Adding Review")
                    
                    review = marketplace.consultation_manager.add_review(
                        appointment_id=appointment.appointment_id,
                        rating=5,
                        review_text="Dr. Johnson was extremely helpful and knowledgeable!",
                        communication_rating=5,
                        expertise_rating=5,
                        helpfulness_rating=5
                    )
                    
                    if review:
                        logger.info(f"\n✅ Review Added:")
                        logger.info(f"   Rating: {review.rating}/5 ⭐")
                        logger.info(f"   Review: {review.review_text}")
                        
                        # Update professional stats
                        marketplace.update_professional_stats(professional.professional_id)
                        
                        logger.info(f"\n   Updated Professional Stats:")
                        logger.info(f"     Average Rating: {professional.average_rating:.1f}/5")
                        logger.info(f"     Total Consultations: {professional.total_consultations}")
                        logger.info(f"     Reviews: {professional.review_count}")
    
    # Test 8: Professional earnings
    logger.info(f"\n{'='*60}")
    logger.info("Test 8: Professional Earnings")
    
    earnings = marketplace.payment_processor.get_professional_earnings(
        professional.professional_id
    )
    
    logger.info(f"\n💰 Professional Earnings:")
    logger.info(f"   Total Earned: ${earnings['total_earned']}")
    logger.info(f"   In Escrow: ${earnings['in_escrow']}")
    logger.info(f"   Available for Payout: ${earnings['available_for_payout']}")
    
    # Platform revenue
    logger.info(f"\n💰 Platform Revenue:")
    logger.info(f"   Total: ${marketplace.payment_processor.platform_revenue}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Phase 8: Professional Consultations Complete!")
    logger.info(f"\n📊 Summary:")
    logger.info(f"   Professionals: {len(marketplace.professionals)}")
    logger.info(f"   Appointments: {len(marketplace.consultation_manager.appointments)}")
    logger.info(f"   Revenue Split: 80% Professional / 20% Platform")
    logger.info(f"   Platform Earnings: ${marketplace.payment_processor.platform_revenue}")
