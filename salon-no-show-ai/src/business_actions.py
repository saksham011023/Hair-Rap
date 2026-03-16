"""
Business Action Engine - Salon No-Show Intervention System
Converts a risk prediction into a specific, data-grounded recommended action.

Rules are derived from actual dataset statistics:
  Cash payment no-show rate    : 39.4%  (vs Card 14.9% -- 2.6x multiplier)
  Weekend (Sat/Sun) no-show    : ~25.2% (vs Tue 23.0%)
  Facial / Pedicure / Manicure : highest no-show rates (~26%)
  Bridal Makeup / Hair Coloring: lowest no-show rates (~19-21%)
  Mean lead time               : 3.6 days  |  max: 13 days

Action types (in escalating order)
-----------------------------------
  NO_ACTION          -- booking shows no elevated risk; no intervention needed
  SMS_REMINDER       -- single automated SMS 24h before appointment
  MULTI_REMINDER     -- SMS + Email at both 3 days and 1 day before
  CALL_REMINDER      -- personal staff call the day before
  SOFT_DEPOSIT       -- 25 % deposit collected at booking; refunded on attendance
  HARD_DEPOSIT       -- 50 % deposit collected at booking; refunded on attendance
  PREPAYMENT         -- 100 % upfront; refunded only on cancellation > 24 h prior
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data-grounded constants
# ---------------------------------------------------------------------------

# No-show rates by service (from EDA) -- used to decide deposit thresholds
SERVICE_VALUE_TIER = {
    "Bridal Makeup"  : "premium",   # 19.8 % no-show, high revenue
    "Hair Coloring"  : "premium",   # 20.6 % no-show, high revenue / time slot
    "Hair Spa"       : "premium",   # 25.1 % no-show, lengthy slot
    "Haircut"        : "standard",  # 25.1 % no-show, common
    "Facial"         : "standard",  # 26.0 % no-show, highest cancellation
    "Pedicure"       : "standard",  # 25.8 % no-show
    "Manicure"       : "standard",  # 25.3 % no-show
}

# Cash has 2.6x the no-show rate of Card -- primary deposit trigger
HIGH_RISK_PAYMENT_METHODS = {"Cash"}

# Weekend slots carry modestly elevated risk
WEEKEND_DAYS = {"Saturday", "Sunday"}

# Chronic no-shower threshold (rate above this → escalate regardless of band)
CHRONIC_NOSHOWER_RATE = 0.40

# "Long lead" threshold -- bookings made far ahead are more likely forgotten
LONG_LEAD_DAYS = 7

# Probability band boundaries (derived from model threshold 0.35 and tuning)
BAND_CRITICAL  = 0.75
BAND_HIGH      = 0.60
BAND_ELEVATED  = 0.40
BAND_CAUTION   = 0.25


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RiskDiagnosis:
    """
    Identifies which data-grounded factors are active for a booking.
    Built once per booking to drive action selection without repetition.
    """
    probability       : float
    service_tier      : str           # "premium" | "standard"
    is_cash           : bool
    is_weekend        : bool
    is_new_customer   : bool
    is_edge_hour      : bool
    is_long_lead      : bool
    is_chronic        : bool          # no-show rate >= CHRONIC_NOSHOWER_RATE
    no_show_rate      : float
    cancellation_ratio: float
    lead_time_days    : int | None
    service_type      : str
    payment_method    : str
    day_of_week       : str

    @property
    def active_drivers(self) -> list[str]:
        """Return ordered list of risk factors that are present."""
        drivers = []
        if self.is_chronic:
            drivers.append(
                f"chronic no-show history ({self.no_show_rate:.0%} rate)"
            )
        if self.is_cash:
            drivers.append(
                "cash payment (39.4% dataset no-show rate vs 14.9% for card)"
            )
        if self.is_new_customer:
            drivers.append("first-time customer -- no prior visit history")
        if self.is_long_lead:
            drivers.append(
                f"booked {self.lead_time_days}d in advance (mean=3.6d) -- "
                "higher likelihood of forgetting"
            )
        if self.is_edge_hour:
            drivers.append("edge-hour slot (<=9h or >=18h) -- inconvenient timing")
        if self.is_weekend:
            drivers.append(
                f"{self.day_of_week} slot (25.2% weekend no-show rate)"
            )
        if self.cancellation_ratio > 0.25:
            drivers.append(
                f"high cancellation history ({self.cancellation_ratio:.0%} rate)"
            )
        return drivers


@dataclass
class BusinessAction:
    """
    Recommended intervention for a single booking.

    Attributes
    ----------
    action_type        : slug identifying the action tier
    channels           : communication channels to use
    timing_days_before : list of days-before-appointment to trigger each touch
    deposit_pct        : percentage of service value to collect (0 = none)
    reason             : plain-English explanation grounded in data
    risk_drivers       : active factors that determined this action
    priority           : 1 (routine) → 5 (critical)
    escalation_note    : what staff should do if intervention goes unanswered
    booking_id         : passed through from input
    no_show_probability: passed through from prediction
    risk_level         : passed through from prediction
    """
    action_type        : str
    channels           : list[str]
    timing_days_before : list[int]
    deposit_pct        : int
    reason             : str
    risk_drivers       : list[str]
    priority           : int
    escalation_note    : str
    booking_id         : Any  = None
    no_show_probability: float = 0.0
    risk_level         : str   = ""

    def to_dict(self) -> dict:
        return {
            "booking_id"          : self.booking_id,
            "no_show_probability" : self.no_show_probability,
            "risk_level"          : self.risk_level,
            "action_type"         : self.action_type,
            "channels"            : self.channels,
            "timing_days_before"  : self.timing_days_before,
            "deposit_pct"         : self.deposit_pct,
            "reason"              : self.reason,
            "risk_drivers"        : self.risk_drivers,
            "priority"            : self.priority,
            "escalation_note"     : self.escalation_note,
        }

    def summary(self) -> str:
        dep = f"  Deposit       : {self.deposit_pct}%\n" if self.deposit_pct else ""
        drivers = "\n".join(f"    - {d}" for d in self.risk_drivers) or "    (none)"
        timing  = ", ".join(f"{t}d before" for t in self.timing_days_before) or "N/A"
        return (
            f"Booking {self.booking_id}  |  P={self.no_show_probability:.4f}"
            f"  |  {self.risk_level} risk\n"
            f"  Action        : {self.action_type}\n"
            f"  Channels      : {', '.join(self.channels)}\n"
            f"  Timing        : {timing}\n"
            f"{dep}"
            f"  Priority      : {self.priority}/5\n"
            f"  Reason        : {self.reason}\n"
            f"  Risk drivers  :\n{drivers}\n"
            f"  If no response: {self.escalation_note}"
        )


# ---------------------------------------------------------------------------
# Diagnosis builder
# ---------------------------------------------------------------------------

def _diagnose(booking: dict, prediction: dict) -> RiskDiagnosis:
    """Extract and interpret features from booking + prediction context."""

    # Raw booking fields
    service    = booking.get("Service_Type", "")
    payment    = booking.get("Payment_Method", "")
    day        = booking.get("Day_of_Week", "")
    lead       = booking.get("Booking_Lead_Time_Days") or booking.get("Lead_Time_Days")
    past_visits = int(booking.get("Past_Visit_Count", 0))
    past_ns     = int(booking.get("Past_No_Show_Count", 0))
    past_cancel = int(booking.get("Past_Cancellation_Count", 0))

    # Derived rates (safe for new customers)
    ns_rate     = (past_ns     / past_visits) if past_visits > 0 else 0.0
    cancel_rate = (past_cancel / past_visits) if past_visits > 0 else 0.0

    # Edge-hour flag -- use precomputed Is_Edge_Hour if present, else derive
    appt_time = booking.get("Appointment_Time", "")
    is_edge = bool(booking.get("Is_Edge_Hour", 0))
    if not is_edge and appt_time:
        try:
            import pandas as pd
            hour = pd.to_datetime(appt_time).hour
            is_edge = hour <= 9 or hour >= 18
        except Exception:
            pass

    return RiskDiagnosis(
        probability        = prediction.get("no_show_probability", 0.0),
        service_tier       = SERVICE_VALUE_TIER.get(service, "standard"),
        is_cash            = payment in HIGH_RISK_PAYMENT_METHODS,
        is_weekend         = day in WEEKEND_DAYS,
        is_new_customer    = past_visits == 0,
        is_edge_hour       = is_edge,
        is_long_lead       = int(lead or 0) >= LONG_LEAD_DAYS,
        is_chronic         = ns_rate >= CHRONIC_NOSHOWER_RATE,
        no_show_rate       = round(ns_rate, 4),
        cancellation_ratio = round(cancel_rate, 4),
        lead_time_days     = int(lead) if lead is not None else None,
        service_type       = service,
        payment_method     = payment,
        day_of_week        = day,
    )


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

def _action_no_action(diag: RiskDiagnosis) -> BusinessAction:
    return BusinessAction(
        action_type        = "NO_ACTION",
        channels           = [],
        timing_days_before = [],
        deposit_pct        = 0,
        reason             = (
            f"P={diag.probability:.2%} is below the caution threshold ({BAND_CAUTION:.0%}). "
            "All booking signals are within normal range."
        ),
        risk_drivers       = [],
        priority           = 1,
        escalation_note    = "No escalation needed.",
    )


def _action_sms_reminder(diag: RiskDiagnosis, days: list[int]) -> BusinessAction:
    drivers = diag.active_drivers
    trigger = drivers[0] if drivers else f"P={diag.probability:.2%} in caution zone"
    return BusinessAction(
        action_type        = "SMS_REMINDER",
        channels           = ["SMS"],
        timing_days_before = days,
        deposit_pct        = 0,
        reason             = (
            f"Mildly elevated risk (P={diag.probability:.2%}). "
            f"Primary driver: {trigger}. "
            "A timely automated reminder is the lowest-friction effective intervention."
        ),
        risk_drivers       = drivers,
        priority           = 2,
        escalation_note    = "If SMS undelivered, send WhatsApp. No deposit required.",
    )


def _action_multi_reminder(diag: RiskDiagnosis) -> BusinessAction:
    days    = [3, 1] if (diag.is_long_lead or (diag.lead_time_days or 0) >= 3) else [1]
    drivers = diag.active_drivers
    return BusinessAction(
        action_type        = "MULTI_REMINDER",
        channels           = ["SMS", "Email"],
        timing_days_before = days,
        deposit_pct        = 0,
        reason             = (
            f"Elevated risk (P={diag.probability:.2%}) without a single dominant "
            "override factor. Multi-channel reminders reduce no-show probability "
            "without creating friction via deposit."
        ),
        risk_drivers       = drivers,
        priority           = 3,
        escalation_note    = (
            "If both channels undelivered, escalate to CALL_REMINDER. "
            "Flag slot for waitlist standby."
        ),
    )


def _action_call_reminder(diag: RiskDiagnosis) -> BusinessAction:
    drivers = diag.active_drivers
    return BusinessAction(
        action_type        = "CALL_REMINDER",
        channels           = ["Call", "SMS"],
        timing_days_before = [1],
        deposit_pct        = 0,
        reason             = (
            f"High-value {diag.service_type} service at an inconvenient "
            f"edge-hour slot (P={diag.probability:.2%}). "
            "Personal call maximises confirmation rates for time-intensive bookings."
        ),
        risk_drivers       = drivers,
        priority           = 3,
        escalation_note    = "If unreachable by phone, request confirmation SMS reply. Flag for waitlist.",
    )


def _action_soft_deposit(diag: RiskDiagnosis) -> BusinessAction:
    drivers = diag.active_drivers
    trigger = drivers[0] if drivers else "elevated probability"
    return BusinessAction(
        action_type        = "SOFT_DEPOSIT",
        channels           = ["SMS", "Email"],
        timing_days_before = [2, 1],
        deposit_pct        = 25,
        reason             = (
            f"25% refundable deposit warranted. P={diag.probability:.2%}. "
            f"Key driver: {trigger}. "
            "A low-friction deposit filters casual no-shows while remaining "
            "fair to genuine customers."
        ),
        risk_drivers       = drivers,
        priority           = 3,
        escalation_note    = "If deposit not paid within 24h of booking, send one follow-up. Cancel if unpaid at T-3d.",
    )


def _action_hard_deposit(diag: RiskDiagnosis) -> BusinessAction:
    drivers = diag.active_drivers
    trigger = drivers[0] if drivers else "high probability"
    pct     = 50
    note    = "If deposit not paid within 12h of booking, offer slot to waitlist."
    if diag.service_tier == "premium" and diag.probability >= BAND_HIGH:
        pct  = 50
        note = (
            f"Premium {diag.service_type} slot. If deposit unpaid within 12h, "
            "release to waitlist immediately."
        )
    return BusinessAction(
        action_type        = "HARD_DEPOSIT",
        channels           = ["SMS", "Email", "Call"],
        timing_days_before = [3, 1],
        deposit_pct        = pct,
        reason             = (
            f"50% refundable deposit required. P={diag.probability:.2%}. "
            f"Primary driver: {trigger}. "
            "Cash dataset no-show rate is 39.4% -- deposit is the most effective "
            "deterrent for uncommitted bookings."
            if diag.is_cash else
            f"50% refundable deposit required. P={diag.probability:.2%}. "
            f"Primary driver: {trigger}."
        ),
        risk_drivers       = drivers,
        priority           = 4,
        escalation_note    = note,
    )


def _action_prepayment(diag: RiskDiagnosis) -> BusinessAction:
    drivers = diag.active_drivers
    return BusinessAction(
        action_type        = "PREPAYMENT",
        channels           = ["Call", "SMS", "Email"],
        timing_days_before = [3, 1],
        deposit_pct        = 100,
        reason             = (
            f"Full prepayment required. P={diag.probability:.2%}. "
            f"This booking combines: {'; '.join(drivers[:3])}. "
            "The combination of chronic no-show history on a premium/high-revenue "
            "service slot justifies full upfront collection to protect staff time."
        ),
        risk_drivers       = drivers,
        priority           = 5,
        escalation_note    = (
            "If customer refuses prepayment, offer: (1) reschedule to weekday "
            "midday slot to reduce risk, or (2) card-on-file authorisation. "
            "Do NOT confirm booking without financial commitment."
        ),
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ActionEngine:
    """
    Converts a prediction result into a specific business action.

    All rules are grounded in the dataset's observed statistics rather than
    generic probability bands -- payment method, service tier, customer history,
    slot timing, and lead time all influence the output independently.
    """

    def evaluate(self, booking: dict, prediction: dict) -> BusinessAction:
        """
        Evaluate a booking and return the recommended BusinessAction.

        Args:
            booking   : Raw booking fields dict (same schema as training data).
            prediction: Output of NoShowPredictor.predict_one() or predict_with_flag().

        Returns:
            BusinessAction with full intervention details.
        """
        diag = _diagnose(booking, prediction)
        prob = diag.probability

        action = self._select_action(diag, prob)
        action.booking_id           = prediction.get("booking_id")
        action.no_show_probability  = prob
        action.risk_level           = prediction.get("risk_level", "")
        return action

    def _select_action(self, d: RiskDiagnosis, prob: float) -> BusinessAction:

        # ------------------------------------------------------------------
        # BAND 0: Below caution -- no intervention
        # ------------------------------------------------------------------
        if prob < BAND_CAUTION:
            return _action_no_action(d)

        # ------------------------------------------------------------------
        # BAND 1: Caution  (0.25 - 0.40)
        # Low-friction interventions only
        # ------------------------------------------------------------------
        if prob < BAND_ELEVATED:
            # Cash payment alone at this band warrants a soft deposit
            # (data: cash 39.4% no-show -- above the 0.25 caution threshold by itself)
            if d.is_cash:
                return _action_soft_deposit(d)

            # Long advance booking -- multi-touch reminder to prevent forgetting
            if d.is_long_lead:
                return _action_sms_reminder(d, days=[3, 1])

            # New customer + edge hour -- double the uncertainty
            if d.is_new_customer and d.is_edge_hour:
                return _action_multi_reminder(d)

            # Default caution: single SMS reminder
            return _action_sms_reminder(d, days=[1])

        # ------------------------------------------------------------------
        # BAND 2: Elevated  (0.40 - 0.60)
        # Deposits enter on strong single-factor triggers
        # ------------------------------------------------------------------
        if prob < BAND_HIGH:
            # Chronic no-shower: history speaks louder than any single factor
            if d.is_chronic:
                return _action_hard_deposit(d)

            # Cash x weekend = highest double-factor no-show risk
            if d.is_cash and d.is_weekend:
                return _action_hard_deposit(d)

            # Cash x premium service = revenue exposure warrants deposit
            if d.is_cash and d.service_tier == "premium":
                return _action_hard_deposit(d)

            # New customer paying cash: no history, no commitment
            if d.is_new_customer and d.is_cash:
                return _action_soft_deposit(d)

            # Premium service + edge hour: personal call is most effective
            if d.service_tier == "premium" and d.is_edge_hour:
                return _action_call_reminder(d)

            # Long lead + new customer: high uncertainty, multi-channel
            if d.is_long_lead and d.is_new_customer:
                return _action_multi_reminder(d)

            # Default elevated: SMS + Email multi-reminder
            return _action_multi_reminder(d)

        # ------------------------------------------------------------------
        # BAND 3: High  (0.60 - 0.75)
        # Financial commitment required in most paths
        # ------------------------------------------------------------------
        if prob < BAND_CRITICAL:
            # Chronic no-shower on premium slot: hard deposit is non-negotiable
            if d.is_chronic and d.service_tier == "premium":
                return _action_hard_deposit(d)

            # Chronic no-shower regardless of service
            if d.is_chronic:
                return _action_hard_deposit(d)

            # Cash + multi-factor: default to hard deposit
            if d.is_cash:
                return _action_hard_deposit(d)

            # Premium service: protect the long time slot
            if d.service_tier == "premium":
                return _action_hard_deposit(d)

            # All other high-risk: hard deposit + multi-channel reminders
            return _action_hard_deposit(d)

        # ------------------------------------------------------------------
        # BAND 4: Critical  (>= 0.75)
        # Full prepayment for premium; hard deposit minimum for all others
        # ------------------------------------------------------------------
        # Chronic + premium at critical probability: full prepayment
        if d.is_chronic and d.service_tier == "premium":
            return _action_prepayment(d)

        # Chronic no-shower at critical probability regardless of service
        if d.is_chronic:
            return _action_prepayment(d)

        # Cash + critical: prepayment (empirically this segment has 39.4% base rate)
        if d.is_cash and d.service_tier == "premium":
            return _action_prepayment(d)

        # All other critical: hard deposit + immediate call
        return _action_hard_deposit(d)


# ---------------------------------------------------------------------------
# Convenience integration with NoShowPredictor
# ---------------------------------------------------------------------------

def evaluate_booking(booking: dict, models_dir: str = None) -> dict:
    """
    End-to-end: predict risk then generate business action for one booking.

    Loads model from disk on every call. Use NoShowPredictor + ActionEngine
    directly when processing many bookings.

    Returns:
        Merged dict containing prediction fields and action fields.
    """
    import sys
    sys.path.insert(0, str(__file__))
    from predict import NoShowPredictor

    predictor = NoShowPredictor(models_dir)
    prediction = predictor.predict_one(booking)

    engine = ActionEngine()
    action = engine.evaluate(booking, prediction)
    return action.to_dict()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from predict import NoShowPredictor

    predictor = NoShowPredictor()
    engine    = ActionEngine()

    test_bookings = [
        # 1. Loyal card customer, midday weekday -- should be NO_ACTION
        {
            "Booking_ID": 101, "Customer_ID": 1,
            "Service_Type": "Haircut", "Branch": "Delhi_CP",
            "Booking_Time": "2024-06-10 09:00:00",
            "Appointment_Time": "2024-06-11 14:00:00",
            "Booking_Lead_Time_Days": 1,
            "Past_Visit_Count": 18, "Past_Cancellation_Count": 0, "Past_No_Show_Count": 0,
            "Payment_Method": "Card", "Day_of_Week": "Tuesday",
            "Customer_Latent_Risk": 0.0,
        },
        # 2. New customer, cash, Saturday -- SOFT/HARD DEPOSIT
        {
            "Booking_ID": 202, "Customer_ID": 2,
            "Service_Type": "Facial", "Branch": "Mumbai_Bandra",
            "Booking_Time": "2024-06-08 10:00:00",
            "Appointment_Time": "2024-06-12 10:00:00",
            "Booking_Lead_Time_Days": 4,
            "Past_Visit_Count": 0, "Past_Cancellation_Count": 0, "Past_No_Show_Count": 0,
            "Payment_Method": "Cash", "Day_of_Week": "Saturday",
            "Customer_Latent_Risk": 0.3,
        },
        # 3. Chronic no-show customer, Bridal Makeup, early slot
        {
            "Booking_ID": 303, "Customer_ID": 3,
            "Service_Type": "Bridal Makeup", "Branch": "Chennai",
            "Booking_Time": "2024-06-01 08:00:00",
            "Appointment_Time": "2024-06-10 08:30:00",
            "Booking_Lead_Time_Days": 9,
            "Past_Visit_Count": 10, "Past_Cancellation_Count": 2, "Past_No_Show_Count": 5,
            "Payment_Method": "Cash", "Day_of_Week": "Sunday",
            "Customer_Latent_Risk": 0.8,
        },
        # 4. Returning customer, UPI, midday Wednesday -- low risk
        {
            "Booking_ID": 404, "Customer_ID": 4,
            "Service_Type": "Hair Coloring", "Branch": "Pune_KP",
            "Booking_Time": "2024-06-09 11:00:00",
            "Appointment_Time": "2024-06-12 12:00:00",
            "Booking_Lead_Time_Days": 3,
            "Past_Visit_Count": 8, "Past_Cancellation_Count": 1, "Past_No_Show_Count": 1,
            "Payment_Method": "UPI", "Day_of_Week": "Wednesday",
            "Customer_Latent_Risk": 0.1,
        },
        # 5. Moderate risk, long lead, edge hour, cash
        {
            "Booking_ID": 505, "Customer_ID": 5,
            "Service_Type": "Hair Spa", "Branch": "Surat",
            "Booking_Time": "2024-06-02 09:00:00",
            "Appointment_Time": "2024-06-12 19:00:00",
            "Booking_Lead_Time_Days": 10,
            "Past_Visit_Count": 4, "Past_Cancellation_Count": 1, "Past_No_Show_Count": 1,
            "Payment_Method": "Cash", "Day_of_Week": "Friday",
            "Customer_Latent_Risk": 0.5,
        },
    ]

    print("\n" + "#"*70)
    print("#   BUSINESS ACTION ENGINE -- DEMO")
    print("#"*70)

    for booking in test_bookings:
        pred   = predictor.predict_one(booking)
        action = engine.evaluate(booking, pred)
        print()
        print(action.summary())
        print("-"*70)
