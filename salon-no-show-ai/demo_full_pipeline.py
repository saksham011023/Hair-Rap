#!/usr/bin/env python3
"""
HAIRRAP FULL PIPELINE DEMO
Shows: Prediction → Business Action → Executive Insights
Perfect for video demos

Usage:
    python demo_full_pipeline.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from predict import NoShowPredictor
from business_actions import ActionEngine

# ============================================================================
# SAMPLE BOOKINGS (Representative scenarios)
# ============================================================================

SAMPLE_BOOKINGS = [
    # SCENARIO 1: Low-Risk Loyal Customer
    {
        "Booking_ID": 1001,
        "Customer_ID": 100,
        "Service_Type": "Haircut",
        "Branch": "Delhi_CP",
        "Booking_Time": "2024-03-10 10:00:00",
        "Appointment_Time": "2024-03-12 14:00:00",
        "Booking_Lead_Time_Days": 2,
        "Past_Visit_Count": 18,
        "Past_Cancellation_Count": 0,
        "Past_No_Show_Count": 0,
        "Payment_Method": "Card",
        "Day_of_Week": "Tuesday",
        "Customer_Latent_Risk": 0.05,
    },
    # ⚠️ SCENARIO 2: New Customer, Cash, Weekend (Medium-High Risk)
    {
        "Booking_ID": 1002,
        "Customer_ID": 101,
        "Service_Type": "Facial",
        "Branch": "Mumbai_Bandra",
        "Booking_Time": "2024-03-08 15:00:00",
        "Appointment_Time": "2024-03-15 10:00:00",
        "Booking_Lead_Time_Days": 7,
        "Past_Visit_Count": 0,
        "Past_Cancellation_Count": 0,
        "Past_No_Show_Count": 0,
        "Payment_Method": "Cash",
        "Day_of_Week": "Saturday",
        "Customer_Latent_Risk": 0.4,
    },
    # 🔴 SCENARIO 3: Chronic No-Shower, Premium Service (Critical Risk)
    {
        "Booking_ID": 1003,
        "Customer_ID": 102,
        "Service_Type": "Bridal Makeup",
        "Branch": "Bangalore_MG",
        "Booking_Time": "2024-03-01 09:00:00",
        "Appointment_Time": "2024-03-15 08:30:00",
        "Booking_Lead_Time_Days": 14,
        "Past_Visit_Count": 8,
        "Past_Cancellation_Count": 1,
        "Past_No_Show_Count": 4,
        "Payment_Method": "Cash",
        "Day_of_Week": "Friday",
        "Customer_Latent_Risk": 0.75,
    },
    # ⚠️ SCENARIO 4: Edge Hour, Long Lead (Medium Risk)
    {
        "Booking_ID": 2004,
        "Customer_ID": 103,
        "Service_Type": "Hair Spa",
        "Branch": "Pune_KP",
        "Booking_Time": "2024-03-02 11:00:00",
        "Appointment_Time": "2024-03-16 19:00:00",
        "Booking_Lead_Time_Days": 14,
        "Past_Visit_Count": 3,
        "Past_Cancellation_Count": 0,
        "Past_No_Show_Count": 1,
        "Payment_Method": "UPI",
        "Day_of_Week": "Saturday",
        "Customer_Latent_Risk": 0.35,
    },
]

# ============================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    DIVIDER = "\033[36m"


def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.RESET}\n")


def print_section(title):
    print(f"{Colors.DIVIDER}{'-'*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.DIVIDER}{'-'*80}{Colors.RESET}")


def print_risk_level(level):
    if level == "High":
        return f"{Colors.RED}HIGH RISK{Colors.RESET}"
    elif level == "Medium":
        return f"{Colors.YELLOW}MEDIUM RISK{Colors.RESET}"
    else:
        return f"{Colors.GREEN}LOW RISK{Colors.RESET}"


def print_action_type(action_type):
    action_colors = {
        "NO_ACTION": f"{Colors.GREEN}{action_type}{Colors.RESET}",
        "SMS_REMINDER": f"{Colors.YELLOW}{action_type}{Colors.RESET}",
        "MULTI_REMINDER": f"{Colors.YELLOW}{action_type}{Colors.RESET}",
        "CALL_REMINDER": f"{Colors.YELLOW}{action_type}{Colors.RESET}",
        "SOFT_DEPOSIT": f"{Colors.YELLOW}{action_type}{Colors.RESET}",
        "HARD_DEPOSIT": f"{Colors.RED}{action_type}{Colors.RESET}",
        "PREPAYMENT": f"{Colors.RED}{action_type}{Colors.RESET}",
    }
    return action_colors.get(action_type, action_type)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    print_header("HAIRRAP AI PIPELINE DEMO")
    print("Showing: Prediction > Business Action > Executive Summary")
    print()

    # Initialize engines
    print(f"{Colors.BOLD}Loading models...{Colors.RESET}")
    predictor = NoShowPredictor(str(ROOT / "models"))
    engine = ActionEngine()
    print(f"{Colors.GREEN}[OK] Models loaded successfully{Colors.RESET}\n")

    # Process bookings
    results = []
    for i, booking in enumerate(SAMPLE_BOOKINGS, 1):
        print_section(f"BOOKING {i}: {booking['Service_Type']} @ {booking['Branch']}")

        # Step 1: Predict
        print(f"{Colors.BOLD}[1] PREDICTION:{Colors.RESET}")
        prediction = predictor.predict_one(booking)
        print(f"    Booking ID: {prediction['booking_id']}")
        print(f"    No-Show Probability: {prediction['no_show_probability']:.1%}")
        print(f"    Risk Level: {print_risk_level(prediction['risk_level'])}")

        # Step 2: Generate Action
        print(f"\n{Colors.BOLD}[2] BUSINESS ACTION:{Colors.RESET}")
        action = engine.evaluate(booking, prediction)
        print(f"    Recommended Action: {print_action_type(action.action_type)}")
        print(f"    Communication: {', '.join(action.channels) if action.channels else 'None'}")
        if action.timing_days_before:
            print(f"    Timing: {', '.join(str(d) + 'd before' for d in action.timing_days_before)}")
        if action.deposit_pct > 0:
            print(f"    Deposit Required: {Colors.RED}{action.deposit_pct}%{Colors.RESET}")
        print(f"    Priority: {action.priority}/5")

        # Step 3: Risk Drivers
        print(f"\n{Colors.BOLD}[3] WHY THIS ACTION?{Colors.RESET}")
        print(f"    Reason: {action.reason}")
        if action.risk_drivers:
            print(f"    Risk Factors Detected:")
            for driver in action.risk_drivers:
                print(f"      • {driver}")
        else:
            print(f"    No elevated risk factors")

        # Step 4: Escalation
        print(f"\n{Colors.BOLD}[4] IF NO RESPONSE:{Colors.RESET}")
        print(f"    {action.escalation_note}")

        results.append(action.to_dict())
        print()

    # ========================================================================
    # EXECUTIVE SUMMARY
    # ========================================================================

    print_header("EXECUTIVE SUMMARY (4 Bookings Analyzed)")

    df_results = pd.DataFrame(results)

    # Count by action type
    print(f"{Colors.BOLD}Bookings by Recommended Action:{Colors.RESET}")
    action_counts = df_results["action_type"].value_counts().to_dict()
    for action_type, count in sorted(action_counts.items()):
        print(f"  • {action_type}: {count}")

    # Count by risk level
    print(f"\n{Colors.BOLD}Bookings by Risk Level:{Colors.RESET}")
    risk_counts = df_results["risk_level"].value_counts().to_dict()
    for level in ["High", "Medium", "Low"]:
        count = risk_counts.get(level, 0)
        if count > 0:
            print(f"  • {print_risk_level(level)}: {count}")

    # Revenue at risk (if we had pricing data)
    print(f"\n{Colors.BOLD}Key Insights:{Colors.RESET}")
    high_risk = len(df_results[df_results["risk_level"] == "High"])
    print(f"  • {Colors.RED}{high_risk} booking(s) require deposit/payment{Colors.RESET}")
    print(f"  • Average no-show probability: {df_results['no_show_probability'].mean():.1%}")

    # Show required actions
    print(f"\n{Colors.BOLD}Staff Actions Required:{Colors.RESET}")
    deposit_actions = df_results[df_results["action_type"].isin(["SOFT_DEPOSIT", "HARD_DEPOSIT", "PREPAYMENT"])]
    if len(deposit_actions) > 0:
        print(f"  • {len(deposit_actions)} booking(s) need payment collection")
        for _, row in deposit_actions.iterrows():
            pct = row["action_type"].split("_")[-1]
            print(f"    - Booking {row['booking_id']}: {row['action_type']} ({Colors.RED}Priority {row['priority']}/5{Colors.RESET})")

    reminder_actions = df_results[df_results["action_type"].isin(["SMS_REMINDER", "MULTI_REMINDER", "CALL_REMINDER"])]
    if len(reminder_actions) > 0:
        print(f"  • {len(reminder_actions)} booking(s) need reminders")
        for _, row in reminder_actions.iterrows():
            print(f"    - Booking {row['booking_id']}: {row['action_type']}")

    no_action = df_results[df_results["action_type"] == "NO_ACTION"]
    if len(no_action) > 0:
        print(f"  • {len(no_action)} booking(s) are low-risk ({Colors.GREEN}no action needed{Colors.RESET})")

    print(f"\n{Colors.BOLD}{Colors.GREEN}Pipeline demo complete!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
