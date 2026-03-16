"""
HAIRRAP BY YOYO - No-Show AI Dashboard
Executive Overview | AI Insights | Customer Behaviour | Filters
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap so we can import from src/
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from predict import NoShowPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Logo path
# ---------------------------------------------------------------------------
LOGO_PATH = ROOT / "dashboard" / "assets" / "logo.png"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "HAIRRAP BY YOYO",
    page_icon   = str(LOGO_PATH) if LOGO_PATH.exists() else "💇",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------
SERVICE_PRICES = {
    "Bridal Makeup"  : 8000,
    "Hair Coloring"  : 3000,
    "Hair Spa"       : 2000,
    "Facial"         : 1200,
    "Pedicure"       :  800,
    "Manicure"       :  700,
    "Haircut"        :  600,
}

RISK_COLORS = {
    "Low"    : "#2ecc71",
    "Medium" : "#f39c12",
    "High"   : "#e74c3c",
}

PALETTE = px.colors.qualitative.Set2

def _apply(fig):
    """Apply consistent margin to any Plotly figure."""
    fig.update_layout(margin=dict(t=40, b=10))
    return fig

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_predictor() -> NoShowPredictor:
    return NoShowPredictor(str(ROOT / "models"))


@st.cache_data(show_spinner="Scoring 50 000 bookings...")
def load_and_score() -> pd.DataFrame:
    predictor = load_predictor()

    df = pd.read_csv(ROOT / "data" / "raw" / "salon_bookings.csv")
    df["Booking_Time"]     = pd.to_datetime(df["Booking_Time"])
    df["Appointment_Time"] = pd.to_datetime(df["Appointment_Time"])

    # Batch-score
    results  = predictor.predict_batch(df)
    res_df   = pd.DataFrame(results)

    df["no_show_probability"] = res_df["no_show_probability"].values
    df["risk_level"]          = res_df["risk_level"].values

    # Derived helper columns
    df["No_Show"]        = (df["Outcome"] == "No-Show").astype(int)
    df["Is_New_Customer"]= (df["Past_Visit_Count"] == 0).astype(int)
    df["Revenue"]        = df["Service_Type"].map(SERVICE_PRICES).fillna(600)
    df["Revenue_At_Risk"]= df["Revenue"] * df["no_show_probability"]
    df["Appt_Hour"]      = df["Appointment_Time"].dt.hour
    df["Appt_Date"]      = df["Appointment_Time"].dt.date

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    st.metric(label=label, value=value, delta=delta if delta else None,
              delta_color=delta_color)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar Filters  (Step 10)
# ---------------------------------------------------------------------------

def sidebar_filters(df: pd.DataFrame):
    # Logo
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), use_column_width=True)
    else:
        st.sidebar.markdown(
            "<h2 style='text-align:center; color:#e74c3c; font-family:serif;'>"
            "HAIRRAP<br><span style='font-size:0.6em; color:#555;'>BY YOYO</span>"
            "</h2>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")
    st.sidebar.title("Filters")

    # Date range
    min_date = df["Appointment_Time"].dt.date.min()
    max_date = df["Appointment_Time"].dt.date.max()
    date_range = st.sidebar.date_input(
        "Appointment Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Branch
    all_branches = sorted(df["Branch"].unique())
    branches = st.sidebar.multiselect(
        "Branch",
        options=all_branches,
        default=all_branches,
    )

    # Service type
    all_services = sorted(df["Service_Type"].unique())
    services = st.sidebar.multiselect(
        "Service Type",
        options=all_services,
        default=all_services,
    )

    # Risk level
    all_risk = ["Low", "Medium", "High"]
    risk_levels = st.sidebar.multiselect(
        "Risk Level",
        options=all_risk,
        default=all_risk,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("HAIRRAP BY YOYO  |  Jan 2025 - Dec 2025  |  50,000 bookings")

    return date_range, branches, services, risk_levels


def apply_filters(df: pd.DataFrame, date_range, branches, services, risk_levels):
    fdf = df.copy()
    if len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        fdf = fdf[(fdf["Appointment_Time"] >= start) & (fdf["Appointment_Time"] <= end)]
    if branches:
        fdf = fdf[fdf["Branch"].isin(branches)]
    if services:
        fdf = fdf[fdf["Service_Type"].isin(services)]
    if risk_levels:
        fdf = fdf[fdf["risk_level"].isin(risk_levels)]
    return fdf


# ---------------------------------------------------------------------------
# Section 1 — Executive Overview  (Step 7)
# ---------------------------------------------------------------------------

def section_executive_overview(df: pd.DataFrame):
    section_header(
        "Executive Overview",
        "Key business metrics for the filtered selection",
    )

    total       = len(df)
    no_show_pct = df["No_Show"].mean() * 100 if total > 0 else 0
    high_risk   = (df["risk_level"] == "High").sum()
    rev_loss    = df.loc[df["No_Show"] == 1, "Revenue"].sum()
    rev_at_risk = df["Revenue_At_Risk"].sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Bookings", f"{total:,}")
    with c2:
        metric_card("No-Show Rate", f"{no_show_pct:.1f}%",
                    delta=f"{no_show_pct - 24.1:.1f}% vs baseline",
                    delta_color="inverse")
    with c3:
        metric_card("Predicted High-Risk", f"{high_risk:,}",
                    delta=f"{high_risk/total*100:.1f}% of bookings" if total > 0 else "")
    with c4:
        metric_card("Estimated Revenue at Risk", f"Rs.{rev_at_risk:,.0f}",
                    delta=f"Actual loss Rs.{rev_loss:,.0f}")

    st.markdown(" ")

    # Outcome breakdown donut
    c_left, c_right = st.columns([1, 2])
    with c_left:
        outcome_counts = df["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig_donut = px.pie(
            outcome_counts,
            names="Outcome", values="Count",
            hole=0.55,
            color_discrete_sequence=PALETTE,
            title="Booking Outcomes",
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        fig_donut.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_right:
        # Monthly no-show trend
        df_monthly = (
            df.assign(Month=df["Appointment_Time"].dt.to_period("M").astype(str))
              .groupby("Month")
              .agg(Total=("No_Show", "count"), No_Shows=("No_Show", "sum"))
              .assign(Rate=lambda x: x["No_Shows"] / x["Total"] * 100)
              .reset_index()
        )
        fig_trend = px.line(
            df_monthly, x="Month", y="Rate",
            markers=True,
            title="Monthly No-Show Rate (%)",
            labels={"Rate": "No-Show Rate (%)"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig_trend.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 2 — AI Insights  (Step 8)
# ---------------------------------------------------------------------------

def section_ai_insights(df: pd.DataFrame):
    section_header(
        "AI Insights",
        "Model risk scores, feature drivers, and no-show patterns by segment",
    )

    # Row 1: Risk histogram + Feature importance
    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df,
            x="no_show_probability",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            nbins=40,
            title="Risk Score Distribution",
            labels={"no_show_probability": "Predicted No-Show Probability",
                    "risk_level": "Risk Level"},
            category_orders={"risk_level": ["Low", "Medium", "High"]},
        )
        _apply(fig_hist).update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        # LR feature importance (absolute coefficients)
        try:
            predictor = load_predictor()
            coefs     = predictor.model.coef_[0]
            feat_cols = predictor.feature_cols
            imp_df    = (
                pd.DataFrame({"Feature": feat_cols, "Importance": np.abs(coefs)})
                  .sort_values("Importance", ascending=False)
                  .head(15)
            )
            fig_imp = px.bar(
                imp_df.sort_values("Importance"),
                x="Importance", y="Feature",
                orientation="h",
                title="Top 15 Feature Importances (|LR coefficient|)",
                color="Importance",
                color_continuous_scale="Reds",
            )
            _apply(fig_imp).update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")

    # Row 2: No-show by branch + by service
    c3, c4 = st.columns(2)

    with c3:
        branch_df = (
            df.groupby("Branch")
              .agg(No_Show_Rate=("No_Show", "mean"),
                   Count=("No_Show", "count"))
              .assign(No_Show_Rate=lambda x: x["No_Show_Rate"] * 100)
              .sort_values("No_Show_Rate", ascending=True)
              .reset_index()
        )
        fig_branch = px.bar(
            branch_df, x="No_Show_Rate", y="Branch",
            orientation="h",
            text="Count",
            title="No-Show Rate by Branch (%)",
            labels={"No_Show_Rate": "No-Show Rate (%)"},
            color="No_Show_Rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig_branch.update_traces(texttemplate="%{text:,}", textposition="outside")
        _apply(fig_branch).update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_branch, use_container_width=True)

    with c4:
        svc_df = (
            df.groupby("Service_Type")
              .agg(No_Show_Rate=("No_Show", "mean"),
                   Avg_Revenue=("Revenue", "mean"),
                   Count=("No_Show", "count"))
              .assign(No_Show_Rate=lambda x: x["No_Show_Rate"] * 100)
              .sort_values("No_Show_Rate", ascending=True)
              .reset_index()
        )
        fig_svc = px.bar(
            svc_df, x="No_Show_Rate", y="Service_Type",
            orientation="h",
            title="No-Show Rate by Service Type (%)",
            labels={"No_Show_Rate": "No-Show Rate (%)"},
            color="Avg_Revenue",
            color_continuous_scale="Blues",
            color_continuous_midpoint=svc_df["Avg_Revenue"].median(),
        )
        _apply(fig_svc).update_layout(coloraxis_colorbar=dict(title="Avg Price (Rs.)"))
        st.plotly_chart(fig_svc, use_container_width=True)

    # Row 3: Heatmap — Day of Week x Appointment Hour
    pivot = (
        df.assign(Hour=df["Appointment_Time"].dt.hour)
          .groupby(["Day_of_Week", "Hour"])["No_Show"]
          .mean()
          .mul(100)
          .round(1)
          .reset_index()
          .pivot(index="Day_of_Week", columns="Hour", values="No_Show")
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        title="No-Show Rate Heatmap: Day of Week x Appointment Hour (%)",
        labels=dict(x="Hour of Day", y="Day of Week", color="No-Show %"),
    )
    _apply(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 3 — Customer Behaviour  (Step 9)
# ---------------------------------------------------------------------------

def section_customer_behaviour(df: pd.DataFrame):
    section_header(
        "Customer Behaviour",
        "Repeat vs new customers, lead-time patterns, and peak no-show slots",
    )

    # Row 1: New vs Repeat pie + Lead time histogram
    c1, c2 = st.columns(2)

    with c1:
        cust_df = (
            df.groupby("Is_New_Customer")["No_Show"]
              .agg(["count", "mean"])
              .reset_index()
              .assign(
                  Label=lambda x: x["Is_New_Customer"].map(
                      {0: "Repeat Customer", 1: "New Customer"}
                  ),
                  No_Show_Rate=lambda x: (x["mean"] * 100).round(1),
              )
        )
        fig_cust = px.pie(
            cust_df, names="Label", values="count",
            hole=0.5,
            color_discrete_sequence=["#3498db", "#e67e22"],
            title="New vs Repeat Customers",
            custom_data=["No_Show_Rate"],
        )
        fig_cust.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>No-Show Rate: %{customdata[0]}%",
        )
        _apply(fig_cust)
        st.plotly_chart(fig_cust, use_container_width=True)

        # Stat callout
        new_rate    = cust_df.loc[cust_df["Is_New_Customer"] == 1, "No_Show_Rate"].values
        repeat_rate = cust_df.loc[cust_df["Is_New_Customer"] == 0, "No_Show_Rate"].values
        if len(new_rate) and len(repeat_rate):
            st.info(
                f"New customers no-show at **{new_rate[0]:.1f}%**  vs  "
                f"**{repeat_rate[0]:.1f}%** for repeat customers."
            )

    with c2:
        fig_lead = px.histogram(
            df,
            x="Booking_Lead_Time_Days",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            nbins=14,
            title="Booking Lead Time Distribution",
            labels={"Booking_Lead_Time_Days": "Days Before Appointment",
                    "risk_level": "Risk Level"},
            category_orders={"risk_level": ["Low", "Medium", "High"]},
            barmode="stack",
        )
        _apply(fig_lead)
        st.plotly_chart(fig_lead, use_container_width=True)

    # Row 2: Peak no-show hour + No-show by payment method
    c3, c4 = st.columns(2)

    with c3:
        hour_df = (
            df.groupby("Appt_Hour")["No_Show"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "No_Show_Rate", "count": "Bookings"})
              .assign(No_Show_Rate=lambda x: x["No_Show_Rate"] * 100)
        )
        fig_hour = px.bar(
            hour_df, x="Appt_Hour", y="No_Show_Rate",
            title="No-Show Rate by Appointment Hour",
            labels={"Appt_Hour": "Hour of Day", "No_Show_Rate": "No-Show Rate (%)"},
            color="No_Show_Rate",
            color_continuous_scale="Reds",
            text="Bookings",
        )
        fig_hour.update_traces(texttemplate="%{text:,}", textposition="outside",
                               textfont_size=9)
        _apply(fig_hour).update_layout(coloraxis_showscale=False,
                                       xaxis=dict(tickmode="linear", dtick=1))
        st.plotly_chart(fig_hour, use_container_width=True)

    with c4:
        pay_df = (
            df.groupby("Payment_Method")
              .agg(No_Show_Rate=("No_Show", "mean"),
                   Revenue_At_Risk=("Revenue_At_Risk", "sum"),
                   Count=("No_Show", "count"))
              .assign(No_Show_Rate=lambda x: x["No_Show_Rate"] * 100)
              .sort_values("No_Show_Rate", ascending=False)
              .reset_index()
        )
        fig_pay = px.bar(
            pay_df, x="Payment_Method", y="No_Show_Rate",
            title="No-Show Rate by Payment Method",
            labels={"Payment_Method": "Payment Method",
                    "No_Show_Rate": "No-Show Rate (%)"},
            color="No_Show_Rate",
            color_discrete_sequence=PALETTE,
            text="No_Show_Rate",
        )
        fig_pay.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _apply(fig_pay).update_layout(showlegend=False)
        st.plotly_chart(fig_pay, use_container_width=True)
        st.caption(
            "Cash has a 39.4% no-show rate -- 2.6x higher than card payments. "
            "Used as a primary deposit trigger in the action engine."
        )

    # Row 3: No-show rate vs lead time (scatter with trend)
    lead_ns = (
        df.groupby("Booking_Lead_Time_Days")
          .agg(No_Show_Rate=("No_Show", "mean"), Count=("No_Show", "count"))
          .assign(No_Show_Rate=lambda x: x["No_Show_Rate"] * 100)
          .reset_index()
    )
    fig_scatter = px.scatter(
        lead_ns, x="Booking_Lead_Time_Days", y="No_Show_Rate",
        size="Count", trendline="lowess",
        title="No-Show Rate vs Booking Lead Time (bubble size = # bookings)",
        labels={"Booking_Lead_Time_Days": "Lead Time (days)",
                "No_Show_Rate": "No-Show Rate (%)"},
        color_discrete_sequence=["#9b59b6"],
    )
    _apply(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ---------------------------------------------------------------------------
# Section 4 — Booking Explorer (bonus: filtered table)
# ---------------------------------------------------------------------------

def section_booking_explorer(df: pd.DataFrame):
    section_header("Booking Explorer", "Filtered bookings with AI risk scores")

    display_cols = [
        "Booking_ID", "Customer_ID", "Service_Type", "Branch",
        "Appointment_Time", "Day_of_Week", "Payment_Method",
        "Booking_Lead_Time_Days", "Past_Visit_Count",
        "Past_No_Show_Count", "no_show_probability", "risk_level", "Outcome",
    ]
    show_cols = [c for c in display_cols if c in df.columns]

    def highlight_risk(val):
        colors = {"High": "background-color:#fde8e8",
                  "Medium": "background-color:#fef9e7",
                  "Low": "background-color:#eafaf1"}
        return colors.get(val, "")

    styled = (
        df[show_cols]
          .sort_values("no_show_probability", ascending=False)
          .head(500)
          .style
          .applymap(highlight_risk, subset=["risk_level"])
          .format({"no_show_probability": "{:.3f}"})
    )
    st.dataframe(styled, use_container_width=True, height=400)
    st.caption(f"Showing top 500 highest-risk bookings from {len(df):,} filtered records.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Load data
    df_full = load_and_score()

    # Sidebar filters
    date_range, branches, services, risk_levels = sidebar_filters(df_full)
    df = apply_filters(df_full, date_range, branches, services, risk_levels)

    # Header — logo + brand name side by side
    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=90)
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom:0; font-family:serif;'>HAIRRAP "
            "<span style='font-size:0.55em; color:#888;'>BY YOYO</span></h1>"
            "<p style='margin-top:4px; color:#555;'>No-Show Prediction & Business Intelligence Dashboard</p>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"Analysing **{len(df):,}** bookings across **{df['Branch'].nunique()}** branches  "
        f"|  Model: Logistic Regression  |  Threshold: 0.350"
    )

    if df.empty:
        st.warning("No bookings match the current filters. Adjust the sidebar filters.")
        return

    # Sections
    section_executive_overview(df)
    section_ai_insights(df)
    section_customer_behaviour(df)
    section_booking_explorer(df)


if __name__ == "__main__":
    main()
