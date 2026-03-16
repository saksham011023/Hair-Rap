#!/usr/bin/env python
# coding: utf-8

# # 📊 Salon No-Show Dataset – Exploratory Data Analysis (EDA)
# This notebook loads the salon booking dataset and performs basic exploratory analysis.

# ## Cell 1 – Import Libraries

# In[1]:


import pandas as pd
import numpy as np

print('Libraries imported successfully ✅')


# ## Cell 2 – Load Dataset

# In[2]:


# Update the path below to point to your actual dataset file
DATA_PATH = '../data/raw/salon_bookings.csv'

df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded successfully ✅  →  {df.shape[0]} rows, {df.shape[1]} columns')


# ## Cell 3 – Preview the Data (head)

# In[3]:


print('First 5 rows of the dataset:')
df.head()


# ## Cell 4 – Dataset Shape

# In[4]:


print(f'Shape of the dataset: {df.shape}')
print(f'  → Rows   : {df.shape[0]}')
print(f'  → Columns: {df.shape[1]}')


# ## Cell 5 – Column Names

# In[5]:


print('Column names:')
print(df.columns.tolist())


# ## Cell 6 – Data Types

# In[6]:


print('Data types of each column:')
df.dtypes


# ## Cell 7 – Missing Values

# In[7]:


missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %':    missing_pct
})

print('Missing values per column:')
missing_df[missing_df['Missing Count'] > 0]


# ## Cell 8 – Full Dataset Info (dtypes + non-null counts)

# In[8]:


print('Dataset Info:')
df.info()


# ## Cell 9 – Basic Statistical Summary

# In[9]:


print('Statistical summary of numeric columns:')
df.describe()


# ## Cell 10 – Duplicate Rows Check

# In[10]:


duplicate_count = df.duplicated().sum()
print(f'Total duplicate rows: {duplicate_count}')


# ## Cell 11 – Descriptive Statistics for Numerical Features

# In[11]:


# Exclude ID columns (Booking_ID, Customer_ID) -- they are identifiers, not features
numerical_cols = ["Booking_Lead_Time_Days", "Past_Visit_Count",
                  "Past_Cancellation_Count", "Past_No_Show_Count"]

print("=" * 65)
print("NUMERICAL FEATURES - Descriptive Statistics")
print("=" * 65)
df[numerical_cols].describe().round(2)


# ## Cell 12 – Value Counts for Categorical Variables

# In[12]:


# Exclude timestamp columns (Booking_Time, Appointment_Time) -- thousands of unique values
categorical_cols = ["Service_Type", "Branch", "Payment_Method",
                    "Day_of_Week", "Outcome"]

print("=" * 65)
print("CATEGORICAL FEATURES - Value Counts")
print("=" * 65)

for col in categorical_cols:
    vc  = df[col].value_counts()
    pct = df[col].value_counts(normalize=True).mul(100).round(1)
    summary = pd.DataFrame({"Count": vc, "Percentage (%)": pct})
    print(f"\n{col}")
    print(summary.to_string())
    print("-" * 45)


# ## Cell 13 – Appointment Outcome Distribution (Show vs No-Show)

# In[13]:


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Count & percentage ───────────────────────────────────────────────
outcome_counts = df["Outcome"].value_counts()
outcome_pct    = df["Outcome"].value_counts(normalize=True).mul(100).round(1)

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

colors = ["#4CAF50", "#F44336"]          # green = Show, red = No-Show
bars = ax.bar(outcome_counts.index, outcome_counts.values,
              color=colors, width=0.5, edgecolor="white", linewidth=1.2)

# Annotate bars with count + percentage
for bar, pct in zip(bars, outcome_pct.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{bar.get_height():,}\n({pct}%)",
        ha="center", va="bottom", fontsize=12, fontweight="bold"
    )

ax.set_title("Appointment Outcome Distribution", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Outcome", fontsize=12)
ax.set_ylabel("Number of Bookings", fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, outcome_counts.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("../reports/figures/outcome_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nShow     : {outcome_counts['Show']:,}  ({outcome_pct['Show']}%)")
print(f"No-Show  : {outcome_counts['No-Show']:,}  ({outcome_pct['No-Show']}%)")


# ## Cell 14 – No-Show Rate by Day of Week & Appointment Hour

# In[14]:


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Feature engineering ──────────────────────────────────────────────
df["Appointment_Time"] = pd.to_datetime(df["Appointment_Time"])
df["Appt_Hour"]        = df["Appointment_Time"].dt.hour
df["Is_NoShow"]        = (df["Outcome"] == "No-Show").astype(int)

# Ordered days for x-axis
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Aggregate ────────────────────────────────────────────────────────
day_noshow = (
    df.groupby("Day_of_Week")["Is_NoShow"]
    .mean().mul(100).round(1)
    .reindex(day_order)
)

hour_noshow = (
    df.groupby("Appt_Hour")["Is_NoShow"]
    .mean().mul(100).round(1)
)

# ── Plot: side-by-side ───────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("No-Show Rate Analysis", fontsize=15, fontweight="bold", y=1.02)

# --- Left: by Day of Week ---
bar_colors = ["#EF5350" if v == day_noshow.max() else "#42A5F5"
              for v in day_noshow.values]
bars = ax1.bar(day_noshow.index, day_noshow.values,
               color=bar_colors, width=0.6, edgecolor="white", linewidth=1)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1,
             f"{bar.get_height():.1f}%",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_title("No-Show Rate by Day of Week", fontsize=12, fontweight="bold")
ax1.set_xlabel("Day of Week", fontsize=10)
ax1.set_ylabel("No-Show Rate (%)", fontsize=10)
ax1.set_xticklabels(day_order, rotation=30, ha="right")
ax1.set_ylim(0, day_noshow.max() * 1.2)
ax1.spines[["top","right"]].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# --- Right: by Appointment Hour ---
ax2.plot(hour_noshow.index, hour_noshow.values,
         marker="o", color="#AB47BC", linewidth=2, markersize=6,
         markerfacecolor="white", markeredgewidth=2)
ax2.fill_between(hour_noshow.index, hour_noshow.values, alpha=0.15, color="#AB47BC")
ax2.set_title("No-Show Rate by Appointment Hour", fontsize=12, fontweight="bold")
ax2.set_xlabel("Hour of Day (24h)", fontsize=10)
ax2.set_ylabel("No-Show Rate (%)", fontsize=10)
ax2.set_xticks(range(0, 24, 2))
ax2.spines[["top","right"]].set_visible(False)
ax2.grid(linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("../reports/figures/noshow_by_day_and_hour.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Quick text summary ───────────────────────────────────────────────
print(f"Highest no-show day : {day_noshow.idxmax()} ({day_noshow.max()}%)")
print(f"Lowest  no-show day : {day_noshow.idxmin()} ({day_noshow.min()}%)")
print(f"Highest no-show hour: {hour_noshow.idxmax()}:00 ({hour_noshow.max()}%)")
print(f"Lowest  no-show hour: {hour_noshow.idxmin()}:00 ({hour_noshow.min()}%)")


# ## Cell 15 – Grouped Statistics: Behavioural Features vs Outcome

# In[15]:


behavioural_cols = ["Past_Visit_Count", "Past_Cancellation_Count", "Past_No_Show_Count"]

grouped = df.groupby("Outcome")[behavioural_cols].agg(["mean", "median", "std"]).round(3)

print("=" * 65)
print("Grouped Statistics: Behavioural Features vs Outcome")
print("=" * 65)
grouped


# ## Cell 16 – Correlation Heatmap (Numerical Features + Is_NoShow)

# In[16]:


import matplotlib.pyplot as plt
import numpy as np

# Ensure Is_NoShow exists (created in Cell 14; recreate if kernel restarted)
df["Is_NoShow"] = (df["Outcome"] == "No-Show").astype(int)

corr_cols = ["Booking_Lead_Time_Days", "Past_Visit_Count",
             "Past_Cancellation_Count", "Past_No_Show_Count", "Is_NoShow"]

corr = df[corr_cols].corr().round(3)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

labels = ["Lead Time", "Past Visits", "Past Cancels", "Past No-Shows", "Is No-Show"]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)

# Annotate cells
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color=color, fontweight="bold")

ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("../reports/figures/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ## Cell 17 – Box Plots: Behavioural Features by Outcome

# In[17]:


import matplotlib.pyplot as plt

show_data    = df[df["Outcome"] == "Show"]
noshow_data  = df[df["Outcome"] == "No-Show"]

features = [
    ("Past_Visit_Count",        "Past Visit Count"),
    ("Past_Cancellation_Count", "Past Cancellation Count"),
    ("Past_No_Show_Count",      "Past No-Show Count"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Behavioural Features by Appointment Outcome",
             fontsize=13, fontweight="bold", y=1.02)

palette = {"Show": "#4CAF50", "No-Show": "#F44336"}

for ax, (col, label) in zip(axes, features):
    bp = ax.boxplot(
        [show_data[col], noshow_data[col]],
        labels=["Show", "No-Show"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3)
    )
    for patch, clr in zip(bp["boxes"], ["#4CAF50", "#F44336"]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_ylabel("Count", fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("../reports/figures/behavioural_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Mean comparison summary ─────────────────────────────────────────
print("\nMean comparison (Show vs No-Show):")
print(df.groupby("Outcome")[["Past_Visit_Count","Past_Cancellation_Count","Past_No_Show_Count"]].mean().round(3).to_string())

