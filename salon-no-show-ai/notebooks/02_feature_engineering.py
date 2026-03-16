#!/usr/bin/env python
# coding: utf-8

# # 🔧 Salon No-Show Dataset – Feature Engineering
# Features created from raw timestamps and customer history:
# - **Lead_Time_Days**, **Appt_Hour** — from timestamps
# - **Customer_No_Show_Rate**, **Cancellation_Ratio** — historical ratios
# - **Is_New_Customer** — flag for first-time customers
# - **Successful_Visit_Rate** — loyalty metric
# - **Is_Edge_Hour** — appointments at very early/late hours

# ## Cell 1 – Import Libraries

# In[1]:


import pandas as pd
import numpy as np

print('Libraries imported successfully ✅')


# ## Cell 2 – Load Dataset

# In[2]:


DATA_PATH = '../data/raw/salon_bookings.csv'
df = pd.read_csv(DATA_PATH)
print(f'Dataset loaded ✅  →  {df.shape[0]:,} rows, {df.shape[1]} columns')
df.head(3)


# ## Cell 3 – Parse Timestamp Columns

# In[3]:


df['Booking_Time']     = pd.to_datetime(df['Booking_Time'])
df['Appointment_Time'] = pd.to_datetime(df['Appointment_Time'])
print('Timestamps parsed ✅')
print(df[['Booking_Time', 'Appointment_Time']].dtypes)


# ## Cell 4 – Feature 1: Lead Time in Days
# **.dt.days** on the raw timedelta floors total elapsed hours to whole days.

# In[4]:


df['Lead_Time_Days'] = (df['Appointment_Time'] - df['Booking_Time']).dt.days
print(f'Lead_Time_Days  →  Min: {df["Lead_Time_Days"].min()}  Max: {df["Lead_Time_Days"].max()}  Mean: {df["Lead_Time_Days"].mean():.2f}')


# ## Cell 5 – Feature 2: Hour of Appointment

# In[5]:


df['Appt_Hour'] = df['Appointment_Time'].dt.hour
print(f'Appt_Hour  →  Min: {df["Appt_Hour"].min()}  Max: {df["Appt_Hour"].max()}  Mean: {df["Appt_Hour"].mean():.2f}')


# ## Cell 6 – Feature 3: Customer No-Show Rate
# **Past_No_Show_Count / Past_Visit_Count** — safeguarded for new customers (0 visits → 0.0).

# In[6]:


df['Customer_No_Show_Rate'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    df['Past_No_Show_Count'] / df['Past_Visit_Count']
).round(4)
print('Customer_No_Show_Rate ✅')
print(df['Customer_No_Show_Rate'].describe().round(4).to_string())


# ## Cell 7 – Feature 4: Cancellation Ratio

# In[7]:


df['Cancellation_Ratio'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    df['Past_Cancellation_Count'] / df['Past_Visit_Count']
).round(4)
print('Cancellation_Ratio ✅')
print(df['Cancellation_Ratio'].describe().round(4).to_string())


# ## Cell 8 – Feature 5: Is New Customer
# **Definition:** `1` if `Past_Visit_Count == 0` (first-time customer), else `0`.
# 
# New customers have no track record — the model treats them as uncertain and they tend to have higher no-show rates. This captures that signal directly without keeping the raw count.

# In[8]:


df['Is_New_Customer'] = (df['Past_Visit_Count'] == 0).astype(int)

vc  = df['Is_New_Customer'].value_counts().rename({0: 'Returning (0)', 1: 'New (1)'})
pct = df['Is_New_Customer'].value_counts(normalize=True).mul(100).round(1).rename({0: 'Returning (0)', 1: 'New (1)'})
print('Is_New_Customer – distribution:')
print(pd.DataFrame({'Count': vc, 'Percentage (%)': pct}).to_string())


# ## Cell 9 – Feature 6: Successful Visit Rate
# **Definition:** Fraction of past visits that were completed successfully (neither no-show nor cancellation).
# 
# ```
# Successful_Visit_Rate = (Past_Visit_Count - Past_No_Show_Count - Past_Cancellation_Count)
#                         / Past_Visit_Count
# ```
# 
# This is the **loyalty metric** — a customer with a 0.9 rate is reliable; 0.2 means 80% of bookings ended badly. Complements `Customer_No_Show_Rate` by capturing cancellations too.

# In[9]:


good_visits = (df['Past_Visit_Count']
               - df['Past_No_Show_Count']
               - df['Past_Cancellation_Count'])

df['Successful_Visit_Rate'] = np.where(
    df['Past_Visit_Count'] == 0, 0.0,
    good_visits / df['Past_Visit_Count']
).round(4)

# Clip to [0, 1] in case of data inconsistencies
df['Successful_Visit_Rate'] = df['Successful_Visit_Rate'].clip(0, 1)

print('Successful_Visit_Rate ✅')
print(df['Successful_Visit_Rate'].describe().round(4).to_string())


# ## Cell 10 – Feature 7: Is Edge Hour
# **Definition:** `1` if appointment is at `≤ 9:00` (early morning) or `≥ 18:00` (evening), else `0`.
# 
# Edge-hour appointments have higher no-show rates — early morning slots are easy to forget, late-evening slots often get bumped by end-of-day plans. This converts the continuous `Appt_Hour` into a high-signal binary flag.

# In[10]:


df['Is_Edge_Hour'] = (
    (df['Appointment_Time'].dt.hour <= 9) |
    (df['Appointment_Time'].dt.hour >= 18)
).astype(int)

vc  = df['Is_Edge_Hour'].value_counts().rename({0: 'Mid-day (0)', 1: 'Edge hour (1)'})
pct = df['Is_Edge_Hour'].value_counts(normalize=True).mul(100).round(1).rename({0: 'Mid-day (0)', 1: 'Edge hour (1)'})
print('Is_Edge_Hour – distribution:')
print(pd.DataFrame({'Count': vc, 'Percentage (%)': pct}).to_string())


# ## Cell 11 – One-Hot Encode Categorical Variables

# In[11]:


cat_cols = ['Service_Type', 'Branch', 'Payment_Method', 'Day_of_Week']

for col in cat_cols:
    print(f'{col} ({df[col].nunique()} unique): {sorted(df[col].unique())}')

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
ohe_cols = [c for c in df_encoded.columns if any(c.startswith(f'{col}_') for col in cat_cols)]
print(f'\nOne-hot columns created ({len(ohe_cols)})')
print(f'DataFrame shape after encoding: {df_encoded.shape}')


# ## Cell 12 – Define Target Variable

# In[12]:


df_encoded['No_Show'] = (df_encoded['Outcome'] == 'No-Show').astype(int)

vc  = df_encoded['No_Show'].value_counts().rename({0: 'Show (0)', 1: 'No-Show (1)'})
pct = df_encoded['No_Show'].value_counts(normalize=True).mul(100).round(1).rename({0: 'Show (0)', 1: 'No-Show (1)'})
print('Target variable – No_Show:')
print(pd.DataFrame({'Count': vc, 'Percentage (%)': pct}).to_string())


# ## Cell 13 – Split into Features (X) and Target (y)
# 
# | Column(s) | Reason dropped |
# |---|---|
# | `Booking_ID`, `Customer_ID` | Identifiers |
# | `Booking_Time`, `Appointment_Time` | Raw timestamps — already extracted |
# | `Booking_Lead_Time_Days` | Superseded by `Lead_Time_Days` |
# | `Outcome`, `No_Show` | Target columns |
# | `Past_Visit_Count`, `Past_No_Show_Count`, `Past_Cancellation_Count` | Superseded by ratio features + `Is_New_Customer` + `Successful_Visit_Rate` |

# In[13]:


drop_cols = [
    'Booking_ID', 'Customer_ID',
    'Booking_Time', 'Appointment_Time',
    'Booking_Lead_Time_Days',
    'Outcome', 'No_Show',
    # Raw counts superseded by ratio + new features
    'Past_Visit_Count', 'Past_No_Show_Count', 'Past_Cancellation_Count',
]

X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded['No_Show']

print(f'Feature matrix X : {X.shape[0]:,} rows x {X.shape[1]} columns')
print(f'Target vector  y : {y.shape[0]:,} rows  |  No-show rate: {y.mean():.2%}')
print(f'\nAll feature columns:')
print(X.columns.tolist())


# ## Cell 14 – Preview X

# In[14]:


print('First 5 rows of feature matrix X:')
display(X.head())
print(f'\nStatistical summary of engineered features:')
eng_features = ['Lead_Time_Days','Appt_Hour','Customer_No_Show_Rate',
                'Cancellation_Ratio','Is_New_Customer',
                'Successful_Visit_Rate','Is_Edge_Hour']
display(X[eng_features].describe().round(4))


# ## Cell 15 – Save Feature Matrix and Target

# In[15]:


df_encoded.to_csv('../data/processed/salon_bookings_featured.csv', index=False)
X.to_csv('../data/processed/X.csv', index=False)
y.to_csv('../data/processed/y.csv', index=False, header=True)

print('Saved:')
print(f'  salon_bookings_featured.csv  {df_encoded.shape}')
print(f'  X.csv  {X.shape}')
print(f'  y.csv  {y.shape}')
print(f'\nNew features added: Is_New_Customer, Successful_Visit_Rate, Is_Edge_Hour')

