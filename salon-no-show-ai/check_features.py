import pandas as pd
df = pd.read_csv('data/raw/salon_bookings.csv')
df['Booking_Time'] = pd.to_datetime(df['Booking_Time'])
df['Appointment_Time'] = pd.to_datetime(df['Appointment_Time'])
df['Engineered_Lead_Time_Days'] = (df['Appointment_Time'] - df['Booking_Time']).dt.days

print("Mismatched lead times:", (df['Booking_Lead_Time_Days'] != df['Engineered_Lead_Time_Days']).sum())
print(df[['Booking_Lead_Time_Days', 'Engineered_Lead_Time_Days']].head(10))
