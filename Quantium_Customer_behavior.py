import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the datasets
# Provide the file paths for the customer and transaction data
file_path1 = 'd:/projects/QVI_purchase_behaviour.csv' 
file_path2 = 'd:/projects/QVI_transaction_data.csv' 

# Read the datasets into pandas DataFrames
data = pd.read_csv(file_path1) 
transactions = pd.read_csv(file_path2)

# Display first few rows of the datasets
print("Customer Data:")
print(data.head())
print("\nTransaction Data:")
print(transactions.head())

# Step 2: Handle Missing Data
# Checking for missing values in both datasets
print("\nMissing Values in Transactions Data:")
print(transactions.isnull().sum())
print("\nMissing Values in Customer Data:")
print(data.isnull().sum())

# Handle missing values by removing rows with NaN values
data_cleaned = data.dropna()
transactions_cleaned = transactions.dropna()

# Alternatively, missing values could be handled by filling NaNs
# data_cleaned = data.fillna(0)  # Or replace with zero
# data_cleaned['column_name'] = data_cleaned['column_name'].fillna(data_cleaned['column_name'].mean())  # Custom replacement

# Step 3: Remove Duplicates
# Check for duplicates in the data
duplicates_data = data_cleaned.duplicated().sum()
duplicates_transactions = transactions_cleaned.duplicated().sum()

print(f"\nDuplicates in Customer Data: {duplicates_data}")
print(f"Duplicates in Transaction Data: {duplicates_transactions}")

# Remove duplicate rows
data_cleaned = data_cleaned.drop_duplicates()
transactions_cleaned = transactions_cleaned.drop_duplicates()

# Step 4: Data Type Conversion
# Convert relevant columns to proper data types
transactions_cleaned['DATE'] = pd.to_datetime(transactions_cleaned['DATE'], errors='coerce')  # Convert DATE column to datetime
data_cleaned['LIFESTAGE'] = data_cleaned['LIFESTAGE'].astype('category')  # Convert LIFESTAGE to category
data_cleaned['PREMIUM_CUSTOMER'] = data_cleaned['PREMIUM_CUSTOMER'].astype('category')  # Convert PREMIUM_CUSTOMER to category

# Step 5: Merge Customer and Transaction Data
# Merge cleaned customer data with transaction data based on loyalty card number
merged_data = pd.merge(data_cleaned, transactions_cleaned, on='LYLTY_CARD_NBR', how='left')

# Display first few rows of the merged data
print("\nMerged Data:")
print(merged_data.head())
print("\nMerged Data Shape:", merged_data.shape)

# Step 6: Exploratory Data Analysis (EDA)
# Explore the distribution of premium customers
premium_counts = merged_data['PREMIUM_CUSTOMER'].value_counts()
print("\nPremium Customer Distribution:")
print(premium_counts)

# Sales analysis by Lifestage and Customer Type
# Total sales by lifestage
sales_by_lifestage = merged_data.groupby('LIFESTAGE')['TOT_SALES'].sum()
print("\nTotal Sales by Lifestage:")
print(sales_by_lifestage.sort_values(ascending=False))

# Total sales by premium customer status
sales_by_premium = merged_data.groupby('PREMIUM_CUSTOMER')['TOT_SALES'].sum()
print("\nTotal Sales by Premium Customer Type:")
print(sales_by_premium.sort_values())

# Step 7: Data Visualization
# Plot total sales by lifestage
sales_by_lifestage.plot(kind='bar', title='Total Sales by Lifestage')
plt.xlabel('Lifestage')
plt.ylabel('Total Sales')
plt.show()

# Plot total sales by premium customer
sales_by_premium.plot(kind='bar', title='Total Sales by Premium Customer')
plt.xlabel('Customer Type')
plt.ylabel('Total Sales')
plt.show()

# Step 8: Focus on High-Value Segments (e.g., OLDER SINGLES/COUPLES & Mainstream Customers)
high_value_segments = merged_data[(merged_data['LIFESTAGE'] == 'OLDER SINGLES/COUPLES') & 
                                  (merged_data['PREMIUM_CUSTOMER'] == 'Mainstream')]

# Calculate total sales for high-value segments
total_sales_high_value = high_value_segments['TOT_SALES'].sum()
print(f"\nTotal Sales for High-Value Segments: {total_sales_high_value}")

# Calculate average sales per transaction for high-value segments
average_sales_high_value = high_value_segments['TOT_SALES'].mean()
print(f"Average Sales per Transaction for High-Value Segments: {average_sales_high_value}")

# Calculate purchase frequency for high-value segments
purchase_frequency_high_value = high_value_segments['LYLTY_CARD_NBR'].count() / len(high_value_segments['LYLTY_CARD_NBR'].unique())
print(f"Average Purchase Frequency for High-Value Segments: {purchase_frequency_high_value}")

# Step 9: Handling Julian Dates
# Convert Julian dates to datetime using '1900-01-01' as the base date
julian_dates = merged_data['DATE']  # Assuming this column is in Julian format
base_date = pd.Timestamp('1900-01-01')

# Convert Julian days to datetime
if isinstance(julian_dates, pd.Series) and julian_dates.dtype == 'datetime64[ns]':
    merged_data['DATE'] = julian_dates  # If already in datetime format
else:
    merged_data['DATE'] = base_date + pd.to_timedelta(julian_dates, unit='D')

# Display the first few rows after conversion
print("\nUpdated Date Format (Julian to Datetime):")
print(merged_data['DATE'].head())

# Step 10: Grouping and Summarizing Sales
# Group by lifestage and summarize sales
sales_by_lifestage = merged_data.groupby('LIFESTAGE')['TOT_SALES'].sum()

# Group by premium customer type and summarize sales
sales_by_premium = merged_data.groupby('PREMIUM_CUSTOMER')['TOT_SALES'].sum()

# Calculate total sales
total_sales = merged_data['TOT_SALES'].sum()

# Filter for high-value premium segments and calculate sales metrics
high_value_segments = merged_data[merged_data['PREMIUM_CUSTOMER'] == 'Premium']
total_sales_high_value = high_value_segments['TOT_SALES'].sum()
avg_sales_per_transaction = high_value_segments['TOT_SALES'].mean()
avg_purchase_frequency = high_value_segments['TOT_SALES'].count() / len(high_value_segments['LYLTY_CARD_NBR'].unique())

# Print the results
print(f"\nTotal Sales by Lifestage: {sales_by_lifestage.sum()}")
print(f"Total Sales by Premium Customer: {sales_by_premium.sum()}")
print(f"Total Sales for High-Value Segments: {total_sales_high_value}")
print(f"Average Sales per Transaction for High-Value Segments: {avg_sales_per_transaction}")
print(f"Average Purchase Frequency for High-Value Segments: {avg_purchase_frequency}")

# Step 11: Group Sales by Day of the Week
# Ensure 'DATE' column is in numeric format
merged_data['DATE'] = pd.to_numeric(merged_data['DATE'], errors='coerce')

# Now convert it to datetime using the correct origin
merged_data['DATE'] = pd.to_datetime(merged_data['DATE'], unit='D', origin='1900-01-01')

# Define the correct order for days of the week
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Group sales by day of the week
merged_data['DAY_OF_WEEK'] = merged_data['DATE'].dt.day_name()
sales_by_day = merged_data.groupby('DAY_OF_WEEK')['TOT_SALES'].sum().reindex(days_of_week)

# Visualize total sales by day of the week
sales_by_day.plot(kind='bar', title='Total Sales by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.show()

# Step 12: Group Sales by Month
# Extract the month name from the DATE column
merged_data['MONTH'] = merged_data['DATE'].dt.month_name()

# Group by month and summarize sales
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sales_by_month = merged_data.groupby('MONTH')['TOT_SALES'].sum().reindex(months)

# Visualize total sales by month
sales_by_month.plot(kind='bar', title='Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# Step 13: Verify Data and Check Date Range
print("\nDate Range in Dataset:")
print(merged_data['DATE'].min(), merged_data['DATE'].max())
print("\nUnique Dates:")
print(merged_data['DATE'].unique())
