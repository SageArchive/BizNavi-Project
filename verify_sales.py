import pandas as pd
import os

# --- Configuration ---
# Define the path to the data file relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "Amazon Sale Report.csv")


def verify_sales_data():
    """
    Independent verification script to calculate sales figures using raw Pandas.
    This bypasses the LLM to provide ground truth data.
    """
    print("--- 🔍 Starting Data Verification Process ---")

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: File not found at {DATA_FILE}")
        return

    print(f"Loading data from: {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
    except Exception as e:
        print(f"Critical Error: Failed to load CSV. {e}")
        return

    # 2. Data Preprocessing (Must match the Agent's logic)
    # Convert 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Convert 'Amount' column to numeric, replacing errors/NaNs with 0
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    # 3. Apply Filters
    # Filter A: Date is in April (Month = 4)
    mask_april = df['Date'].dt.month == 4

    # Filter B: Category is 'Kurta' (Case-insensitive comparison)
    mask_kurta = df['Category'].str.upper() == 'KURTA'

    # Create a subset dataframe with applied filters
    target_df = df[mask_april & mask_kurta]

    if target_df.empty:
        print("Warning: No records found for Category 'Kurta' in April.")
        return

    # 4. Calculate Metrics
    # Metric 1: Gross Revenue (Sum of 'Amount' for ALL matching records, including Cancelled)
    gross_revenue = target_df['Amount'].sum()

    # Metric 2: Net Revenue (Exclude rows where Status contains 'Cancelled')
    # We filter out rows where Status is 'Cancelled'
    valid_orders_df = target_df[~target_df['Status'].astype(str).str.contains('Cancelled', case=False, na=False)]
    net_revenue = valid_orders_df['Amount'].sum()

    # 5. Display Results
    print("\n" + "=" * 60)
    print("📊 [Verification Results] Sales Analysis for 'Kurta' in April")
    print("=" * 60)
    print(f"Total Records Found      : {len(target_df)}")
    print(f"Valid Records (No Cancel): {len(valid_orders_df)}")
    print("-" * 60)
    print(f"1. Gross Revenue (Including Cancelled) : INR {gross_revenue:,.2f}")
    print(f"2. Net Revenue   (Excluding Cancelled) : INR {net_revenue:,.2f}")
    print("=" * 60)
    print("\nCompare these figures with your AI Agent's response.")


if __name__ == "__main__":
    verify_sales_data()