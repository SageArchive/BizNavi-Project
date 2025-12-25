import pandas as pd
import os
from prophet import Prophet
import logging

# Suppress Prophet logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "data", "Amazon Sale Report.csv")


def forecast_demand(category: str, days: int = 30) -> str:
    """
    Predicts future sales quantity for a specific category using Facebook Prophet.
    Args:
        category (str): The product category (e.g., 'Kurta', 'Set').
        days (int): Number of days to forecast into the future.
    """
    try:
        # 1. Load Data
        if not os.path.exists(DATA_FILE):
            return "Error: Data file not found."

        df = pd.read_csv(DATA_FILE, low_memory=False)

        # 2. Preprocessing
        # Convert Date and Filter by Category
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Case-insensitive filtering
        target_df = df[df['Category'].str.lower() == category.lower()].copy()

        if target_df.empty:
            return f"Error: No data found for category '{category}'."

        # Group by Date to get Daily Quantity (Prophet expects 'ds' and 'y' columns)
        # We predict Quantity (Qty) for inventory planning
        daily_sales = target_df.groupby('Date')['Qty'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']

        # Minimum data check (Prophet needs at least a few data points)
        if len(daily_sales) < 10:
            return f"Not enough data points to forecast for '{category}'. Need at least 10 days of history."

        # 3. Model Training
        # yearly_seasonality=True helps if you have >1 year of data.
        # Since this dataset is short (Apr-Jun), we rely on weekly/daily trends.
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(daily_sales)

        # 4. Prediction
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)

        # 5. Extract Results
        # Get only the future part
        future_forecast = forecast.tail(days)
        total_predicted_qty = future_forecast['yhat'].sum()
        avg_daily_qty = future_forecast['yhat'].mean()

        # Optional: Save plot (Agents can't display images, but can save them)
        # fig = m.plot(forecast)
        # fig.savefig(f"{category}_forecast.png")

        return (
            f"📈 **Forecast Report for '{category}' (Next {days} days)**\n"
            f"- Predicted Total Sales Quantity: {int(total_predicted_qty)} units\n"
            f"- Average Daily Sales: {avg_daily_qty:.1f} units/day\n"
            f"- Insight: Based on historical trends, prepare inventory for approx {int(total_predicted_qty)} units."
        )

    except Exception as e:
        return f"Error during forecasting: {str(e)}"


# Test execution
if __name__ == "__main__":
    print(forecast_demand("Kurta", 30))