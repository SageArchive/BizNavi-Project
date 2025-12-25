import pandas as pd
import plotly.express as px
import json
import os
import plotly.utils

# --- Configuration ---
# Define paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOADED_DATA_FILE = os.path.join(BASE_DIR, "data", "uploaded_data.csv")
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "data", "Amazon Sale Report.csv")
PLOT_FILE = os.path.join(BASE_DIR, "data", "latest_plot.json")


def create_sales_chart(group_by: str, metric: str = "Amount") -> str:
    """
    Generates a bar chart based on the category, status, or other grouping columns.

    Args:
        group_by (str): Column to group by (e.g., 'Category', 'Status', 'Size').
        metric (str): Column to sum (e.g., 'Amount', 'Qty').

    Returns:
        str: A status message used by the UI to trigger rendering ("CHART_GENERATED").
    """
    try:
        # 1. Determine which file to load (Uploaded vs Default)
        path = UPLOADED_DATA_FILE if os.path.exists(UPLOADED_DATA_FILE) else DEFAULT_DATA_FILE

        if not os.path.exists(path):
            return "Error: No data file found."

        df = pd.read_csv(path, low_memory=False)

        # 2. Handle Case-Insensitive Column Matching
        # Create a map of {lowercase_name: Actual_Name}
        col_map = {c.lower(): c for c in df.columns}

        group_col = col_map.get(group_by.lower())
        metric_col = col_map.get(metric.lower())

        if not group_col or not metric_col:
            return f"Error: Columns '{group_by}' or '{metric}' not found in the dataset."

        # 3. Aggregate Data
        # Group by the target column and sum the metric
        chart_data = df.groupby(group_col)[metric_col].sum().reset_index()

        # Sort by value and take the Top 10 for readability
        chart_data = chart_data.sort_values(by=metric_col, ascending=False).head(10)

        # 4. Generate Chart using Plotly Express
        fig = px.bar(
            chart_data,
            x=group_col,
            y=metric_col,
            title=f"Total {metric_col} by {group_col} (Top 10)",
            color=metric_col,
            template="plotly_white",
            text_auto='.2s'  # Format numbers on bars
        )

        # 5. Save Chart to JSON
        # We save the figure as JSON so Streamlit can read and render it later
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(PLOT_FILE), exist_ok=True)

        with open(PLOT_FILE, "w") as f:
            f.write(plot_json)

        # Return a specific keyword that the UI (app.py) listens for
        return "CHART_GENERATED"

    except Exception as e:
        return f"Error creating chart: {str(e)}"