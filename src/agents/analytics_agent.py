import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_FILE = os.path.join(BASE_DIR, "data", "Amazon Sale Report.csv")
UPLOADED_DATA_FILE = os.path.join(BASE_DIR, "data", "uploaded_data.csv")

def get_sales_data():
    """Loads and preprocesses the data."""
    print("Loading Sales Data...")
    try:
        # Check if uploaded data exists
        if os.path.exists(UPLOADED_DATA_FILE):
            print(f"📂 Loading UPLOADED data from: {UPLOADED_DATA_FILE}")
            df = pd.read_csv(UPLOADED_DATA_FILE, low_memory=False)
        elif os.path.exists(DEFAULT_DATA_FILE):
            print(f"📂 Loading DEFAULT data from: {DEFAULT_DATA_FILE}")
            df = pd.read_csv(DEFAULT_DATA_FILE, low_memory=False)
        else:
            print("❌ No data file found.")
            return pd.DataFrame()

        # Preprocessing
        df.drop_duplicates(['Order ID', 'ASIN'], inplace=True, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        # Ensure Status is string for easier filtering
        df['Status'] = df['Status'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

# Load data once
df = get_sales_data()

def analyze_sales_data(query: str) -> str:
    """
    Analyzes the 'Amazon Sale Report' dataset using Python/Pandas.
    Useful for calculating revenue, counting orders, checking stock/qty, or filtering by category/date.
    """
    if df.empty:
        return "Error: Sales data is not loaded."

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prefix_instructions = """
        You are a Python Data Analyst working with a pandas DataFrame named `df`.
        This dataframe is ALREADY LOADED in your environment.

        CRITICAL RULES:
        1. DO NOT create a new dataframe (e.g., never write `df = pd.DataFrame(...)`). Use the existing `df`.
        2. ALWAYS start your code with `import pandas as pd`.
        3. 'Total Sales' or 'Gross Revenue' = Sum of Amount (Include Cancelled).
        4. 'Net Sales' or 'Real Revenue' = Sum of Amount WHERE Status does NOT contain 'Cancelled'.
        5. If the user does not specify 'Net', assume 'Total/Gross'.
        6. Format the output in INR using INTERNATIONAL standard (e.g., 8,017,145.48). DO NOT use the Indian numbering system (Lakhs/Crores).
        """

    # Create the internal agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="openai-tools",
        agent_executor_kwargs={"handle_parsing_errors": True},
        prefix=prefix_instructions,
    )

    try:
        response = agent.invoke(query)
        return response['output']
    except Exception as e:
        return f"Error during analysis: {str(e)}"