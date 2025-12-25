import streamlit as st
import pandas as pd
import os
import json
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.io as pio
import plotly.graph_objs as go
from dotenv import load_dotenv

# API Key Setup
load_dotenv()

# Import our custom orchestrator
from src.agents.orchestration import get_orchestrator_agent

# Page Configuration
st.set_page_config(
    page_title="BizNavi Dashboard",
    page_icon="🧭",
    layout="wide"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

UPLOADED_DATA_FILE = os.path.join(DATA_DIR, "uploaded_data.csv")
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "Amazon Sale Report.csv")
PLOT_FILE = os.path.join(DATA_DIR, "latest_plot.json")

# --- Helper: Data Loader ---
@st.cache_data
def load_data():
    if os.path.exists(UPLOADED_DATA_FILE):
        df = pd.read_csv(UPLOADED_DATA_FILE, low_memory=False)
        source = "Uploaded Data"
    elif os.path.exists(DEFAULT_DATA_FILE):
        df = pd.read_csv(DEFAULT_DATA_FILE, low_memory=False)
        source = "Default Data"
    else:
        return pd.DataFrame(), "No Data"

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df, source


# --- [Sidebar] Settings & Status ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/814/814254.png", width=80)
    st.title("BizNavi")
    st.caption("Your E-Commerce Assistant 🧭")

    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        with open(UPLOADED_DATA_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ New Data Uploaded!")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    # Current data in use
    _, data_source = load_data()
    st.caption(f"Using: {data_source}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if os.path.exists(PLOT_FILE):
            os.remove(PLOT_FILE)
        st.rerun()

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 AI Assistant (Chat)", "📈 Sales Forecast (Dashboard)"])

# ==========================================
# TAB 1: AI Chat Interface
# ==========================================
with tab1:
    st.header("🧭 Operations Copilot")

    # Initialize Session State for Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load Agent (Cached for performance)
    @st.cache_resource
    def load_agent():
        # This function calls the orchestrator which needs the API key
        return get_orchestrator_agent()

    try:
        agent_executor = load_agent()
    except:
        st.error("API Key Error")
        st.stop()

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart_data" in message:
                try:
                    fig = pio.from_json(message["chart_data"])
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

    # Handle User Input
    if prompt := st.chat_input("Type your question here... (e.g., Total revenue for Kurta in April? or Visualize sales by Category)"):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing data and retrieving policies..."):
                try:
                    if os.path.exists(PLOT_FILE): os.remove(PLOT_FILE)

                    response = agent_executor.invoke({"input": prompt})
                    full_response = response["output"]

                    chart_json = None
                    if "CHART_GENERATED" in full_response:
                        full_response = full_response.replace("CHART_GENERATED", "📊 Visualization:")
                        if os.path.exists(PLOT_FILE):
                            with open(PLOT_FILE, "r") as f: chart_json = f.read()

                    message_placeholder.markdown(full_response)
                    if chart_json:
                        fig = pio.from_json(chart_json)
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    full_response = f"Error: {e}"
                    message_placeholder.error(full_response)
                    chart_json = None

        # 3. Save Assistant Response
        msg_data = {"role": "assistant", "content": full_response}
        if chart_json: msg_data["chart_data"] = chart_json
        st.session_state.messages.append(msg_data)
        st.rerun()

# ==========================================
# TAB 2: Forecasting Dashboard
# ==========================================
with tab2:
    st.header("📈 Sales Forecasting")
    st.write("Select a category to predict future sales for the next 30 days using the Prophet model.")

    # Load Data for Visualization
    df, _ = load_data()

    if df.empty:
        st.warning("No data available.")
    else:
        categories = df['Category'].dropna().unique()
        selected_category = st.selectbox("Select Category:", categories)

        if st.button("🚀 Run Forecast Model"):
            with st.spinner(f"Training model for '{selected_category}'..."):
                # Data Preprocessing for Prophet
                target_df = df[df['Category'] == selected_category].copy()

                # Group by Date (Daily Sum of Qty)
                daily_sales = target_df.groupby('Date')['Qty'].sum().reset_index()
                daily_sales.columns = ['ds', 'y']

                if len(daily_sales) < 5:
                    st.error("Not enough data points to forecast this category.")
                else:
                    # Prophet Modeling
                    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                    m.fit(daily_sales)
                    future = m.make_future_dataframe(periods=30)
                    forecast = m.predict(future)

                    # 1. Plotly Interactive Chart
                    st.subheader(f"📈 30-Day Sales Forecast for {selected_category}")
                    fig = plot_plotly(m, forecast)
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. Key Metrics
                    future_30 = forecast.tail(30)
                    total_pred = future_30['yhat'].sum()
                    avg_daily = future_30['yhat'].mean()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Predicted Sales (30d)", f"{int(total_pred)} units")
                    col2.metric("Avg Daily Sales", f"{avg_daily:.1f} units/day")
                    col3.metric("Training Data Points", f"{len(daily_sales)} days")

                    # Show Raw Data
                    with st.expander("View Detailed Forecast Data"):
                        st.dataframe(future_30[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())