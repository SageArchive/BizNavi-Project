import streamlit as st
import pandas as pd
import os
import json
import plotly.io as pio
import re
from prophet import Prophet
from prophet.plot import plot_plotly
from dotenv import load_dotenv

# --- API Key Setup ---
load_dotenv()

from src.agents.orchestration import get_orchestrator_agent

# --- App Configuration ---
st.set_page_config(page_title="BizNavi Dashboard", page_icon="🧭", layout="wide")

# [CRITICAL] Standardize paths based on current working directory (os.getcwd)
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

UPLOADED_DATA_FILE = os.path.join(DATA_DIR, "uploaded_data.csv")
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "Amazon Sale Report.csv")
PLOT_FILE = os.path.join(DATA_DIR, "latest_plot.json")


# --- Helper: Data Loader ---
@st.cache_data
def load_data():
    """Loads sales data from uploaded file or default file."""
    if os.path.exists(UPLOADED_DATA_FILE):
        return pd.read_csv(UPLOADED_DATA_FILE, low_memory=False), "Uploaded Data"
    elif os.path.exists(DEFAULT_DATA_FILE):
        return pd.read_csv(DEFAULT_DATA_FILE, low_memory=False), "Default Data"
    return pd.DataFrame(), "No Data Available"


# ==========================================
# SIDEBAR: Settings & Upload
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/814/814254.png", width=80)
    st.title("BizNavi")
    st.caption("AI Operations Assistant 🧭")

    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload Monthly Sales CSV", type=["csv"])

    if uploaded_file:
        # Save uploaded file
        with open(UPLOADED_DATA_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Uploaded Successfully!")
        # Clear cache to reload new data
        st.cache_data.clear()
        st.rerun()

    # Show current data source
    _, src = load_data()
    st.info(f"Using: {src}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        # Remove old plot file
        if os.path.exists(PLOT_FILE):
            os.remove(PLOT_FILE)
        st.rerun()

# --- Main Tabs ---
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📈 Forecasting Dashboard"])

# ==========================================
# TAB 1: Chat Interface
# ==========================================
with tab1:
    st.header("🧭 Operations Copilot")

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Load Agent (Cached)
    @st.cache_resource
    def load_agent():
        return get_orchestrator_agent()


    try:
        agent = load_agent()
    except Exception as e:
        st.error("❌ Error loading agent. Please check your API Key.")
        st.stop()

    # 1. Display Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict): content = str(content)
            st.markdown(content)

            if "chart_data" in msg:
                try:
                    st.plotly_chart(pio.from_json(msg["chart_data"]), use_container_width=True, key=f"chart_{i}")
                except:
                    pass

    # 2. Handle User Input
    if prompt := st.chat_input("Ask about sales, policies, or visualize data (e.g., 'Visualize sales by Category')"):
        # (A) Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # (B) Generate Assistant Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Analyzing data & Thinking..."):
                try:
                    # Remove old plot file before starting
                    if os.path.exists(PLOT_FILE):
                        os.remove(PLOT_FILE)

                    # Invoke Agent
                    response = agent.invoke({"input": prompt})

                    # Handle Dict vs String output
                    raw_output = response.get("output", "No response")
                    if isinstance(raw_output, dict):
                        resp = json.dumps(raw_output, ensure_ascii=False)
                    else:
                        resp = str(raw_output)

                    # Remove hallucinated image tags (e.g., ![image](...))
                    resp = re.sub(r'!\[[^\]]*\]\([^\)]*\)', '', resp)

                    # Detect Chart Generation Signal
                    chart_json = None

                    # Read the generated JSON file
                    if os.path.exists(PLOT_FILE):
                        with open(PLOT_FILE, "r") as f:
                            chart_json = f.read()
                        resp += "\n\n📊 **Visualization Generated:**"


                    # Display Text Response
                    placeholder.markdown(resp)

                    # Display Chart (Immediate Render)
                    if chart_json:
                        st.plotly_chart(pio.from_json(chart_json), use_container_width=True)

                except Exception as e:
                    resp = f"❌ An error occurred: {str(e)}"
                    placeholder.error(resp)
                    chart_json = None

        # (C) Save & Rerun (Fixes Layout)
        msg_data = {"role": "assistant", "content": resp}
        if chart_json:
            msg_data["chart_data"] = chart_json

        st.session_state.messages.append(msg_data)
        st.rerun()

# ==========================================
# TAB 2: Forecasting Dashboard
# ==========================================
with tab2:
    st.header("📈 Demand Forecasting Radar")
    st.write("Predict future sales trends using the Prophet AI model.")

    df, _ = load_data()

    if df.empty:
        st.warning("⚠️ No data available for forecasting.")
    else:
        # Preprocessing
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # UI Controls
        col1, col2 = st.columns([1, 3])
        with col1:
            categories = df['Category'].dropna().unique()
            selected_cat = st.selectbox("Select Category:", categories)
            run_btn = st.button("🚀 Run Forecast")

        # Run Model
        if run_btn:
            with st.spinner(f"Training model for '{selected_cat}'..."):
                # Prepare data for Prophet
                target_df = df[df['Category'] == selected_cat].copy()
                daily_sales = target_df.groupby('Date')['Qty'].sum().reset_index()
                daily_sales.columns = ['ds', 'y']

                if len(daily_sales) < 5:
                    st.error("❌ Not enough historical data to forecast this category.")
                else:
                    try:
                        # Fit Model
                        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                        m.fit(daily_sales)

                        # Predict
                        future = m.make_future_dataframe(periods=30)
                        forecast = m.predict(future)

                        # Visuals
                        st.subheader(f"30-Day Forecast: {selected_cat}")
                        fig = plot_plotly(m, forecast)
                        st.plotly_chart(fig, use_container_width=True)

                        # Metrics
                        future_30 = forecast.tail(30)
                        total_pred = int(future_30['yhat'].sum())
                        st.success(f"🤖 Predicted Total Sales (Next 30 Days): **{total_pred} units**")

                    except Exception as e:
                        st.error(f"Modeling Error: {e}")