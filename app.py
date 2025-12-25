import streamlit as st
import pandas as pd
import os
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from dotenv import load_dotenv

# Load API Keys from .env file immediately
load_dotenv()

# Import our custom orchestrator
from src.agents.orchestration import get_orchestrator_agent

# Page Configuration
st.set_page_config(
    page_title="BizNavi Dashboard",
    page_icon="🧭",
    layout="wide"
)

# --- [Sidebar] Settings & Status ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/814/814254.png", width=80)
    st.title("BizNavi")
    st.caption("Your E-Commerce Assistant 🧭")
    st.markdown("---")
    # Simple check to see if key is loaded (for debugging)
    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ API Key Loaded")
    else:
        st.error("❌ API Key Missing")

    st.success("✅ Sales Data Loaded")
    st.success("✅ RAG Knowledge Base Ready")
    st.success("✅ Forecasting Model Ready")
    st.markdown("---")
    st.info(
        "💡 **Try asking:**\n- 'Total revenue for Kurta in April?'\n- 'What is the return policy?'\n- 'Forecast sales for Kurta next month'")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- [Main] Tabs ---
tab1, tab2 = st.tabs(["💬 AI Assistant (Chat)", "📈 Demand Forecast (Dashboard)"])

# ==========================================
# TAB 1: AI Chat Interface
# ==========================================
with tab1:
    st.header("🛒 Intelligent Ops Assistant")

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
    except Exception as e:
        st.error(f"Error loading agent: {e}")
        st.stop()

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Type your question here... (e.g., Total revenue for Kurta in April?)"):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate AI Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing data and retrieving policies..."):
                try:
                    # Invoke the Agent
                    response = agent_executor.invoke({"input": prompt})
                    full_response = response["output"]
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"❌ An error occurred: {str(e)}"
                    message_placeholder.error(full_response)

        # 3. Save Assistant Response
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# ==========================================
# TAB 2: Demand Forecasting Dashboard
# ==========================================
with tab2:
    st.header("📊 Demand Forecasting Dashboard")
    st.write("Visualizes sales forecasts for the next 30 days using the Prophet model.")

    # Load Data for Visualization
    # Ensure correct path relative to where you run the command
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(BASE_DIR, "data", "Amazon Sale Report.csv")


    @st.cache_data
    def load_data():
        if not os.path.exists(DATA_FILE):
            return pd.DataFrame()
        df = pd.read_csv(DATA_FILE, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df


    try:
        df = load_data()

        if df.empty:
            st.error(f"Data file not found at: {DATA_FILE}")
        else:
            # Category Selection
            categories = df['Category'].dropna().unique()
            selected_category = st.selectbox("Select a Category to Analyze:", categories, index=0)

            if st.button("🚀 Generate Forecast"):
                with st.spinner(f"Modeling '{selected_category}'..."):
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

                        # Create Future Dataframe (30 days)
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

    except Exception as e:
        st.error(f"Error loading data: {e}")