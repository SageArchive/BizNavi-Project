from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Import our custom logic
from src.agents.analytics_agent import analyze_sales_data
from src.rag.retriever import query_warehouse_policy
from src.tools.forecasting import forecast_demand
from src.tools.visualization import create_sales_chart

# 1. Define Tools (Decorators make them compatible with LangChain)
@tool
def sales_tool(query: str):
    """Useful for quantitative questions about sales, revenue, orders, categories, or dates from the Amazon Sale Report."""
    return analyze_sales_data(query)


@tool
def policy_tool(query: str):
    """Useful for questions about warehouse rules, SOPs, KPIs, packaging guidelines, or fees."""
    return query_warehouse_policy(query)

@tool
def forecasting_tool(category: str):
    """
    Useful ONLY when the user asks for 'prediction', 'forecast', 'future sales', or 'next month demand'.
    Input should be the Category name (e.g., 'Kurta', 'Set', 'Western Dress').
    """
    return forecast_demand(category)

@tool
def visualization_tool(query: str):
    """
    Useful when the user asks to 'visualize', 'plot', 'draw a chart', or 'show graph'.
    Input string should be the column to group by (e.g., 'Category', 'Status', 'Size').
    """
    return create_sales_chart(query)

tools = [sales_tool, policy_tool, forecasting_tool, visualization_tool]


# 2. Setup the Main Orchestrator Agent
def get_orchestrator_agent():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the 'E-Commerce Operation Assistant'. You help e-commerce managers.\n"
         "You have 3 tools:\n"
         "1. Sales Tool: For PAST data analysis (revenue, counts).\n"
         "2. Policy Tool: For warehouse rules and FAQs.\n"
         "3. Forecasting Tool: For FUTURE demand prediction.\n"
         "If the user asks about the future (e.g., 'predict', 'next month'), use the Forecasting Tool.\n"
         "Do not guess. Use the tools."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # You will see the agent deciding which tool to use
        handle_parsing_errors=True
    )

    return agent_executor