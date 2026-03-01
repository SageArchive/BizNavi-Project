from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
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
    """
    Useful for questions about warehouse rules, SOPs, KPIs, packaging guidelines, or fees.
    Input MUST be ONLY the exact core noun/topic (e.g., 'Allowed Shrinkage', 'Outbound', 'Penalty', 'Customer Complaints').
    CRITICAL: DO NOT include words like 'limit', 'price', 'fee', 'what is', or 'policy' in the query.
    Just the exact topic name.
    """
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
    llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are the 'E-Commerce Operation Assistant'. You help e-commerce managers.\n"
         "You have 4 tools: Sales, Policy, Forecasting, and Visualization.\n\n"
         "TOOL USAGE GUIDELINES:\n"
         "1. Sales Tool: Use for PAST data analysis (e.g., revenue, counts, specific dates/categories).\n"
         "2. Policy Tool: Use for warehouse rules, SOPs, KPIs, pricing, and limits.\n"
         "   -> CRITICAL: When using the Policy Tool, you MUST pass the FULL user question as the input (e.g., 'What is the allowed shrinkage limit?'). Do NOT pass just single keywords.\n"
         "3. Forecasting Tool: Use for FUTURE demand prediction (e.g., 'predict', 'forecast', 'next month').\n"
         "4. Visualization Tool: Use when the user asks to 'visualize', 'plot', or 'draw a chart'.\n\n"
         "CRITICAL RULES FOR READING POLICY TOOL OUTPUT:\n"
         "- The tool returns multiple numbered documents (e.g., [Document 1], [Document 2]).\n"
         "- Find the ONE document that exactly matches the user's intent based on the 'Section' and 'Topic'.\n"
         "- DO NOT mix or combine information from different documents.\n"
         "- Extract the exact value or description from the matching document only. Do not guess or hallucinate."
         ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # You will see the agent deciding which tool to use
        handle_parsing_errors=True
    )

    return agent_executor