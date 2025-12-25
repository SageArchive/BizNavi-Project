import os
from dotenv import load_dotenv
from src.agents.orchestration import get_orchestrator_agent

# Load API Keys
load_dotenv()


def main():
    print("Initializing E-Commerce Operation Platform...")

    # Check if Vector DB exists, warn if not
    if not os.path.exists("chroma_db"):
        print("WARNING: 'chroma_db' not found. RAG queries might fail.")
        print("Run 'python src/rag/vector_store.py' to build the database.")

    agent_executor = get_orchestrator_agent()

    print("\n✅ Your Assistant is Ready! (Type 'quit' to exit)")
    print("-------------------------------------------------------")
    print("Try asking: 'What was the total revenue for Kurta in April?'")
    print("Or: 'What is the allowed shrinkage limit?'")
    print("-------------------------------------------------------")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        try:
            # The agent decides which tool to use internally
            result = agent_executor.invoke({"input": user_input})
            print(f"\nAssistant: {result['output']}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()