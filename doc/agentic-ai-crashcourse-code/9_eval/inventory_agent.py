from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
# llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


@tool
def inventory_tool(product_name: str) -> str:
    """Check inventory availability for a given product name."""
    print(f"TOOL CALLED for: {product_name}")
    inventory = {
        "iPhone 15": "In Stock: Available Items = 2",
        "AirPods Pro": "Out of Stock: Available Items = 0",
        "MacBook Air M3": "In Stock: Available Items = 5"
    }
    return inventory.get(product_name, "Product not found in inventory.")

# Inventory Agent
agent = create_agent(
    model=llm,
    tools=[inventory_tool],
    system_prompt="""
    You are an inventory assistant.

    - If a question is out of scope which is not related to inventory then just say "Sorry, I can’t assist with that." No need to say anything else.
    
    When a user asks about a product, use the inventory_tool to fetch the inventory data.

    - Always call the inventory_tool with the full product name.
    - inventory_tool will return a dictionary which you need to parse to extract stock status, inventory items etc.
    - Respond with clear, concise information including:
        1. The stock status (e.g., "In Stock", "Out of Stock")
        2. The number of available items (if applicable)
    - If the product is not found, say: "The product is not available in our inventory."

    Never guess or hallucinate information. Do not respond unless the inventory_tool is called.
    Keep your response short and informative.
    """
)

def run(question: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    # question = "Do you have any airpods pro in stock?"
    # question = "I want to travel to moon, how much will it cost?"
    question = "What is the inventory status of iPhone 15?"
    print(run(question))
