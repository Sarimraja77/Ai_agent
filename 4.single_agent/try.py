import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

PRODUCTS = {
    "Wireless": {"Price": 79.99, "Stock": 10, "rating": 4.5, "description": "Over-ear Bluetooth, 30-hr Battery, active noise cancellation."},
    "smart watch": {"Price": 199.99, "Stock": 5, "rating": 4.2, "description": "Tracks heart rate and sleep. 5-Day battery, water resistant."},
    "Gaming Keyboard": {"Price": 200.00, "Stock": 3, "rating": 4.7, "description": "Tenkey less, Cherry MX Brown, per key RGB."},
    "Speakers": {"Price": 249.99, "Stock": 7, "rating": 4.6, "description": "Best Audionic Speaker, Bass-Boost Sound, RGB-lights."},
    "Mosquito Killer": {"Price": 49.99, "Stock": 15, "rating": 4.9, "description": "UV light mosquito killer, 20m² coverage, quiet operation."},
    "Electric Tyre Pump": {"Price": 149.99, "Stock": 6, "rating": 4.5, "description": "Portable Electric Tire Pump for cars, SUVs and motorcycles. Quickly inflate your tires with a convenient compact air compressor — ideal for emergencies, road trips and everyday driving in Saudi Arabia."},
}
PRODUCTS_LOOKUP = {k.lower(): v for k, v in PRODUCTS.items()}

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, rating, stock and description."""
    p = PRODUCTS_LOOKUP.get(name.lower())
    if not p:
        return f"product not found. Available: {', '.join(PRODUCTS)}"
    return str(p)

@tool
def list_products() -> str:
    """List all available products with their prices."""
    lines = [f"{name}: ${info['Price']}" for name, info in PRODUCTS.items()]
    return "\n".join(lines)

@st.cache_resource
def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return create_agent(
        llm,
        tools=[get_product, list_products],
        system_prompt=(
            "You are a product assistant for an online tech store. "
            "Always call the get_product tool with the user's best-guess product name — "
            "do not ask the user to confirm the name before calling the tool. "
            "If the user asks to see all products, or the full catalog, call the list_products tool. "
            "Only ask for clarification if the tool returns a 'not found' result."
        ),
    )

agent = get_agent()

st.title("🛒 Tech Store Product Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask about our products...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = agent.invoke({"messages": st.session_state.history})
    reply = result["messages"][-1].content
    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)