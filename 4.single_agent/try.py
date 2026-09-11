import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

PRODUCTS = {

    "LED Photon Rejuvenation Face Mask": {"Price": 77.99, "Stock": 10, "rating": 4.5, "description": "The LED Photon Rejuvenation Face Mask is an advanced skincare device designed to improve skin health using LED light therapy. It helps rejuvenate the skin, reduce wrinkles, improve skin tone, and support collagen production. The mask features multiple LED light modes that target different skin concerns such as acne, aging, and dull skin. It comes with a remote control for easy operation and a comfortable wearable design for home beauty treatments."},
    "2-in-1 Oil Dispenser with Silicone Brush": {"Price": 150.00 , "Stock": 5, "rating": 4.7, "description": "Upgrade your cooking with the 2-in-1 Oil Dispenser featuring a silicone brush and spray bottle. Perfect for BBQ, grilling, baking, roasting, and everyday cooking. Durable glass design, reusable, and easy to refill. Shop online with delivery across Saudi Arabia."},
    "Automatic Fresh Juicer | Portable USB Citrus Juicer KSA": {"Price": 200.00, "Stock": 3, "rating": 4.7, "description": "Tenkey less, Cherry MX Brown, per key RGB."},
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

st.title("🛒 Nexgen Agent")

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