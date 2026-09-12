import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()


STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN")
STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_PUBLIC_TOKEN")

def fetch_products():
    """Fetch live products from Shopify Storefront API."""
    url = f"https://{STORE_DOMAIN}/api/2024-10/graphql.json"
    query = """
    {
      products(first: 50) {
        edges {
          node {
            title
            description
            priceRange {
              minVariantPrice {
                amount
                currencyCode
              }
            }
            totalInventory
            rating: metafield(namespace: "custom", key: "rating") {
              value
            }
          }
        }
      }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }
    response = requests.post(url, json={"query": query}, headers=headers)
    data = response.json()
    products = {}
    for edge in data["data"]["products"]["edges"]:
        node = edge["node"]
        name = node["title"]
        price = node["priceRange"]["minVariantPrice"]["amount"]
        currency = node["priceRange"]["minVariantPrice"]["currencyCode"]
        rating = node["rating"]["value"] if node["rating"] else "Not rated yet"
        products[name] = {
            "Price": price,
            "Currency": currency,
            "Stock": node["totalInventory"],
            "Rating": rating,
            "description": node["description"],
        }
    return products

@st.cache_data(ttl=300)  # refresh every 5 minutes
def get_products():
    return fetch_products()

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price and description."""
    products = get_products()
    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(name.lower())
    if not p:
        return f"product not found. Available: {', '.join(products)}"
    return str(p)

@tool
def list_products() -> str:
    """List all available products with their prices, stock and rating."""
    products = get_products()
    lines = [
        f"{name}: {info['Price']} {info['Currency']} | Stock: {info['Stock']} | Rating: {info['Rating']}"
        for name, info in products.items()
    ]
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

st.set_page_config(page_title="Nexgen Assistant", page_icon="🛒")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stChatMessage {
        border-radius: 14px;
        padding: 4px 10px;
    }
    div[data-testid="stChatMessageContent"] {
        font-size: 14px;
    }
    .welcome-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 14px;
        text-align: center;
    }
    .welcome-box h3 {
        margin: 0 0 4px 0;
        font-size: 17px;
    }
    .welcome-box p {
        margin: 0;
        font-size: 13px;
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-box">
    <h3>🛒 Nexgen Assistant</h3>
    <p>Ask me about prices, stock, or ratings — I'm here to help!</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    avatar = "🧑" if msg["role"] == "user" else "🛒"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

question = st.chat_input("Ask about our products...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.write(question)

    with st.spinner("Thinking..."):
        result = agent.invoke({"messages": st.session_state.history})
        reply = result["messages"][-1].content

    st.session_state.history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🛒"):
        st.write(reply)