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