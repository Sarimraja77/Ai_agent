import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN")
STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_PUBLIC_TOKEN")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")
FB_VERIFY_TOKEN = os.getenv("FB_VERIFY_TOKEN")

app = FastAPI()

# ---- Shopify product fetching (same logic as try.py) ----

def fetch_products():
    url = f"https://{STORE_DOMAIN}/api/2024-10/graphql.json"
    query = """
    {
      products(first: 50) {
        edges {
          node {
            title
            description
            priceRange {
              minVariantPrice { amount currencyCode }
            }
            totalInventory
            rating: metafield(namespace: "custom", key: "rating") { value }
            variants(first: 1) {
              edges { node { id } }
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
        variant_id = node["variants"]["edges"][0]["node"]["id"] if node["variants"]["edges"] else None
        products[name] = {
            "Price": price,
            "Currency": currency,
            "Stock": node["totalInventory"],
            "Rating": rating,
            "description": node["description"],
            "variant_id": variant_id,
        }
    return products

_products_cache = {"data": None}

def get_products():
    if _products_cache["data"] is None:
        _products_cache["data"] = fetch_products()
    return _products_cache["data"]

def create_cart(variant_id, quantity=1):
    url = f"https://{STORE_DOMAIN}/api/2024-10/graphql.json"
    mutation = """
    mutation($variantId: ID!, $qty: Int!) {
      cartCreate(input: { lines: [{ merchandiseId: $variantId, quantity: $qty }] }) {
        cart { id checkoutUrl }
        userErrors { field message }
      }
    }
    """
    variables = {"variantId": variant_id, "qty": quantity}
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }
    response = requests.post(url, json={"query": mutation, "variables": variables}, headers=headers)
    return response.json()

# ---- Agent tools ----

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, stock, rating and description."""
    products = get_products()
    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(products)}"
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

@tool
def add_to_cart(product_name: str, quantity: int = 1) -> str:
    """Add a product to the cart by name and return a checkout link."""
    products = get_products()
    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(product_name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(products)}"
    if not p.get("variant_id"):
        return "Sorry, this product cannot be added to cart right now."
    result = create_cart(p["variant_id"], quantity)
    cart_data = result.get("data", {}).get("cartCreate", {})
    errors = cart_data.get("userErrors", [])
    if errors:
        return f"Could not add to cart: {errors[0]['message']}"
    checkout_url = cart_data["cart"]["checkoutUrl"]
    return f"Added {quantity} x {product_name} to cart! Complete your order here: {checkout_url}"

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
agent = create_agent(
    llm,
    tools=[get_product, list_products, add_to_cart],
    system_prompt=(
        "You are a product assistant for an online tech store. "
        "Always call the get_product tool with the user's best-guess product name — "
        "do not ask the user to confirm the name before calling the tool. "
        "If the user asks to see all products, or the full catalog, call the list_products tool. "
        "If the user wants to add a product to their cart or buy it, call the add_to_cart tool "
        "with the exact product name. Always share the checkout link you get back. "
        "Only ask for clarification if a tool returns a 'not found' result. "
        "Keep replies short and conversational, suitable for a Messenger chat."
    ),
)

# per-user conversation memory (in-memory, resets on restart)
conversations = {}

def get_bot_reply(sender_id, message_text):
    history = conversations.get(sender_id, [])
    history.append({"role": "user", "content": message_text})
    result = agent.invoke({"messages": history})
    reply = result["messages"][-1].content
    history.append({"role": "assistant", "content": reply})
    conversations[sender_id] = history[-20:]  # keep last 20 messages
    return reply

def send_message(recipient_id, text):
    url = f"https://graph.facebook.com/v21.0/me/messages?access_token={FB_PAGE_ACCESS_TOKEN}"
    payload = {"recipient": {"id": recipient_id}, "message": {"text": text}}
    requests.post(url, json=payload)

# ---- Webhook endpoints ----

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == FB_VERIFY_TOKEN:
        return PlainTextResponse(content=params.get("hub.challenge"))
    return PlainTextResponse(content="Verification failed", status_code=403)

@app.post("/webhook")
async def handle_webhook(request: Request):
    data = await request.json()
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]
                if "message" in event and "text" in event["message"]:
                    user_text = event["message"]["text"]
                    reply = get_bot_reply(sender_id, user_text)
                    send_message(sender_id, reply)
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "Messenger bot is running"}