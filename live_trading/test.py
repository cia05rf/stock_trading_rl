import websocket
import json
import httpx
import asyncio
import threading
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
FMP_API_KEY = os.getenv("FMP_API_KEY")
T212_API_KEY = os.getenv("T212_API_KEY")
T212_BASE_URL = "https://demo.trading212.com/api/v0" # Switch to 'live' when ready

# The stocks you want to watch (Add more S&P 500 tickers as needed)
WATCHLIST = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "EURUSD"]

# --- TRADING EXECUTION (Trading212) ---
def execute_trade(ticker, side="BUY"):
    """
    Sends a Market Order to Trading212.
    Note: We use a sync wrapper here because WebSocket callbacks are synchronous.
    For high performance, you might push this to an async queue.
    """
    # T212 US tickers usually have "_US_EQ" appended
    t212_ticker = f"{ticker}_US_EQ"
    
    print(f"🚀 EXECUTING {side} for {ticker} on Trading212...")
    
    headers = {"Authorization": T212_API_KEY}
    payload = {
        "ticker": t212_ticker,
        "quantity": 1, # Be careful with quantity!
        "orderType": "MARKET"
    }
    
    try:
        # We use httpx.Client (sync) specifically for this simple example
        with httpx.Client(base_url=T212_BASE_URL, headers=headers) as client:
            response = client.post("/equity/orders/place", json=payload)
            if response.status_code == 200:
                print(f"✅ Order Placed: {response.json()}")
            else:
                print(f"❌ Order Failed: {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

# --- REAL-TIME DATA (FMP WebSocket) ---
def on_message(ws, message):
    data = json.loads(message)
    
    # FMP sends data like: {'type': 'trade', 's': 'AAPL', 'p': 175.50, ...}
    print(data)
    if data.get('type') == 'trade':
        ticker = data['s']
        price = data['p']
        
        print(f"Tick: {ticker} @ ${price}")

def on_error(ws, error):
    print(f"Stream Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Stream Closed ###")

def on_open(ws):
    print("✅ Connected to FMP Real-Time Stream!")
    
    # Subscribe to the specific watchlist to save bandwidth
    subscribe_msg = {
        "event": "subscribe",
        "data": {
            "ticker": WATCHLIST
        }
    }
    ws.send(json.dumps(subscribe_msg))

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    # 1. Start the WebSocket
    ws_url = f"wss://websockets.financialmodelingprep.com?apikey={FMP_API_KEY}"
    
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run forever
    ws.run_forever()