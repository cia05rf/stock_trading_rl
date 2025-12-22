import requests
import json
import sys
import config

def search_symbol(query):
    """Search for a symbol to find correct ticker and exchange."""
    url = f"{config.EODHD_BASE_URL}/search/{query}"
    params = {"api_token": config.API_KEY, "fmt": "json"}
    
    print(f"Searching for '{query}'...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("No results found.")
            return

        print(f"\nFound {len(data)} results:")
        print(f"{'Code':<15} {'Exchange':<10} {'Name':<30} {'Type'}")
        print("-" * 70)
        for item in data[:10]:  # Show top 10
            print(f"{item.get('Code', ''):<15} {item.get('Exchange', ''):<10} {item.get('Name', '')[:30]:<30} {item.get('Type', '')}")
            
    except Exception as e:
        print(f"Error searching: {e}")

def get_fundamentals(ticker, exchange):
    """Get fundamentals to see valid dates."""
    full_symbol = f"{ticker}.{exchange}"
    url = f"{config.EODHD_BASE_URL}/fundamentals/{full_symbol}"
    params = {"api_token": config.API_KEY, "fmt": "json"}
    
    print(f"\nFetching fundamentals for {full_symbol}...")
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 404:
            print(f"❌ Error 404: Symbol '{full_symbol}' not found.")
            return
            
        response.raise_for_status()
        data = response.json()
        
        general = data.get("General", {})
        print(f"✓ Symbol Found: {general.get('Code')}.{general.get('Exchange')}")
        print(f"  Name: {general.get('Name')}")
        print(f"  Currency: {general.get('CurrencyCode')}")
        print(f"  Start Date: {general.get('StartDate', 'N/A')}")
        print(f"  End Date: {general.get('EndDate', 'N/A')}")
        print(f"  Is Delisted: {general.get('IsDelisted', 'N/A')}")
        
    except Exception as e:
        print(f"Error fetching fundamentals: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ticker.py <symbol_or_search_term> [exchange_code]")
        print("Example: python debug_ticker.py SIE XETR")
        print("Example: python debug_ticker.py Siemens")
        sys.exit(1)
        
    arg1 = sys.argv[1]
    
    if len(sys.argv) == 3:
        # Ticker and Exchange provided
        get_fundamentals(arg1, sys.argv[2])
    else:
        # Just search
        search_symbol(arg1)

