#!/usr/bin/env python
"""Test OpenCelliD tower API connectivity and key validity"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OCID_KEY = os.environ.get('OPENCELLID_API_KEY')
print(f"OpenCelliD API Key configured: {bool(OCID_KEY)}")

if not OCID_KEY:
    print("❌ ERROR: OPENCELLID_API_KEY not found in .env file")
    print("\nTo get a free API key:")
    print("1. Visit: https://opencellid.org/")
    print("2. Sign up for a free account")
    print("3. Copy your API key")
    print("4. Update .env file with: OPENCELLID_API_KEY=your_key_here")
    print("5. Restart Flask: python app.py")
    exit(1)

print(f"API Key: {OCID_KEY[:20]}...")

# Test with Mumbai coordinates
lat, lng = 19.0760, 72.8777
delta = 0.008
bbox = f"{lat-delta},{lng-delta},{lat+delta},{lng+delta}"
url = f"https://opencellid.org/cell/getInArea?key={OCID_KEY}&BBOX={bbox}&format=json&limit=500"

print(f"\n🔍 Testing tower fetch for Mumbai area...")
print(f"URL: {url[:80]}...")

try:
    resp = requests.get(url, timeout=15)
    print(f"Status Code: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        cells = data.get('cells', [])
        print(f"✅ SUCCESS! Found {len(cells)} towers")
        
        if cells:
            first = cells[0]
            print(f"\nFirst tower sample:")
            print(f"  - Location: ({first.get('lat')}, {first.get('lon')})")
            print(f"  - Radio: {first.get('radio')}")
            print(f"  - MCC/MNC: {first.get('mcc')}/{first.get('mnc')}")
            print(f"  - Cell ID: {first.get('cellid')}")
    else:
        print(f"❌ API Error: {resp.status_code}")
        print(f"Response: {resp.text[:200]}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")
    print("\nTroubleshooting:")
    print("- Check internet connection")
    print("- Verify OpenCelliD API is reachable")
    print("- Check if API key is valid/not rate-limited")

print("\n" + "="*60)
print("To fix the issue:")
print("="*60)
print("\n1. If API key is invalid/expired:")
print("   - Visit https://opencellid.org/")
print("   - Get a new free API key")
print("   - Update .env: OPENCELLID_API_KEY=your_new_key")
print("\n2. If API is rate limited (free tier limit):")
print("   - Wait a few minutes for rate limit to reset")
print("   - Or upgrade to paid plan")
print("\n3. If test passes but Flask still fails:")
print("   - Kill Flask server (CTRL+C)")
print("   - Run: python app.py again")
print("   - Refresh browser")
