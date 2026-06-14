#!/usr/bin/env python3
"""Test script — simulates an incoming WhatsApp webhook event."""
import sys, time, requests

msg = sys.argv[1] if len(sys.argv) > 1 else "Hello, do you have coconut oil?"

payload = {
    "object": "whatsapp_business_account",
    "entry": [{"id": "123", "changes": [{"value": {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15551498140", "phone_number_id": "test"},
        "messages": [{
            "from": "9779765341686",
            "id": f"wamid.test_{int(time.time())}",
            "timestamp": str(int(time.time())),
            "type": "text",
            "text": {"body": msg},
        }]
    }, "field": "messages"}]}]
}

try:
    r = requests.post("http://localhost:5001/webhook", json=payload, timeout=5)
    print(f"Response: {r.status_code} {r.text}")
except Exception as e:
    print(f"Error: {e} — is the webhook server running on :5001?")
