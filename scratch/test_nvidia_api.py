import requests
import os

NVIDIA_API_KEY = "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"
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "moonshotai/kimi-k2.5"

headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Accept": "application/json"
}

payload = {
    "model": NVIDIA_MODEL,
    "messages": [{"role": "user", "content": "Hello, respond with 'API OK'"}],
    "max_tokens": 100,
    "temperature": 0.2,
    "top_p": 1.0,
    "stream": False,
    "chat_template_kwargs": {"thinking": False},
}

try:
    print(f"Testing NVIDIA API with model {NVIDIA_MODEL}...")
    response = requests.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=10)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response:", response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
    else:
        print("Error details:", response.text)
except Exception as e:
    print(f"Test failed: {e}")
