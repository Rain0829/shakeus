import requests, sys
from config import MAC_IP, MAC_PORT

def trigger_now():
    url = f"http://{MAC_IP}:{MAC_PORT}/alarm/trigger"
    r = requests.get(url)
    print(f"Response: {r.status_code} {r.text}")

if __name__ == "__main__":
    trigger_now()
