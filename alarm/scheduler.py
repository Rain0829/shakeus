import schedule, time, requests
from config import ALARM_TIME, MAC_IP, MAC_PORT

def trigger():
    try:
        requests.get(f"http://{MAC_IP}:{MAC_PORT}/alarm/trigger", timeout=5)
        print("Alarm triggered!")
    except Exception as e:
        print(f"Failed to reach Mac: {e}")

schedule.every().day.at(ALARM_TIME).do(trigger)
print(f"Alarm set for {ALARM_TIME}. Waiting...")

while True:
    schedule.run_pending()
    time.sleep(10)
