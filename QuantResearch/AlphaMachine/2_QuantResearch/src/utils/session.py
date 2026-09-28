import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth

with open(expanduser('brain.txt')) as f:
    credentials = json.load(f)

username, password = credentials

sess = requests.Session()
sess.auth = HTTPBasicAuth(username, password)

response = sess.post('https://api.worldquantbrain.com/authentication')
response.raise_for_status()

print("Authentication status:", response.status_code)
print("Authentication response:", response.json())

cookies = sess.cookies.get_dict()
token = cookies.get("t")

if not token:
    raise RuntimeError("Login failed: token not found in cookies")

print("JWT token (t):", token)


def fetch_all_datafields(sess):
    url = "https://platform.worldquantbrain.com/data/data-sets/analyst4"

    headers = {
        "Accept": "application/json;version=2.0",
        "Origin": "https://platform.worldquantbrain.com",
        "Referer": "https://platform.worldquantbrain.com/",
        "User-Agent": "Mozilla/5.0"
    }

    base_params = {
        "delay": 1,
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "limit": 200
    }

    all_fields = []
    offset = 0

    while True:
        params = {**base_params, "offset": offset}

        resp = sess.get(url, params=params, headers=headers)
        resp.raise_for_status()

        payload = resp.json()
        results = payload.get("results", [])

        if not results:
            break

        all_fields.extend(results)
        offset += len(results)

        print(f"Fetched {offset} / {payload.get('count')}")

    return all_fields


datafields = fetch_all_datafields(sess)

print("Total fields:", len(datafields))
print(datafields[0])
