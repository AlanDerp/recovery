import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth

try:
    with open(expanduser('brain.txt')) as f:
        credentials = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Credentials file 'brain.txt' not found in home directory.")

username, password = credentials

sess = requests.Session()
sess.auth = HTTPBasicAuth(username, password)

response = sess.post('https://api.worldquantbrain.com/authentication')

print(response.status_code)
print(response.json())

simulation_data = {
    'type': 'REGULAR',
    'settings': {
        'instrumentType': 'EQUITY',
        'region': 'USA',
        'universe': 'TOP3000',
        'delay': 1,
        'decay': 0,
        'neutralization': 'INDUSTRY',
        'truncation': 0.08,
        'pasteurization': 'ON',
        'unitHandling': 'VERIFY',
        'nanHandling': 'OFF',
        'language': 'FASTEXPR',
        'visualization': False,
    },
    'regular': 'liabilities/assets'
}

from time import sleep

sim_resp = sess.post(
    'https://api.worldquantbrain.com/simulations',
    json=simulation_data,
)

sim_progress_url = sim_resp.headers['Location']
print(sim_progress_url)
while True:
    sim_progress_resp = sess.get(sim_progress_url)
    retry_after_sec = float(sim_progress_resp.headers.get("Retry-After", 0))
    if retry_after_sec == 0:
        break
    sleep(retry_after_sec)

alpha_id = sim_progress_resp.json()["alpha"]

print(alpha_id)
