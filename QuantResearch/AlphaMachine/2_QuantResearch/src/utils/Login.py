import json
import requests
from os.path import expanduser
from requests.auth import HTTPBasicAuth


def sign_in(cred_file="brain.txt"):
    with open(expanduser(cred_file)) as f:
        username, password = json.load(f)

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)
    response = sess.post("https://api.worldquantbrain.com/authentication")
    response.raise_for_status()
    return sess
