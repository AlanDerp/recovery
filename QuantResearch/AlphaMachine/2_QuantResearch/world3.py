import requests
import json
from os.path import expanduser
from requests.auth import HTTPBasicAuth


def sign_in():
    with open(expanduser('brain.txt')) as f:
        credentials = json.load(f)

    username, password = credentials

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)

    response = sess.post('https://api.worldquantbrain.com/authentication')

    print(response.status_code)
    print(response.json())
    return sess


sess = sign_in()


def get_datafields(
        s,
        searchScope,
        dataset_id: str = '',
        search: str = ''
):
    import pandas as pd
    instrument_type = searchScope['instrumentType']
    region = searchScope['region']
    delay = searchScope['delay']
    universe = searchScope['universe']

    if len(search) == 0:
        url_template = "https://api.worldquantbrain.com/data-fields?" + \
                       f"&instrumentType={instrument_type}" + \
                       f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" + \
                       "&offset={x}"
        count = s.get(url_template.format(x=0)).json()['count']
    else:
        url_template = "https://api.worldquantbrain.com/data-fields?" + \
                       f"&instrumentType={instrument_type}" + \
                       f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" + \
                       f"&search={search}" + \
                       "&offset={x}"
        count = 100

    datafields_list = []
    for x in range(0, count, 50):
        datafields = s.get(url_template.format(x=x))
        datafields_list.append(datafields.json()['results'])

    datafields_list_flat = [item for sublist in datafields_list for item in sublist]

    datafields_df = pd.DataFrame(datafields_list_flat)
    return datafields_df


searchScope = {'region': 'USA', 'delay': '1', 'universe': 'TOP3000', 'instrumentType': 'EQUITY'}
fnd6 = get_datafields(s=sess, searchScope=searchScope, dataset_id='fundamental6')
fnd6 = fnd6[fnd6['type'] == "MATRIX"]
datafields_list_fnd6 = fnd6['id'].values
print(datafields_list_fnd6)
print(len(datafields_list_fnd6))

group_compare_op = ['group_rank', 'group_zscore', 'group_neutralize']
ts_compare_op = ['ts_rank', 'ts_zscore', 'ts_av_diff']
company_fundamentals = datafields_list_fnd6
days = [60, 200]
group = ['market', 'industry', 'subindustry', 'sector', 'densify(pv13_h_f1_sector)']
alpha_expressions = []
for gco in group_compare_op:
    for tco in ts_compare_op:
        for cf in company_fundamentals:
            for d in days:
                for grp in group:
                    alpha_expressions.append(f"{gco}({tco}({cf}, {d}), {grp})")

print(f"there are total {len(alpha_expressions)} alpha expressions")

print(alpha_expressions[:5])
print(len(alpha_expressions))

alpha_list = []

for index, alpha_expression in enumerate(alpha_expressions, start=1):
    print(f"正在循环第 {index} 个元素")
    print("正在将如下alpha表达式与setting封装")
    simulation_data = {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "SUBINDUSTRY",
            "truncation": 0.01,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False,
        },
        "regular": alpha_expression
    }
    alpha_list.append(simulation_data)
    print(f"there are {len(alpha_list)} Alphas to simulate")

print(alpha_list[0])

import logging
logging.basicConfig(filename='simulation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from time import sleep

alpha_fail_attempt_tolerance = 15

for alpha in alpha_list[0:3]:
    keep_trying = True
    failure_count = 0

    while keep_trying:
        try:
            sim_resp = sess.post(
                'https://api.worldquantbrain.com/simulations',
                json=alpha
            )

            sim_progress_url = sim_resp.headers['Location']
            logging.info(f'Alpha location is: {sim_progress_url}')
            print(f'Alpha location is: {sim_progress_url}')
            keep_trying = False

        except Exception as e:
            logging.error(f"No Location, sleep 15 and retry, error message: {str(e)}")
            print("No Location, sleep 15 and retry")
            sleep(15)
            failure_count += 1

            if failure_count >= alpha_fail_attempt_tolerance:
                sess = sign_in()
                failure_count = 0
                logging.error(f"No location for too many times, move to next alpha {alpha['regular']}")
                print(f"No location for too many times, move to next alpha {alpha['regular']}")
                break
