import threading
from machine_lib import *
import random
import datetime
import requests
import json
import os
import sys
from os.path import expanduser
from requests.auth import HTTPBasicAuth

# 简单的selection表达式列表
simple_selections = [
    "turnover > 0.1 && long_count > 100",
    "turnover > 0.15 && short_count > 100",
    "fitness > 0.5 && turnover > 0.1",
    "long_count > 200 && fitness > 0.3",
    "short_count > 200 && turnover > 0.2",
    "turnover > 0.05 && long_count > 150",
    "fitness > 0.4 && short_count > 150",
    "turnover > 0.08 && fitness > 0.6",
    "long_count > 300 && turnover > 0.12",
    "short_count > 300 && fitness > 0.5",
    "turnover > 0.06 && long_count > 120 && short_count > 120",
    "fitness > 0.45 && turnover > 0.09",
    "long_count > 180 && fitness > 0.55",
    "short_count > 180 && turnover > 0.14",
    "turnover > 0.07 && fitness > 0.35 && long_count > 90"
]

def login():
    """登录WorldQuant BRAIN平台（支持cookie验证）"""
    cookie_path = os.path.join(os.path.dirname(__file__), "cookie.json")

    session = requests.Session()
    if not os.path.exists(cookie_path):
        print("cookie文件不存在，直接使用账号密码登陆")
        env_dist = os.environ
        username = env_dist.get("WQ_USERNAME")
        password = env_dist.get("WQ_PASSWORD")

        if username is None or password is None:
            with open(expanduser("denglu.txt")) as f:
                credentials = json.load(f)
            username, password = credentials

        session = relogin(username, password)
    else:
        with open(cookie_path, "r") as f:
            cookies = requests.utils.cookiejar_from_dict(json.load(f))
            session.cookies = cookies
        response = session.get("https://api.worldquantbrain.com/operators")
        if response.status_code in (401, 403):
            print("cookie文件失效，使用账号密码重新登陆")
            env_dist = os.environ
            username = env_dist.get("WQ_USERNAME")
            password = env_dist.get("WQ_PASSWORD")

            if username is None or password is None:
                with open(expanduser("denglu.txt")) as f:
                    credentials = json.load(f)
                username, password = credentials

            session = relogin(username, password)
        else:
            print("cookie文件有效")
    return session

def relogin(username: str, password: str):
    """重新登录并保存cookie"""
    session = requests.Session()
    session.auth = (username, password)
    while True:
        try:
            response = session.post("https://api.worldquantbrain.com/authentication")
            response.raise_for_status()
            print("登录成功, cookie文件更新")
            session.cookies = response.cookies
            cookie_path = os.path.join(os.path.dirname(__file__), "cookie.json")
            with open(cookie_path, "w") as f:
                json.dump(requests.utils.dict_from_cookiejar(response.cookies), f)
            return session
        except requests.exceptions.RequestException as e:
            print("登录失败，等待10s后再次尝试登录")
            time.sleep(10)

def get_simple_selection():
    """从简单表达式列表中随机选择一个"""
    return random.choice(simple_selections)

def get_combo_code_list():
    time_windows1 = [60,120,250,500]
    selected_window1 = random.choice(time_windows1)
    time_windows2 = [250, 500]
    selected_window2 = random.choice(time_windows2)
    ret = ['1',
           f'stats = generate_stats(alpha); innerCorr = self_corr(stats.returns, {selected_window2}); ic = if_else(innerCorr == 1.0, nan, innerCorr); maxCorr = reduce_max(ic); 1 - maxCorr',
           f'stats = generate_stats(alpha);alpha_mom = ts_sum(stats.returns,60);mom_rank = ts_rank(alpha_mom,500);if_else(mom_rank>0.9,1,if_else(mom_rank<0.1,-1,0))',
           f'stats = generate_stats(alpha);std = ts_std_dev(stats.returns,60);std_crowd = std /ts_delay(std,60);ts_rank(-std_crowd ,500)',
           f'stats = generate_stats(alpha);tv = ts_mean(stats.trade_value,60);tv_crowd = tv/ts_delay(tv,60);ts_rank(-tv_crowd,500)',
           f'stats = generate_stats(alpha);ic = self_corr(stats.returns,60);inneric = if_else(ic==1,nan,ic);ts_rank(-reduce_min(inneric),500)',
           f'combo_a(alpha, nlength = 255, mode = "algo1")',
           f'combo_a(normalize(alpha), nlength = 250, mode = "algo1")'
           ]
    return ret

def multi_simulate2_sa(alpha_pools, neut, region, universe, start):
    """
    改进的并发模拟函数 - 学习machine_lib.py的进度监控方式
    保持并发数恒定，通过进度监控动态提交新任务
    """
    global s
    s = login()
    brain_api_url = 'https://api.worldquantbrain.com'

    all_sa_pairs = alpha_pools[0]
    total_tasks = len(all_sa_pairs)
    print(f'📊 总任务数: {total_tasks}, 开始位置: {start}, 并发限制: 3')

    task_queue = all_sa_pairs[start:]
    active_tasks = {}
    completed_count = 0
    failed_count = 0

    def submit_simulation(task_index, task_data):
        max_retries = 5
        base_delay = 30

        for attempt in range(max_retries):
            try:
                sim_data_list = generate_sim_data_sa([task_data], region, universe, neut)
                sim_data = sim_data_list[0]

                simulation_response = s.post('https://api.worldquantbrain.com/simulations', json=sim_data)

                if simulation_response.status_code == 429:
                    if attempt < max_retries - 1:
                        retry_after = simulation_response.headers.get('Retry-After')
                        if retry_after:
                            delay = int(retry_after)
                        else:
                            delay = base_delay * (2 ** attempt)
                        print(f"🚦 任务{task_index} 429限流，{delay}秒后重试 ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ 任务{task_index} 429限流，已达到最大重试次数")
                        return None

                if simulation_response.status_code == 400:
                    print(f"❌ 任务{task_index} 400错误: {simulation_response.text}")
                    return None
                elif simulation_response.status_code != 201:
                    print(f"❌ 任务{task_index} 状态码错误: {simulation_response.status_code}")
                    return None

                progress_url = simulation_response.headers.get('Location')
                if progress_url:
                    print(f"✅ 任务{task_index} 已提交: {progress_url}")
                    return progress_url
                else:
                    print(f"❌ 任务{task_index} 无进度URL")
                    return None

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"❌ 任务{task_index} 提交异常: {e}，{delay}秒后重试 ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ 任务{task_index} 提交失败，已达到最大重试次数: {e}")
                    return None

        return None

    def check_simulation_status(task_index, progress_url):
        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = s.get(progress_url)

                if response.status_code == 504:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"⏳ 任务{task_index} 504超时，{delay}秒后重试 ({attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ 任务{task_index} 504超时，已达到最大重试次数")
                        return "ERROR"

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "UNKNOWN")

                    if status in ["ERROR", "FAILED"]:
                        error_msg = data.get("error", "Unknown error")
                        print(f"❌ 任务{task_index} 失败详情: {error_msg}")

                        if "progress" in data:
                            progress_info = data["progress"]
                            if "errors" in progress_info:
                                for error in progress_info["errors"]:
                                    print(f"   错误: {error}")

                    return status
                else:
                    print(f"⚠️ 任务{task_index} 状态检查失败: HTTP {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"   错误详情: {error_data}")
                    except:
                        print(f"   响应内容: {response.text[:200]}...")
                    return "ERROR"

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"❌ 任务{task_index} 网络异常: {e}，{delay}秒后重试 ({attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ 任务{task_index} 网络异常，已达到最大重试次数: {e}")
                    return "ERROR"

        return "ERROR"

    while task_queue or active_tasks:
        while len(active_tasks) < 3 and task_queue:
            task_data = task_queue.pop(0)
            task_index = start + completed_count + failed_count + len(active_tasks)

            progress_url = submit_simulation(task_index, task_data)
            if progress_url:
                active_tasks[task_index] = progress_url

            if task_queue:
                time.sleep(2)

        completed_tasks = []
        for task_index, progress_url in active_tasks.items():
            status = check_simulation_status(task_index, progress_url)

            if status == "COMPLETE":
                print(f"✅ 任务{task_index} 完成")
                completed_tasks.append(task_index)
                completed_count += 1
                time.sleep(15)

            elif status in ["ERROR", "FAILED"]:
                print(f"❌ 任务{task_index} 失败")
                completed_tasks.append(task_index)
                failed_count += 1
                time.sleep(5)

            elif status == "RUNNING":
                print(f"⏳ 任务{task_index} 运行中")

        for task_index in completed_tasks:
            del active_tasks[task_index]

        if active_tasks:
            time.sleep(15)

        if (completed_count + failed_count) % 50 == 0 and (completed_count + failed_count) > 0:
            print("🔄 重新登录...")
            s = login()

    print(f"🎉 模拟完成! 成功: {completed_count}, 失败: {failed_count}, 总计: {completed_count + failed_count}")

def generate_sim_data_sa(alpha_list, region, uni, neut):
    sim_data_list = []

    if isinstance(alpha_list, list) and len(alpha_list) == 1 and isinstance(alpha_list[0], tuple):
        selection_exp, combo_exp = alpha_list[0]

        simulation_data = {
            'type': 'SUPER',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': region,
                'universe': uni,
                'delay': 1,
                'decay': 6,
                'neutralization': neut,
                'truncation': 0.08,
                'pasteurization': 'ON',
                'unitHandling': 'VERIFY',
                'nanHandling': 'ON',
                'language': 'FASTEXPR',
                'visualization': False,
                'selectionHandling': 'POSITIVE',
                'selectionLimit': 10
            },
            'selection': selection_exp,
            'combo': combo_exp
        }

        sim_data_list.append(simulation_data)
    else:
        for selection_exp, combo_exp in alpha_list:
            simulation_data = {
                'type': 'SUPER',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': uni,
                    'delay': 1,
                    'decay': 6,
                    'neutralization': neut,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'ON',
                    'language': 'FASTEXPR',
                    'visualization': False,
                'selectionHandling': 'POSITIVE',
                'selectionLimit': 10,
                },
                'selection': selection_exp,
                'combo': combo_exp
            }

            sim_data_list.append(simulation_data)

    return sim_data_list

def save_progress(session_seed, completed_configs, current_region, current_universe, current_neutralization):
    progress_data = {
        "session_seed": session_seed,
        "completed_configs": completed_configs,
        "current_region": current_region,
        "current_universe": current_universe,
        "current_neutralization": current_neutralization,
        "timestamp": datetime.datetime.now().isoformat(),
        "version": "1.0"
    }

    progress_file = os.path.join(os.path.dirname(__file__), "sa_progress.json")
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        print(f"💾 进度已保存: {len(completed_configs)} 个配置已完成")
    except Exception as e:
        print(f"⚠️ 保存进度失败: {e}")

def load_progress():
    progress_file = os.path.join(os.path.dirname(__file__), "sa_progress.json")
    if not os.path.exists(progress_file):
        return None

    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        print(f"📂 发现进度文件: {len(progress_data.get('completed_configs', []))} 个配置已完成")
        return progress_data
    except Exception as e:
        print(f"⚠️ 加载进度失败: {e}")
        return None

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='SA模拟自动化工具')
    parser.add_argument('--seed', type=int, help='随机种子，用于重现相同序列')
    parser.add_argument('--resume', action='store_true', help='从上次进度继续')
    parser.add_argument('--fresh', action='store_true', help='重新开始，忽略进度文件')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    progress_data = None
    if args.resume and not args.fresh:
        progress_data = load_progress()

    if args.fresh:
        progress_data = None
        print("🆕 强制重新开始，忽略进度文件")

    if args.seed:
        session_seed = args.seed
        print(f"🎲 使用指定种子: {session_seed}")
    elif progress_data and 'session_seed' in progress_data:
        session_seed = progress_data['session_seed']
        print(f"🎲 使用进度文件中的种子: {session_seed}")
    else:
        session_seed = random.randint(1, 1000000)
        print(f"🎲 生成新随机种子: {session_seed}")

    random.seed(session_seed)

    completed_configs = progress_data.get('completed_configs', []) if progress_data else []

    print(f"🔄 会话种子: {session_seed}")
    print(f"✅ 已完成配置: {len(completed_configs)}")

    while True:
        selection_exp = []
        exp = get_simple_selection()
        selection_exp.append(exp)

        combo_exp = get_combo_code_list()
        sa_list = [(i, j) for i in selection_exp for j in combo_exp]
        print(f"\n📋 本轮生成: {len(sa_list)} 个SA策略")
        print(f"Sample Selection: {exp}")

        pools = [sa_list]
        print(f"🔄 总共 {len(sa_list)} 个SA对，将分批并发处理（最多3个并发）")

        region_dict = {"usa": ("USA", ["TOP3000"]),
                       "eur": ("EUR", ["TOP2500"])}

        neut_opt = {
            "USA": [  # USA TOP3000支持的所有neutralization选项
                "NONE",
                "REVERSION_AND_MOMENTUM",
                "STATISTICAL",
                "CROWDING",
                "FAST",
                "SLOW",
                "MARKET",
                "SECTOR",
                "INDUSTRY",
                "SUBINDUSTRY",
                "SLOW_AND_FAST"
            ],
            "EUR": [  # EUR TOP2500支持的所有neutralization选项
                "NONE",
                "REVERSION_AND_MOMENTUM",
                "STATISTICAL",
                "CROWDING",
                "FAST",
                "SLOW",
                "MARKET",
                "SECTOR",
                "INDUSTRY",
                "SUBINDUSTRY",
                "COUNTRY",
                "SLOW_AND_FAST"
            ]
        }

        regi = ['usa', 'eur']
        random.shuffle(regi)

        for k in regi:
            region_name = region_dict[k][0]
            universe_list = region_dict[k][1][:1]
            neutralization_list = neut_opt[k.upper()]

            print(f"\n🌍 开始处理地区: {region_name}")
            print(f"   Universes: {universe_list}")
            print(f"   Neutralizations: {neutralization_list}")

            for universe in universe_list:
                print(f"\n📊 处理Universe: {universe} ({region_name})")

                start_neut_index = 0
                if (progress_data and
                    progress_data.get('current_region') == k and
                    progress_data.get('current_universe') == universe):

                    current_neut = progress_data.get('current_neutralization')
                    if current_neut in neutralization_list:
                        start_neut_index = neutralization_list.index(current_neut)
                        print(f"⏭️ 跳过已完成的neutralization: {neutralization_list[:start_neut_index]}")

                for i, neutralization in enumerate(neutralization_list[start_neut_index:], start_neut_index):
                    print(f"\n⚙️ 配置: {neutralization} neutralization")
                    print(f"   地区: {region_name}")
                    print(f"   Universe: {universe}")

                    multi_simulate2_sa(pools, neutralization, region_name, universe, 0)

                    config_key = f"{region_name}-{universe}-{neutralization}"
                    if config_key not in completed_configs:
                        completed_configs.append(config_key)

                    save_progress(session_seed, completed_configs, k, universe, neutralization)

                    print(f"✅ 完成配置: {neutralization} - {region_name} - {universe}")

                print(f"🎯 完成Universe: {universe} 的所有配置")

            print(f"🏁 完成地区: {region_name} 的所有处理")

        print("\n🎉 本轮循环完成，准备下一轮...")
        print(f"💾 最终进度已保存: {len(completed_configs)} 个配置已完成")
