import os
import requests
import json
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    filename='agent_simulation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class BrainAuth:
    """负责处理 WorldQuant Brain 的登录与会话维持"""
    def __init__(self):
        self.session = requests.Session()
        self.username = os.getenv('API_USERNAME')
        self.password = os.getenv('API_PASSWORD')

    def sign_in(self):
        """加载凭证并完成鉴权"""
        if not self.username or not self.password:
            logger.error("环境变量 API_USERNAME 或 API_PASSWORD 未设置。")
            raise ValueError("请检查 .env 文件中的登录凭证。")

        try:
            self.session.auth = HTTPBasicAuth(self.username, self.password)
            response = self.session.post('https://api.worldquantbrain.com/authentication')
            response.raise_for_status()

            logger.info("成功登录 WorldQuant Brain。")
            print(">>> 登录成功")
            return self.session
        except Exception as e:
            logger.error(f"登录失败: {e}")
            raise


class DataFetcher:
    def __init__(self, session):
        self.session = session

    def get_datafields(self, dataset_id: str, search_scope: dict):
        url_template = (
            f"https://api.worldquantbrain.com/data-fields?"
            f"instrumentType={search_scope.get('instrumentType', 'EQUITY')}&region={search_scope.get('region', 'USA')}"
            f"&delay={search_scope.get('delay', '1')}&universe={search_scope.get('universe', 'TOP3000')}"
            f"&dataset.id={dataset_id}&limit=50&offset={{x}}"
        )
        try:
            initial_resp = self.session.get(url_template.format(x=0)).json()
            count = initial_resp.get('count', 0)
            datafields_list = []
            for x in range(0, count, 50):
                resp = self.session.get(url_template.format(x=x))
                datafields_list.append(resp.json().get('results', []))
            datafields_list_flat = [item for sublist in datafields_list for item in sublist]
            return pd.DataFrame(datafields_list_flat)
        except Exception as e:
            logger.error(f"拉取数据字段失败: {e}")
            return pd.DataFrame()


class LLMGenerator:
    def __init__(self, api_key, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model

    def generate_alpha(self, user_idea: str, data_fields: list, operators: dict) -> str:
        system_prompt = """
        你是一位顶级的 WorldQuant Brain 量化研究员。
        你的任务是将用户的自然语言思路转化为严格的 WorldQuant Brain FastExpr Alpha 表达式。
        请仅返回一个 JSON 对象，格式为：{"alpha_expression": "你的表达式"}。
        """
        user_prompt = f"思路: {user_idea}\n可用字段: {data_fields[:20]}\n操作符: {operators}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            result_json = json.loads(response.choices[0].message.content)
            return result_json.get("alpha_expression", "")
        except Exception as e:
            logger.error(f"DeepSeek 生成失败: {e}")
            return ""


class SimulationAgent:
    def __init__(self, session):
        self.session = session
        self.simulation_url = 'https://api.worldquantbrain.com/simulations'

    def run_simulation(self, alpha_expression: str, settings: dict):
        payload = {"type": "REGULAR", "settings": settings, "regular": alpha_expression}
        try:
            resp = self.session.post(self.simulation_url, json=payload)
            resp.raise_for_status()
            location = resp.headers.get('Location')
            return self._poll_results(location)
        except Exception as e:
            logger.error(f"提交回测报错: {e}")
            return None

    def _poll_results(self, url):
        for _ in range(60):
            res = self.session.get(url).json()
            status = res.get('status')
            print(f"当前状态: {status}")
            if status == 'COMPLETE':
                return res
            if status in ['ERROR', 'FAIL']:
                return res
            time.sleep(5)
        return None


def main_workflow():
    DEEPSEEK_API_KEY = os.getenv('API_KEY')

    if not DEEPSEEK_API_KEY:
        print("错误：未在 .env 中找到 API_KEY")
        return

    user_idea = input("请输入量化因子思路: ")

    search_scope = {'region': 'USA', 'delay': '1', 'universe': 'TOP3000', 'instrumentType': 'EQUITY'}
    sim_settings = {
        "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
        "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY",
        "truncation": 0.01, "pasteurization": "ON", "unitHandling": "VERIFY",
        "nanHandling": "OFF", "language": "FASTEXPR", "visualization": False,
    }
    operators = {'group_ops': ['group_rank'], 'ts_ops': ['ts_rank'], 'group_basis': ['industry']}

    print("\n>>> 正在初始化 Agent...")
    auth = BrainAuth()
    session = auth.sign_in()

    fetcher = DataFetcher(session)
    llm_agent = LLMGenerator(api_key=DEEPSEEK_API_KEY)
    sim_agent = SimulationAgent(session)

    fnd6_df = fetcher.get_datafields(dataset_id='fundamental6', search_scope=search_scope)
    matrix_fields = fnd6_df[fnd6_df['type'] == "MATRIX"]['id'].tolist() if not fnd6_df.empty else ['m_rev']

    alpha_expression = llm_agent.generate_alpha(user_idea, matrix_fields, operators)
    if alpha_expression:
        print(f"\n>>> 表达式: {alpha_expression}")
        final_result = sim_agent.run_simulation(alpha_expression, sim_settings)
        if final_result and final_result.get('status') == 'COMPLETE':
            print(f"\n>>> 挖掘成功！Sharpe: {final_result.get('sharpe')}")


if __name__ == "__main__":
    main_workflow()
