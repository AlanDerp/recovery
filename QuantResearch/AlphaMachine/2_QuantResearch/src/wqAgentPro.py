import requests
import json
import logging
import time
import concurrent.futures
from os.path import expanduser
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("WQAgent")


class BrainAuth:
    """处理 WorldQuant Brain 的认证与会话管理"""
    def __init__(self, cred_file: str = 'brain.txt'):
        self.cred_file = expanduser(cred_file)
        self.session = requests.Session()

    def sign_in(self) -> requests.Session:
        """读取凭证并登录，返回已认证的 Session"""
        try:
            with open(self.cred_file, 'r') as f:
                credentials = json.load(f)
            username, password = credentials

            self.session.auth = HTTPBasicAuth(username, password)
            response = self.session.post('https://api.worldquantbrain.com/authentication')
            response.raise_for_status()
            logger.info("Successfully authenticated with WorldQuant Brain.")
            return self.session
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise


class DataFetcher:
    """处理数据集和字段的获取"""
    def __init__(self, session: requests.Session):
        self.session = session

    def get_datafields(self, dataset_id: str, search_scope: Dict[str, str], field_type: str = "MATRIX") -> List[str]:
        """拉取指定数据集下的字段，并根据 field_type 进行过滤"""
        logger.info(f"Fetching data fields for dataset: {dataset_id}...")
        instrument_type = search_scope.get('instrumentType', 'EQUITY')
        region = search_scope.get('region', 'USA')
        delay = search_scope.get('delay', '1')
        universe = search_scope.get('universe', 'TOP3000')

        url_template = (
            f"https://api.worldquantbrain.com/data-fields?"
            f"instrumentType={instrument_type}&region={region}&delay={delay}"
            f"&universe={universe}&dataset.id={dataset_id}&limit=50&offset={{x}}"
        )

        try:
            initial_resp = self.session.get(url_template.format(x=0)).json()
            count = initial_resp.get('count', 0)

            datafields_list = []
            for x in range(0, count, 50):
                resp = self.session.get(url_template.format(x=x)).json()
                datafields_list.extend(resp.get('results', []))

            filtered_fields = [item['id'] for item in datafields_list if item.get('type') == field_type]
            logger.info(f"Fetched {len(filtered_fields)} fields of type {field_type}.")
            return filtered_fields
        except Exception as e:
            logger.error(f"Failed to fetch data fields: {e}")
            return []


class LLMGenerator:
    """处理与大语言模型的交互，将自然语言转换为 Alpha 模板"""
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        import openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_template(self, user_prompt: str) -> str:
        """
        输入自然语言思路，返回 FastExpr 模板。
        使用占位符 {MATRIX} 代表数据字段，{DAYS} 代表时间周期。
        """
        logger.info(f"Requesting LLM to generate template for: {user_prompt}")
        system_prompt = (
            "你是一个顶级的量化策略专家。请将用户的量化思路转化为 WorldQuant Brain 的 FastExpr 语言模板。\n"
            "规则：\n"
            "1. 使用 {MATRIX} 作为基本面数据字段的占位符。\n"
            "2. 使用 {DAYS} 作为时间周期的占位符（如果需要）。\n"
            "3. 只返回 JSON 格式数据，键名为 'template'，值为表达式字符串。\n"
            "示例输出：{\"template\": \"group_rank(ts_rank({MATRIX}, {DAYS}), industry)\"}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            result = json.loads(response.choices[0].message.content)
            template = result.get("template", "")
            logger.info(f"LLM generated template: {template}")
            return template
        except Exception as e:
            logger.error(f"LLM Generation failed: {e}")
            raise


class SimulationAgent:
    """核心智能体：处理 Alpha 组装、并发提交回测与结果轮询"""
    def __init__(self, session: requests.Session):
        self.session = session
        self.fail_tolerance = 15

    def assemble_alphas(self, template: str, fields: List[str], days_list: List[int]) -> List[str]:
        """将占位符替换为实际字段和参数，组装出所有可执行的 Alpha 表达式"""
        alphas = []
        for field in fields:
            if "{DAYS}" in template:
                for days in days_list:
                    expr = template.replace("{MATRIX}", field).replace("{DAYS}", str(days))
                    alphas.append(expr)
            else:
                expr = template.replace("{MATRIX}", field)
                alphas.append(expr)
        logger.info(f"Assembled {len(alphas)} alpha expressions from template.")
        return alphas

    def _submit_and_poll(self, alpha_expr: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """单个 Alpha 的提交与轮询逻辑"""
        payload = {
            "type": "REGULAR",
            "settings": settings,
            "regular": alpha_expr
        }

        failure_count = 0
        location_url = None

        while failure_count < self.fail_tolerance:
            try:
                resp = self.session.post('https://api.worldquantbrain.com/simulations', json=payload)
                if resp.status_code == 201:
                    location_url = resp.headers['Location']
                    break
                elif resp.status_code == 401:
                    raise Exception("Unauthorized. Session might have expired.")
                else:
                    logger.warning(f"Unexpected status code {resp.status_code}. Retrying...")
                    time.sleep(5)
                    failure_count += 1
            except Exception as e:
                logger.error(f"Submission error: {e}. Retrying in 15s...")
                time.sleep(15)
                failure_count += 1

        if not location_url:
            return {"expr": alpha_expr, "status": "FAILED_TO_SUBMIT"}

        logger.info(f"Polling results for: {location_url}")
        while True:
            try:
                prog_resp = self.session.get(location_url)
                prog_data = prog_resp.json()
                status = prog_data.get("status", "")

                if status not in ["PENDING", "WARNING"]:
                    if status in ["SUCCESS", "FINISHED", "COMPLETE"]:
                        alpha_info = prog_data.get("alpha", {})
                        metrics = {
                            "expr": alpha_expr,
                            "status": status,
                            "fitness": alpha_info.get("fitness"),
                            "sharpe": alpha_info.get("sharpe"),
                            "returns": alpha_info.get("returns"),
                            "margin": alpha_info.get("margin")
                        }
                        logger.info(f"Simulation completed! Sharpe: {metrics['sharpe']}")
                        return metrics
                    else:
                        return {"expr": alpha_expr, "status": status, "error_msg": prog_data.get("message")}

                time.sleep(5)
            except Exception as e:
                logger.error(f"Polling error for {location_url}: {e}")
                time.sleep(5)

    def run_batch(self, alpha_expressions: List[str], settings: Dict[str, Any], concurrency: int = 3) -> List[Dict[str, Any]]:
        """使用线程池并发执行回测任务"""
        logger.info(f"Starting batch simulation with concurrency {concurrency}...")
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_alpha = {
                executor.submit(self._submit_and_poll, expr, settings): expr
                for expr in alpha_expressions
            }

            for future in concurrent.futures.as_completed(future_to_alpha):
                alpha = future_to_alpha[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    logger.error(f"Alpha {alpha} generated an exception: {exc}")
                    results.append({"expr": alpha, "status": "EXCEPTION"})

        return results


def main():
    OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxx"
    USER_PROMPT = "我想要一个基于动量和行业中性化的因子，逻辑是比较不同基本面数据在过去时间窗口的变化率。使用 fundamental6 数据集。"

    SEARCH_SCOPE = {
        'instrumentType': 'EQUITY',
        'region': 'USA',
        'delay': '1',
        'universe': 'TOP3000'
    }

    SIMULATION_SETTINGS = {
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
    }

    auth = BrainAuth('brain.txt')
    session = auth.sign_in()

    fetcher = DataFetcher(session)
    agent = SimulationAgent(session)
    llm = LLMGenerator(api_key=OPENAI_API_KEY)

    template = llm.generate_template(USER_PROMPT)
    print(f"\n>>> 获得 Alpha 模板: {template}\n")

    fields = fetcher.get_datafields(dataset_id='fundamental6', search_scope=SEARCH_SCOPE, field_type="MATRIX")
    fields_to_test = fields[:5]

    days_list = [60, 200]
    alphas = agent.assemble_alphas(template, fields_to_test, days_list)
    print(f"\n>>> 组装完毕，共 {len(alphas)} 个表达式等待回测。\n")

    results = agent.run_batch(alphas, SIMULATION_SETTINGS, concurrency=3)

    print("\n" + "="*40)
    print(" 回测结果报告 (部分)")
    print("="*40)
    for res in results:
        status = res.get('status')
        expr = res.get('expr')
        if status in ["SUCCESS", "FINISHED", "COMPLETE"]:
            print(f"[SUCCESS] Sharpe: {res.get('sharpe', 'N/A')} | Fitness: {res.get('fitness', 'N/A')} | Expr: {expr[:50]}...")
        else:
            print(f"[{status}] Expr: {expr[:50]}... | Msg: {res.get('error_msg', '')}")


if __name__ == "__main__":
    main()
