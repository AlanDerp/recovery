import os
import requests
import json
import time
import logging
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from pro import DataFetcher, LLMGenerator, SimulationAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler('agent_service.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AlphaService")


class BrainAuth:
    def __init__(self):
        self.session = requests.Session()
        self.username = os.getenv('API_USERNAME')
        self.password = os.getenv('API_PASSWORD')
        self.last_login = None

    def sign_in(self):
        """执行登录并记录时间"""
        try:
            self.session.auth = HTTPBasicAuth(self.username, self.password)
            response = self.session.post('https://api.worldquantbrain.com/authentication')
            response.raise_for_status()
            self.last_login = datetime.now()
            logger.info("WorldQuant Brain 登录成功")
            return True
        except Exception as e:
            logger.error(f"登录失败: {e}")
            return False

    def ensure_session(self):
        """检查 Session 是否有效（简单心跳逻辑）"""
        try:
            resp = self.session.get('https://api.worldquantbrain.com/users/self')
            if resp.status_code != 200:
                logger.warning("Session 已失效，正在重新登录...")
                return self.sign_in()
            return True
        except Exception:
            return self.sign_in()


class AlphaWorker:
    def __init__(self):
        self.auth = BrainAuth()
        self.is_running = True

        self.auth.sign_in()
        self.fetcher = DataFetcher(self.auth.session)
        self.llm_agent = LLMGenerator(api_key=os.getenv('API_KEY'))
        self.sim_agent = SimulationAgent(self.auth.session)

        self.sim_settings = {
            "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
            "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY",
            "truncation": 0.01, "pasteurization": "ON", "unitHandling": "VERIFY",
            "nanHandling": "OFF", "language": "FASTEXPR", "visualization": False,
        }

    def fetch_new_tasks(self):
        """
        任务来源逻辑。
        实际应用中，这里可以是从数据库读取、监听消息队列、或者读取一个 JSON 文件。
        现在暂时模拟：每隔一段时间读取 tasks.json。
        """
        if os.path.exists("tasks.json"):
            with open("tasks.json", "r") as f:
                tasks = json.load(f)
            return tasks
        return []

    def process_task(self, idea):
        """单次任务处理流水线"""
        logger.info(f"开始处理思路: {idea}")

        if not self.auth.ensure_session():
            logger.error("无法维持登录状态，跳过当前任务")
            return

        fnd6_df = self.fetcher.get_datafields(dataset_id='fundamental6', search_scope={'region': 'USA'})
        matrix_fields = fnd6_df[fnd6_df['type'] == "MATRIX"]['id'].tolist() if not fnd6_df.empty else ['m_rev']

        alpha_expr = self.llm_agent.generate_alpha(idea, matrix_fields, {"ops": "std"})
        if not alpha_expr:
            logger.warning("LLM 未能生成表达式")
            return

        logger.info(f"提交回测: {alpha_expr}")
        result = self.sim_agent.run_simulation(alpha_expr, self.sim_settings)

        if result and result.get('status') == 'COMPLETE':
            sharpe = result.get('sharpe')
            logger.info(f"任务完成! Sharpe: {sharpe}")
        else:
            logger.error(f"回测失败或超时: {result.get('status') if result else 'Unknown'}")

    def run_forever(self):
        """服务主循环"""
        logger.info("服务已启动，进入监听模式...")

        while self.is_running:
            try:
                tasks = self.fetch_new_tasks()

                if not tasks:
                    time.sleep(10)
                    continue

                for task in tasks:
                    if not self.is_running:
                        break
                    self.process_task(task.get('idea'))
                    time.sleep(5)

                if os.path.exists("tasks.json"):
                    os.remove("tasks.json")

            except Exception as e:
                logger.critical(f"主循环发生未捕获异常: {e}", exc_info=True)
                time.sleep(30)

    def stop(self):
        logger.info("正在关闭服务...")
        self.is_running = False


if __name__ == "__main__":
    worker = AlphaWorker()

    def handle_signal(sig, frame):
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    worker.run_forever()
