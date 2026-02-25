from locust import HttpUser, task, between
import json

# 测试账号（请确保已存在）
TEST_USERNAME = "hao123"
TEST_PASSWORD = "02crhstc"


class LegalAIChatUser(HttpUser):
    # 每次聊天请求间隔 1~2 秒
    wait_time = between(1, 10)

    def on_start(self):
        """每个用户启动时：登录 + 创建一次会话"""

        self.user_id = 'c73c3d09-3cd6-4415-ad1b-72b9b1a0e98c'

        # 2. 创建新会话（每个用户仅一次）
        new_session_resp = self.client.post(
            "/api/new_session",
            params={"user_id": self.user_id},
            name="NEW_SESSION"
        )
        if new_session_resp.status_code != 200:
            print(f"❌ 创建会话失败: {new_session_resp.text}")
            self.environment.runner.quit()
            return

        self.session_id = new_session_resp.json()["session_id"]
        print(f"✅ 用户 {self.user_id} 创建会话 {self.session_id}")

    @task
    def chat_once(self):
        """每个任务发送一条消息（可多次执行）"""
        message = f"劳动仲裁的流程是什么？"
        with self.client.post(
            "/api/chat",
            json={
                "message": message,
                "session_id": self.session_id
            },
            params={"user_id": self.user_id},
            name="CHAT_MESSAGE",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if "response" in data and data["session_id"] == self.session_id:
                        resp.success()
                    else:
                        resp.failure("响应缺少 response 或 session_id 不匹配")
                except Exception as e:
                    resp.failure(f"JSON 解析失败: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")

    def _get_ts(self):
        import time
        return int(time.time())

    # 禁用其他任务（只保留 chat）
    # 注意：Locust 要求至少有一个 @task，我们已满足


    # from locust import HttpUser, task, between
import json

# 测试账号（请确保已存在）
TEST_USERNAME = "hao123"
TEST_PASSWORD = "02crhstc"


class LegalAIChatUser(HttpUser):
    # 每次聊天请求间隔 1~2 秒
    wait_time = between(1, 3)

    def on_start(self):
        """每个用户启动时：登录 + 创建一次会话"""

        self.user_id = 'c73c3d09-3cd6-4415-ad1b-72b9b1a0e98c'

        # 2. 创建新会话（每个用户仅一次）
        new_session_resp = self.client.post(
            "/api/new_session",
            params={"user_id": self.user_id},
            name="NEW_SESSION"
        )
        if new_session_resp.status_code != 200:
            print(f"❌ 创建会话失败: {new_session_resp.text}")
            self.environment.runner.quit()
            return

        self.session_id = new_session_resp.json()["session_id"]
        print(f"✅ 用户 {self.user_id} 创建会话 {self.session_id}")

    @task
    def chat_once(self):
        """每个任务发送一条消息（可多次执行）"""
        message = f"劳动仲裁的流程是什么？"
        with self.client.post(
            "/api/chat",
            json={
                "message": message,
                "session_id": self.session_id
            },
            params={"user_id": self.user_id},
            name="CHAT_MESSAGE",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if "response" in data and data["session_id"] == self.session_id:
                        resp.success()
                    else:
                        resp.failure("响应缺少 response 或 session_id 不匹配")
                except Exception as e:
                    resp.failure(f"JSON 解析失败: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")

    def _get_ts(self):
        import time
        return int(time.time())

    # 禁用其他任务（只保留 chat）
    # 注意：Locust 要求至少有一个 @task，我们已满足


    # locust -f locustfile.py --host=http://localhost:5000