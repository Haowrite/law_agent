from locust import HttpUser, task, between
import json

from locust import HttpUser, task, between
import json

# 测试账号（请确保已存在）
TEST_USERNAME = "hao123"
TEST_PASSWORD = "02crhstc"


class LegalAIChatUser(HttpUser):
    # 每次聊天请求间隔 1~10 秒
    wait_time = between(1, 10)

    def on_start(self):
        """每个用户启动时：登录 + 创建一次会话 + 初始化问题计数器"""
        
        self.user_id = 'c73c3d09-3cd6-4415-ad1b-72b9b1a0e98c'

        # 创建新会话（每个用户仅一次）
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
        
        # 初始化问题计数器
        self.question_count = 0
        
        # 定义5个不同的法律问题
        self.questions = [
            "劳动仲裁的流程是什么？",
            "劳动合同纠纷如何处理？",
            "试用期被无故辞退怎么办？",
            "公司拖欠工资该如何维权？",
            "工伤认定需要哪些材料？"
        ]
        
        print(f"✅ 用户 {self.user_id} 创建会话 {self.session_id}")

    @task
    def chat_once(self):
        """每个任务发送一条消息，循环提问5个不同问题"""
        
        # 如果已经问了5个问题，可以停止或继续循环
        if self.question_count >= 5:
            # 可以选择重置计数器继续循环，或者停止提问
            # 这里我们选择循环提问，重置计数器
            self.question_count = 0
        
        # 获取当前要问的问题
        current_question_index = self.question_count % len(self.questions)
        message = self.questions[current_question_index]
        
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
                        # 成功发送问题后，计数器加1
                        self.question_count += 1
                        print(f"📝 用户 {self.user_id} 已发送问题 {self.question_count}: {message}")
                    else:
                        resp.failure("响应缺少 response 或 session_id 不匹配")
                except Exception as e:
                    resp.failure(f"JSON 解析失败: {e}")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text}")

    # 禁用其他任务（只保留 chat）
    # 注意：Locust 要求至少有一个 @task，我们已满足


    # locust -f locustfile.py --host=http://localhost:5000