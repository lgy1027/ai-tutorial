from langchain_core.messages import HumanMessage, AIMessage,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import List, Any
from langchain_openai import ChatOpenAI
import os
import tiktoken
from dotenv import load_dotenv; load_dotenv()

llm = ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

def basis():
    # 简单的消息历史存储
    class InMemoryChatMessageHistory(BaseChatMessageHistory):
        def __init__(self):
            self.messages = []

        def add_message(self, message):
            self.messages.append(message)

        def clear(self):
            self.messages = []

    # 实例化并添加消息
    history = InMemoryChatMessageHistory()
    history.add_message(HumanMessage(content="你好！"))
    history.add_message(AIMessage(content="你好！有什么可以帮助你的吗？"))
    history.add_message(HumanMessage(content="你是谁？"))

    print("--- ChatMessageHistory 示例 ---")
    for msg in history.messages:
        print(f"{msg.type}: {msg.content}")
    print("\n")


    buffer_memory = ConversationBufferMemory()

    buffer_memory.save_context({"input": "我的名字是小明。"}, {"output": "好的，小明。很高兴认识你！"})
    buffer_memory.save_context({"input": "你知道我叫什么吗？"}, {"output": "我知道，你叫小明。"})

    # load_memory_variables 会返回一个字典，其中包含对话历史
    # 默认键是 "history"
    print("--- ConversationBufferMemory 示例 ---")
    print(buffer_memory.load_memory_variables({}))
    # 你会看到 history 字段包含了所有对话，格式为字符串
    # output: {'history': 'Human: 我的名字是小明。\nAI: 好的，小明。很高兴认识你！\nHuman: 你知道我叫什么吗？\nAI: 我知道，你叫小明。'}
    print("\n")


    # 只保留最近1轮对话 (即只保留当前轮和上一轮)
    window_memory = ConversationBufferWindowMemory(k=1)

    window_memory.save_context({"input": "第一句话。"}, {"output": "这是第一句话的回复。"})
    window_memory.save_context({"input": "第二句话。"}, {"output": "这是第二句话的回复。"})
    window_memory.save_context({"input": "第三句话。"}, {"output": "这是第三句话的回复。"})

    print("--- ConversationBufferWindowMemory (k=1) 示例 ---")
    print(window_memory.load_memory_variables({}))
    # output: {'history': 'Human: 第三句话。\nAI: 这是第三句话的回复。'}
    # 注意，只有最后一句对话被保留了
    print("\n")

    summary_memory = ConversationSummaryMemory(llm=llm)

    summary_memory.save_context({"input": "我最喜欢的颜色是蓝色。"}, {"output": "蓝色很棒！"})
    summary_memory.save_context({"input": "你知道天空为什么是蓝色的吗？"}, {"output": "天空看起来是蓝色是因为瑞利散射。"})
    summary_memory.save_context({"input": "那大海呢？"}, {"output": "大海的蓝色则是由水的固有性质以及对光线的吸收和散射造成的。"})

    print("--- ConversationSummaryMemory 示例 ---")
    print(summary_memory.load_memory_variables({}))
    # 你会看到 history 字段中包含了LLM生成的摘要和最新的对话
    # 摘要可能类似："用户提到他最喜欢的颜色是蓝色，询问天空和大海为何呈现蓝色，AI解释了瑞利散射和水的吸收散射。"
    print("\n")

    # 1. 创建自定义的 Chat Model 类，我本地是vllm部署的模型，计算token方面需要重写方法
    class VllmChatModel(ChatOpenAI):
        
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
            """根据消息列表计算 token 数量。"""
            text = ""
            for message in messages:
                text += message.content
            return len(self._tokenizer.encode(text))
        
        @property
        def _llm_type(self) -> str:
            return "vllm-chat-model"

    llm_token = VllmChatModel(
        model=os.environ.get("OPENAI_MODEL"),
        temperature=0.9,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # 设置最大Token数为 50 （实际会略有浮动，因为是估算）
    token_memory = ConversationTokenBufferMemory(
        llm=llm_token, 
        max_token_limit=50,
    )

    token_memory.save_context({"input": "第一句非常长的对话内容，包含了多个词语，这样才能体现Token的限制作用。"}, {"output": "第一句回复。"})
    token_memory.save_context({"input": "第二句相对较短的对话内容。"}, {"output": "第二句回复。"})
    token_memory.save_context({"input": "第三句简短的话。"}, {"output": "第三句回复。"})

    print("--- ConversationTokenBufferMemory 示例 ---")
    print(token_memory.load_memory_variables({}))
    # 你会发现，可能只有最后几句话被保留了，因为前几句话的Token数已经超出了50的限制
    # print(cb) # 如果是LLM调用，可以看到Token数
    print("\n")


def agent():
    # 1. 定义一个用于存储会话历史的字典 (模拟持久化存储，实际应用中会是数据库/Redis)
    # key 是 session_id, value 是 BaseChatMessageHistory 实例
    store = {}

    # 2. 实现 get_session_history 函数
    # 这个函数会根据 session_id 返回对应的 ChatMessageHistory 对象
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            # 这里我们使用 ConversationBufferWindowMemory，只保留最近3轮对话
            # 你也可以根据需要替换为 ConversationBufferMemory 或其他记忆类型
            store[session_id] = ConversationBufferWindowMemory(
                k=3, # 保持3轮记忆
                return_messages=True, # 以消息对象列表形式返回，适合MessagesPlaceholder
                output_key="output", input_key="input" # 定义输入输出的key
            ).chat_memory # 获取底层的 ChatMessageHistory 对象
        return store[session_id]

    # 3. 定义一个普通的 LCEL 链条 (不带记忆)
    # 注意 PromptTemplate 中使用 MessagesPlaceholder 来占位历史消息
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，请根据上下文进行对话。"),
        MessagesPlaceholder(variable_name="history"), # 这是记忆会填充的地方
        ("user", "{input}") # 用户当前输入
    ])

    # 核心业务逻辑链 (这是我们希望包装成有记忆的链条)
    # input -> prompt (加入history和input) -> LLM -> output parser
    base_conversation_chain = prompt | llm | StrOutputParser()

    # 4. 使用 RunnableWithMessageHistory 包装你的 LCEL 链条
    with_message_history_chain = RunnableWithMessageHistory(
        base_conversation_chain, # 你要包装的 LCEL 链条
        get_session_history, # 会话历史获取函数
        input_messages_key="input", # 链条的输入中，哪个键是用户消息
        history_messages_key="history", # 提示模板中，哪个键是历史消息的占位符
        output_messages_key="output" # 定义输出的key，以便save_context能够工作
    )

    # 5. 进行多轮对话 (需要传入 config 参数，其中包含 session_id)
    print("--- 具有记忆的聊天机器人示例 ---")
    session_id_user1 = "user_123"

    # 第一轮对话
    response1 = with_message_history_chain.invoke(
        {"input": "我的名字是张三。"},
        config={"configurable": {"session_id": session_id_user1}}
    )
    print(f"用户1 (轮1): 我的名字是张三。")
    print(f"AI (轮1): {response1}")

    # 第二轮对话 (同一个 session_id)
    response2 = with_message_history_chain.invoke(
        {"input": "你知道我叫什么吗？"},
        config={"configurable": {"session_id": session_id_user1}}
    )
    print(f"用户1 (轮2): 你知道我叫什么吗？")
    print(f"AI (轮2): {response2}") # AI应该能回答出“张三”

    # 第三轮对话 (同一个 session_id)
    response3 = with_message_history_chain.invoke(
        {"input": "我喜欢吃苹果，不喜欢吃香蕉。"},
        config={"configurable": {"session_id": session_id_user1}}
    )
    print(f"用户1 (轮3): 我喜欢吃苹果，不喜欢吃香蕉。")
    print(f"AI (轮3): {response3}")

    # 第四轮对话 (同一个 session_id)
    response4 = with_message_history_chain.invoke(
        {"input": "我刚才说了我喜欢什么水果？"},
        config={"configurable": {"session_id": session_id_user1}}
    )
    print(f"用户1 (轮4): 我刚才说了我喜欢什么水果？")
    print(f"AI (轮4): {response4}") # AI应该能回答出“苹果”

    # 切换到另一个用户 (不同的 session_id)
    session_id_user2 = "user_456"
    response_user2_1 = with_message_history_chain.invoke(
        {"input": "你好，我是李四。"},
        config={"configurable": {"session_id": session_id_user2}}
    )
    print(f"\n用户2 (轮1): 你好，我是李四。")
    print(f"AI (轮1): {response_user2_1}")

    response_user2_2 = with_message_history_chain.invoke(
        {"input": "我喜欢什么水果？"}, # 故意问这个，看AI是否会混淆
        config={"configurable": {"session_id": session_id_user2}}
    )
    print(f"用户2 (轮2): 我喜欢什么水果？")
    print(f"AI (轮2): {response_user2_2}") # AI应该不知道，因为这是新会话


if __name__ == "__main__":
    # basis()
    agent()