import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# 确保 OPENAI_API_KEY 已设置
print("--- 2.1 LLM 与 Chat Model 接口演示 ---")
chat_model = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.9,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)


# 场景一：简单的用户提问
print("\n--- 场景一：简单的用户提问 ---")
user_query = "你好，请用一句话介绍一下 Langchain。"
response = chat_model.invoke([
    HumanMessage(content=user_query)
])
print(f"用户提问: {user_query}")
print(f"ChatModel 响应: {response.content}") # response 是 AIMessage 对象，内容在 .content 属性中


# 场景二：结合系统消息，为模型设定角色或行为
print("\n--- 场景二：结合系统消息，为模型设定角色或行为 ---")
messages_with_system = [
    SystemMessage(content="你是一个专业的编程语言导师，擅长简洁明了地解释概念。"),
    HumanMessage(content="请解释一下 Python 中的装饰器。")
]
response_system_role = chat_model.invoke(messages_with_system)
print(f"设定角色后 ChatModel 响应: {response_system_role.content}")


# 场景三：模拟多轮对话的结构（实际应用中会结合 Memory 模块）
print("\n--- 场景三：模拟多轮对话结构 ---")
# 这里的 messages 列表代表了对话历史
dialog_history = [
    HumanMessage(content="我最喜欢的编程语言是 Python。"),
    AIMessage(content="Python 是一个非常流行的语言！你喜欢它的哪些特性呢？"),
    HumanMessage(content="我喜欢它的简洁性和丰富的库生态。")
]

response_multi_turn = chat_model.invoke(dialog_history)
print(f"ChatModel 响应 (模拟多轮): {response_multi_turn.content}")


from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

print("\n--- 2.2 Prompt Templates 演示 ---")

# 1. 使用 PromptTemplate (适用于旧式LLM，了解即可)
# 作用：将输入变量填充到模板字符串中
poem_template = PromptTemplate(
    input_variables=["topic", "style"],
    template="请给我写一首关于 {topic} 的诗歌，风格是 {style}。",
)

# 格式化 Prompt，传入变量
formatted_poem_prompt = poem_template.format(topic="秋天的落叶", style="伤感")
print(f"格式化后的 Prompt (PromptTemplate):\n{formatted_poem_prompt}")


# 2. 使用 ChatPromptTemplate (推荐，适用于 ChatModel)
# 作用：构建结构化的消息列表，每个消息可以有自己的模板
coding_assistant_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个经验丰富的Python程序员，擅长解答编程问题，并给出代码示例。"),
    HumanMessage(content="请用 {language} 语言，实现一个 {functionality} 的函数。"),
])

# 格式化 ChatPromptTemplate，传入变量，返回一个消息对象列表
messages_for_coding = coding_assistant_template.format_messages(
    language="Python",
    functionality="计算斐波那契数列的第 n 项"
)
print(f"\n格式化后的 Chat Messages (ChatPromptTemplate):")
for msg in messages_for_coding:
    print(f"  {type(msg).__name__}: {msg.content[:50]}...") # 打印部分内容

# 将格式化后的消息发送给 ChatModel
# chat_model 在 2.1 节已初始化
response_code = chat_model.invoke(messages_for_coding)
print(f"\nChatModel 生成的代码解释:\n{response_code.content[:300]}...") # 打印部分响应

# 另一个更复杂的 ChatPromptTemplate 示例：需要JSON格式输出
json_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个数据提取专家，请将用户信息提取为 JSON 格式。"),
    HumanMessage(content="请从以下文本中提取用户的姓名和年龄：\n\n用户资料：我叫李明，今年28岁。"),
    HumanMessage(content="请确保输出为严格的 JSON 格式，例如: `{{\"name\": \"\", \"age\": 0}}`")
])
json_messages = json_prompt.format_messages() # 无需额外变量，直接格式化

response_json = chat_model.invoke(json_messages)
print(f"\nChatModel 生成的 JSON:\n{response_json.content}")


from langchain_core.output_parsers import StrOutputParser # 用于将模型输出解析成字符串

print("\n--- 2.3 LCEL (Chains) 演示 ---")

# 1. 定义 Prompt Template
code_gen_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个专业的 Python 编程助手，擅长清晰地解释概念并提供简洁的代码示例。"),
    HumanMessage(content="请为我解释并提供一个关于 Python '{concept}' 的代码示例。"),
    HumanMessage(content="请确保解释和代码都清晰易懂。")
])

# 2. 初始化 Chat Model
# chat_model 在 2.1 节已初始化

# 3. 定义输出解析器
# StrOutputParser 会把 ChatModel 返回的 AIMessage 对象中的 .content 部分提取出来
output_parser = StrOutputParser()

# 4. 构建 LCEL 链
# 链的输入是一个字典，例如 {"concept": "装饰器"}
# 1. `code_gen_prompt` 接收 {"concept": "..."}，生成 Chat Messages 列表
# 2. `chat_model` 接收 Chat Messages 列表，返回 AIMessage 对象
# 3. `output_parser` 接收 AIMessage 对象，解析成纯字符串
code_explanation_chain = code_gen_prompt | chat_model | output_parser

# 5. 调用链并传入输入变量
input_concept = "生成器"
print(f"正在生成关于 '{input_concept}' 的代码示例和解释...")
result = code_explanation_chain.invoke({"concept": input_concept})

print(f"\n链的输出:\n{result}")

# 另一个链的例子：先将用户输入转换为英文，再进行解释
from langchain_core.runnables import RunnablePassthrough # 用于传递输入

# 步骤1：翻译 Prompt
translate_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个专业的翻译家，将用户输入的中文短语翻译成英文。"),
    HumanMessage(content="请将 '{text}' 翻译成英文。")
])

# 步骤2：英文解释 Prompt
explain_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个专业词汇解释器，用简洁的英文解释词语。"),
    HumanMessage(content="请解释一下 '{english_text}' 这个词汇的含义。")
])

# 构建一个更复杂的链：
# 1. 接收一个包含 "text" 的字典，例如 {"text": "并行计算"}
# 2. 通过 RunnablePassthrough 将 "text" 传给 translate_prompt
# 3. translate_chain 翻译中文到英文，其输出是英文文本
# 4. assign() 将翻译结果添加到字典中，键为 "english_text"
# 5. explain_prompt 接收 "english_text"，生成解释 Prompt
# 6. chat_model 和 output_parser 处理最终的解释
translation_and_explanation_chain = (
    {"text": RunnablePassthrough()} # 接收原始输入并传递
    | translate_prompt 
    | chat_model 
    | output_parser
    .assign(english_text=lambda x: x) # 将翻译结果赋值给 english_text 键
    | explain_prompt 
    | chat_model 
    | output_parser
)

# 注意：上面的 .assign() 只是一个示意，实际复杂的链可能需要更精妙的 RunnableParallel/RunnableMap 来处理多输入
# 这里简化为串行处理，只传递一个主要输入。

# 假设我们需要一个更清晰的多步骤链，来演示输入传递
# Step 1: 翻译
translator_chain = (
    translate_prompt | chat_model | output_parser
)

# Step 2: 解释 (需要上一步的翻译结果)
# 这里的{"english_text": translator_chain} 表示将 translator_chain 的输出作为 english_text 的值
full_pipeline = {
    "english_text": translator_chain,
    "original_text": RunnablePassthrough() # 传递原始的 "text" 输入，供后续步骤使用（如果需要）
} | explain_prompt | chat_model | output_parser

input_text_for_pipeline = "并发编程"
print(f"\n正在处理概念：'{input_text_for_pipeline}'")
explanation_result = full_pipeline.invoke({"text": input_text_for_pipeline})
print(f"解释结果:\n{explanation_result}")