from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import asyncio
import os

load_dotenv()
model = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL"),
    temperature=0.9,
    base_url=os.environ.get("OPENAI_BASE_URL"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的简短笑话。")
chain = prompt | model | StrOutputParser()

print("--- 1. .invoke()：单次同步调用 ---")
joke_sync = chain.invoke({"topic": "编程"})
print(f"编程笑话: {joke_sync}\n")

print("--- 2. .stream()：流式调用 ---")
print("小猫笑话 (逐字输出):")
for chunk in chain.stream({"topic": "小猫"}):
    print(chunk, end="", flush=True)
print("\n")

print("--- 3. .batch()：批量调用 ---")
topics = ["大象", "咖啡", "侦探"]
jokes_batch = chain.batch([{"topic": t} for t in topics]) # 传入一个字典列表
print("批量生成的笑话:")
for i, joke in enumerate(jokes_batch):
    print(f"关于'{topics[i]}'的笑话: {joke}")
print("\n")

# --- 4. .ainvoke()：异步调用示例 ---
async def run_async_example():
    print("--- .ainvoke()：异步调用 ---")
    joke_async = await chain.ainvoke({"topic": "外星人"})
    print(f"外星人笑话 (异步): {joke_async}")

# if __name__ == "__main__":
#     asyncio.run(run_async_example()) # 运行异步函数

# 示例：简单的文本处理管道
from langchain_core.runnables import RunnableLambda

# 自定义一个简单的Runnable，将文本转为大写
to_uppercase = RunnableLambda(lambda text: text.upper())
# 再自定义一个Runnable，添加感叹号
add_exclamation = RunnableLambda(lambda text: text + "!!!")

simple_pipeline = StrOutputParser() | to_uppercase | add_exclamation

print("--- 管道操作符 | 示例 ---")
processed_text = simple_pipeline.invoke("hello world")
print(f"处理后的文本: {processed_text}\n") # 输出：HELLO WORLD!!!

from langchain_core.runnables import RunnablePassthrough

# 例子：计算文本长度，并将长度信息和原始文本一起传递给LLM

# 链条：原始输入 (例如 {"text": "..."})
# -> RunnablePassthrough.assign(length=lambda x: len(x["text"]))
#    (现在输入变成了 {"text": "...", "length": 123})
# -> Prompt (可以同时使用 {text} 和 {length})

length_and_text_prompt = ChatPromptTemplate.from_template(
    "原始文本: {text}\n文本长度: {length}\n请根据原始文本，总结其主要观点。"
)

# 这里的 RunnablePassthrough.assign() 动态地计算了 'length' 并加入了输入字典
# 原始的 'text' 仍然存在并被传递下去
chain_with_length = (
    RunnablePassthrough.assign(
        length=lambda input_dict: len(input_dict["text"])
    )
    | length_and_text_prompt
    | model
    | StrOutputParser()
)

print("--- RunnablePassthrough.assign() 示例 ---")
result = chain_with_length.invoke({"text": "LangChain让构建大模型应用变得如此简单，大大提高了开发效率！"})
print(f"总结与长度信息: {result}\n")

from langchain_core.runnables import RunnableParallel

# 例子：同时让LLM做文本总结和关键词提取
summarize_chain = ChatPromptTemplate.from_template("总结以下文本:\n{text}") | model | StrOutputParser()
keywords_chain = ChatPromptTemplate.from_template("从以下文本中提取3-5个关键词（用逗号分隔）:\n{text}") | model | StrOutputParser()

# 使用 RunnableParallel 将两个链并行化
# 注意：这里的输入 {text} 会被同时传递给 summarize_chain 和 keywords_chain
parallel_processor = RunnableParallel(
    summary=summarize_chain,
    keywords=keywords_chain
)

text_to_analyze = "区块链技术是一种去中心化的分布式账本技术，广泛应用于加密货币领域，其核心特点是不可篡改性和透明性。"

print("--- RunnableParallel 示例 ---")
results = parallel_processor.invoke({"text": text_to_analyze})
print(f"总结:\n{results['summary']}")
print(f"关键词:\n{results['keywords']}\n")


from langchain_core.runnables import chain

@chain
def clean_text(text: str) -> str:
    """一个自定义Runnable，清理文本中的多余空格并转换为小写。"""
    return text.lower().strip().replace("  ", " ")

@chain
def censor_words(text: str, words_to_censor: list) -> str:
    """一个自定义Runnable，审查特定词汇。"""
    for word in words_to_censor:
        text = text.replace(word, "[CENSORED]")
    return text

custom_logic_chain = (
    ChatPromptTemplate.from_template("请处理这段文本:\n{text_input}")
    | model
    | StrOutputParser()
    | clean_text # 使用自定义函数Runnable
    | censor_words.bind(words_to_censor=["糟糕", "问题"]) # 绑定参数给自定义函数
)

print("--- 自定义 Runnable (函数式) 示例 ---")
original_input = "  这是一个 糟糕的 测试文本，存在 一些 问题。  "
processed_output = custom_logic_chain.invoke({"text_input": original_input})
print(f"原始文本: '{original_input}'")
print(f"处理后文本: '{processed_output}'\n")


from langchain_core.runnables import Runnable
from pydantic import BaseModel,Field

class WordCounter(Runnable):
    def __init__(self, target_word: str):
        self.target_word = target_word.lower()

    def invoke(self, input_dict: dict, config=None) -> dict:
        """计算目标词在文本中出现的次数"""
        input_text = input_dict["text_input"]
        count = input_text.lower().count(self.target_word)
        return {"original_text": input_text, "word_count": count, "target": self.target_word}

    # 可以选择定义输入/输出Schema，有助于类型检查和工具使用
    class InputSchema(BaseModel):
        text_input: str = Field(..., description="需要计数的文本")
    class OutputSchema(BaseModel):
        original_text: str = Field(..., description="原始文本")
        word_count: int = Field(..., description="目标词出现次数")
        target: str = Field(..., description="目标词")
    
    def get_input_schema(self, config=None):
        return self.InputSchema
    def get_output_schema(self, config=None):
        return self.OutputSchema

word_count_chain = (
    RunnablePassthrough.assign(text_input=lambda x: x["text"]) # 从字典中提取文本
    | WordCounter(target_word="代码") # 使用自定义类Runnable
    | ChatPromptTemplate.from_template(
        "在原始文本 '{original_text}' 中，词语 '{target}' 出现了 {word_count} 次。这是一个准确的计数。"
    )
    | model
    | StrOutputParser()
)

print("--- 自定义 Runnable (类式) 示例 ---")
text_for_count = {"text": "LangChain让写代码变得更有趣，代码质量也更高了。"}
result_word_count = word_count_chain.invoke(text_for_count)
print(result_word_count)


# 例子：绑定 temperature 让模型输出更具创意
creative_chain = ChatPromptTemplate.from_template("写一个关于{topic}的诗歌。") | model.bind(temperature=0.8) | StrOutputParser()
strict_chain = ChatPromptTemplate.from_template("总结以下文本的核心要点:\n{text}") | model.bind(temperature=0.0) | StrOutputParser()

print("\n--- .bind() 示例 ---")
print("创意诗歌:")
print(creative_chain.invoke({"topic": "星空"}))
print("\n严格总结:")
print(strict_chain.invoke({"text": "LCEL是LangChain Expression Language的缩写，是LangChain的核心。它提供了一种声明式的方式来构建链条，并通过管道操作符实现组件的连接。LCEL的优势在于其模块化、可组合性和易于调试的特性。"}))

# 示例：为链条添加一个Tag，方便LangSmith追踪
# 假设 chain 是你在上面定义的某个链
tagged_chain = chain.with_config({"tags": ["my_first_tag", "demo_chain"]})
# 当你运行 tagged_chain.invoke() 时，这些tags会出现在LangSmith的追踪记录中

print("\n--------- 综合案例：构建智能内容摘要与关键词提取器---------")

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List # 导入 List
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, chain
from pydantic import BaseModel,Field

load_dotenv()
model = ChatOpenAI(model="Qwen3-235B-A22B", temperature=0.3)
parser = StrOutputParser()

# --- 1. 定义总结的链 ---
summarize_prompt = ChatPromptTemplate.from_template("请对以下文本进行简洁的摘要，不超过100字:\n{text}")
summarize_chain = summarize_prompt | model | parser

# --- 2. 定义提取关键词的链 ---
# 定义 Pydantic 模型来表示关键词列表的输出结构
class Keywords(BaseModel):
    keywords: List[str] = Field(description="一组关键词")

keywords_prompt = ChatPromptTemplate.from_template(
    "从以下文本中提取5个最重要的关键词，以JSON数组格式输出，例如: [\"词1\", \"词2\"]:\n{text}"
)
# 使用 with_structured_output 绑定 schema，确保模型输出JSON
keywords_chain = keywords_prompt | model.with_structured_output(
    schema=Keywords
)

# --- 3. 自定义Runnable，用于文本预处理 ---
@chain
def preprocess_text(text: str) -> str:
    """移除文本中的多余换行符和一些特殊字符。"""
    cleaned = text.replace("\n", " ").replace("  ", " ").replace("。", ".").strip()
    return cleaned

# --- 构建总的LCEL链条 ---
# 整个流程需要输入一个 {text_to_process}
full_content_analyzer = (
    RunnablePassthrough.assign(
        processed_text=lambda x: preprocess_text.invoke(x["text"]), # 明确传递 x["text"] 给 preprocess_text
    )
    | RunnableParallel( # 并行执行摘要和关键词提取
        summary=summarize_chain.with_config(run_name="SummaryGeneration"), # 给子链命名，方便LangSmith追踪
        keywords=keywords_chain.with_config(run_name="KeywordExtraction")
    )
    # 最终输出是 {"summary": "...", "keywords": ["...", "..."]}
)

# --- 运行案例 ---
long_text = """
LangChain是一个非常强大的框架，旨在帮助开发者更高效地构建基于大语言模型（LLMs）的应用程序。
它的核心理念是将复杂的LLM应用分解为模块化的组件（称为Runnables），然后通过LCEL（LangChain Expression Language）将这些组件像管道一样连接起来。

LCEL提供了极其灵活的数据流控制能力，包括并行处理和条件分支。
这使得开发者能够轻松实现各种复杂的AI应用，例如：
1. 问答系统：结合外部知识库（RAG）。
2. 聊天机器人：管理对话历史和上下文。
3. Agent：让LLM具备自主决策和工具调用能力。

LangChain强调LCEL的简洁性和可组合性，以及与LangSmith工具的深度集成，
LangSmith可以帮助开发者进行调试、测试和评估，大大提升了开发体验。
"""

print("--- 智能内容摘要与关键词提取 ---")
analysis_results = full_content_analyzer.invoke({"text": long_text})

print("------ 结果 ------")
print(f"**摘要:**\n{analysis_results['summary']}")
print(f"\n**关键词:**")
print(f"DEBUG: type of keywords: {type(analysis_results['keywords'])}")
print(f"DEBUG: content of keywords: {analysis_results['keywords']}")
print(f"{', '.join(analysis_results['keywords'].keywords)}")
print("------------------")