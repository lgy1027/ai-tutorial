import os
import json
import random
import jieba
import pdfplumber
from tqdm import tqdm
from openai import OpenAI
from rank_bm25 import BM25Okapi

API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
BASE_URL = "http://10.1.18.99:8089/v1" 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(CURRENT_DIR, "knowledge_base")

OUTPUT_FILE = os.path.join(CURRENT_DIR, "finetune_data_mined.jsonl")

# 挖掘配置
CHUNK_SIZE = 500       # 文本切片长度
NEG_COUNT = 7          # 每个问题挖掘多少个负例
BM25_TOP_K = 20        # BM25 检索候选池大小

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def read_txt(file_path):
    """读取 TXT 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"读取 TXT 失败 {file_path}: {e}")
        return ""

def read_pdf(file_path):
    """读取 PDF 文件"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"读取 PDF 失败 {file_path}: {e}")
    return text

def chunk_text(text, chunk_size=500):
    """对单个文档的内容进行切片"""
    if not text:
        return []
    # 这里做简单的定长切分，生产环境建议使用 RecursiveCharacterTextSplitter
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # 过滤掉过短的无效片段（比如页眉页脚残留）
    return [c.strip() for c in chunks if len(c.strip()) > 50]

def load_data_from_dir(directory):
    """遍历目录加载所有支持的文件"""
    all_chunks = []
    files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.pdf'))]
    
    print(f"发现 {len(files)} 个文件，开始处理...")
    
    for filename in tqdm(files, desc="Loading Files"):
        file_path = os.path.join(directory, filename)
        content = ""
        
        if filename.lower().endswith('.pdf'):
            content = read_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            content = read_txt(file_path)
            
        # 对当前文件进行切片
        file_chunks = chunk_text(content, CHUNK_SIZE)
        all_chunks.extend(file_chunks)
        
    return all_chunks

def generate_queries(chunk_text):
    """利用 LLM 基于文档片段生成问题"""
    prompt = f"""
    你是一个专业的数据集生成助手。请根据以下提供的【政务/业务文档片段】，
    生成 2 个用户可能会问的问题（Query）。
    
    要求：
    1. 问题必须针对文档内容，可以用文档内容回答。
    2. 问题风格要模拟真实用户的提问（口语化、简短）。
    3. 只返回问题列表，每行一个问题，不要包含序号或其他废话。
    
    【文档片段】：
    {chunk_text}
    """
    
    try:
        response = client.chat.completions.create(
            model="Qwen3-235B-A22B", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        queries = [line.strip() for line in content.split('\n') if line.strip()]
        return queries
    except Exception as e:
        print(f"LLM 生成出错: {e}")
        return []

def main():
    # 1. 加载所有文件并切片
    print("1. 正在扫描目录并加载数据...")
    if not os.path.exists(INPUT_DIR):
        print(f"目录不存在: {INPUT_DIR}")
        return

    corpus = load_data_from_dir(INPUT_DIR)
    print(f"数据加载完成，共切分为 {len(corpus)} 个片段。")
    
    if len(corpus) == 0:
        print("未找到有效文本数据，请检查目录。")
        return

    # 2. 构建 BM25 索引（全局索引）
    print("2. 正在构建 BM25 索引（用于硬负例挖掘）...")
    # 对中文进行分词
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    dataset = []

    # 3. 开始循环生成
    print("3. 开始生成问题并挖掘负例...")
    # 限制处理数量用于测试，正式跑可以去掉 [:10]
    for idx, doc_text in tqdm(enumerate(corpus), total=len(corpus), desc="Generating"):
        
        # 生成正例 (Query)
        queries = generate_queries(doc_text)
        
        for query in queries:
            # 挖掘负例 (Hard Negatives)
            tokenized_query = list(jieba.cut(query))
            scores = bm25.get_scores(tokenized_query)
            top_n_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]
            
            hard_negatives = []
            for neg_idx in top_n_indexes:
                # 排除掉原文片段自己
                if neg_idx == idx or corpus[neg_idx] == doc_text:
                    continue
                
                hard_negatives.append(corpus[neg_idx])
                if len(hard_negatives) >= NEG_COUNT:
                    break
            
            # 补齐负例
            retry_count = 0
            while len(hard_negatives) < NEG_COUNT and retry_count < 20:
                random_neg = random.choice(corpus)
                if random_neg != doc_text and random_neg not in hard_negatives:
                    hard_negatives.append(random_neg)
                retry_count += 1

            # 组装数据
            data_item = {
                "query": query,
                "pos": [doc_text],
                "neg": hard_negatives
            }
            dataset.append(data_item)

    # 4. 保存文件
    print(f"4. 正在保存数据到 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"任务完成！训练数据已生成。")

if __name__ == "__main__":
    main()