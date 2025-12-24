import os
import json
import random
import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from openai import OpenAI
import pypdf

API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
BASE_URL = "http://10.1.18.99:8089/v1" 


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(CURRENT_DIR, "knowledge_base") # 支持 txt, md, pdf 的文件夹（支持嵌套目录）

OUTPUT_DIR = os.path.join(CURRENT_DIR, "dataset_rerank")

os.makedirs(OUTPUT_DIR, exist_ok=True)

NEG_COUNT = 5       # 1个正例 + 4个硬负例
BM25_TOP_K = 15     # 扩大搜索范围，确保挖到高质量负例

# 切片配置 (PDF提取出的文本通常较长，需要切片)
CHUNK_SIZE = 300    # 每个切片的字符数
OVERLAP = 50        # 切片重叠长度

# 数据集切分比例
SPLIT_RATIO = [0.8, 0.1, 0.1] # 训练 : 验证 : 测试
GOLDEN_EVAL_COUNT = 20        # 预留 20 条做黄金评测

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_text_from_pdf(file_path):
    """解析 PDF 文本"""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"读取 PDF 失败 {file_path}: {e}")
    return text

def simple_chunker(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """简单的滑动窗口切片"""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        # 过滤掉太短的碎片（比如页眉页脚）
        if len(chunk) > 50: 
            chunks.append(chunk)
        start += (size - overlap)
    return chunks

def load_documents_recursive(data_dir):
    """递归加载目录下所有支持的文件"""
    docs = []
    print(f"正在扫描目录: {data_dir} ...")
    
    supported_ext = ('.txt', '.md', '.pdf')
    
    file_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(supported_ext):
                file_list.append(os.path.join(root, file))
    
    print(f"   发现 {len(file_list)} 个文件，开始解析...")

    for file_path in tqdm(file_list, desc="Parsing Files"):
        content = ""
        # 1. 根据后缀解析
        if file_path.lower().endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
        else:
            # TXT / MD
            try:
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"读取文本失败 {file_path}: {e}")
        
        # 2. 进行切片
        if content:
            file_chunks = simple_chunker(content)
            docs.extend(file_chunks)

    print(f"解析完成，共生成 {len(docs)} 个文档片段 (Chunks)。")
    return docs

def generate_query_via_llm(doc_text):
    """调用 LLM 生成搜索 Query"""
    # 截断一下防止 Token 溢出，只取切片的前 800 字符作为上下文
    # context = doc_text[:800]
    
    prompt = f"""
    你是一个构建搜索数据集的专家。请根据下面的【文档片段】，构造一个用户可能会在搜索引擎中输入的【查询问题】(Query)。
    
    要求：
    1. 问题必须可以用该文档片段回答。
    2. 问题要口语化、真实，不要太长。
    3. 严禁包含"根据文档"、"文中提到"等字眼。
    4. 只返回问题文本，不要任何前缀。

    【文档片段】：
    {doc_text}
    """
    try:
        response = client.chat.completions.create(
            model="Qwen3-235B-A22B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM 调用错误: {e}")
        return None

def build_dataset():
    corpus = load_documents_recursive(INPUT_DIR)
    if not corpus: 
        print("未找到有效文档，请检查路径。")
        return

    # 构建 BM25 索引
    print("正在构建 BM25 索引 (用于挖掘硬负例)...")
    # 简单的分词，实际生产可以用更复杂的
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    all_data = []

    # 生成流水线
    print(f"开始生成 Q-A 对及挖掘负例 (预计耗时较长)...")
    # 为了演示，如果你想快速测试，可以把 corpus[:50] 限制数量
    for idx, doc in tqdm(enumerate(corpus), total=len(corpus), desc="Generating"):
        
        # 生成 Query
        query = generate_query_via_llm(doc)
        if not query: continue

        # 挖掘硬负例
        tokenized_query = list(jieba.cut(query))
        scores = bm25.get_scores(tokenized_query)
        # 取 Top-K
        top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_TOP_K]

        hard_negatives = []
        for neg_idx in top_indexes:
            candidate = corpus[neg_idx]
            # 排除原文本身，且去重
            if candidate != doc and candidate not in hard_negatives:
                hard_negatives.append(candidate)
            # 凑够数量就停
            if len(hard_negatives) >= (NEG_COUNT - 1):
                break
        
        # 如果硬负例不够，用随机负例凑
        while len(hard_negatives) < NEG_COUNT:
            rand_doc = random.choice(corpus)
            if rand_doc != doc and rand_doc not in hard_negatives:
                hard_negatives.append(rand_doc)

        # Rerank 格式标准
        item = {
            "query": query,
            "pos": [doc],
            "neg": hard_negatives[:NEG_COUNT] # 确保数量一致
        }
        all_data.append(item)

    # 切分数据集
    print("正在切分 Train / Val / Test / Eval ...")
    random.shuffle(all_data)
    
    # 评测集 (人工 check 用)
    golden_data = all_data[:GOLDEN_EVAL_COUNT]
    rest_data = all_data[GOLDEN_EVAL_COUNT:]

    total = len(rest_data)
    train_end = int(total * SPLIT_RATIO[0])
    val_end = train_end + int(total * SPLIT_RATIO[1])

    train_set = rest_data[:train_end]
    val_set = rest_data[train_end:val_end]
    test_set = rest_data[val_end:]

    def save_jsonl(data, name):
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"   {name}: {len(data)} 条")

    save_jsonl(train_set, "train.jsonl")
    save_jsonl(val_set, "val.jsonl")
    save_jsonl(test_set, "test.jsonl")
    save_jsonl(golden_data, "eval_golden.jsonl")

    print(f"\n数据集生成完毕！请查看目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    if API_KEY.startswith("sk-"):
        build_dataset()
    else:
        print("请先在脚本开头配置 API_KEY")