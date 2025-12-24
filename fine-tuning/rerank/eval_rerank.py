import json
import os
import torch
from sentence_transformers import CrossEncoder

BASE_MODEL_NAME = "BAAI/bge-reranker-large" 
FT_MODEL_PATH = "./output_rerank_final"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_rerank")
EVAL_FILE = os.path.join(DATA_DIR, "eval_golden.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_eval_cases():
    """
    加载评测案例：
    1. 包含 2 个固定的经典硬负例 Case
    2. 尝试从 eval_golden.jsonl 加载生成的案例
    """
    cases = []
    
    # 固定测试实例
    cases.append({
        "query": "高层次人才购房补贴标准是多少？",
        "pos": "对A类人才，给予最高300万元的购房补贴，分5年发放。",
        "neg": "人才公寓租赁管理办法规定，租金标准按照市场价的70%执行。" # 干扰项：都有人才、钱、标准，但意思是租房
    })
    cases.append({
        "query": "企业研发费用加计扣除申报流程",
        "pos": "企业需在税务系统中填报《研发费用加计扣除优惠明细表》进行申报。",
        "neg": "关于高新技术企业认定管理办法的通知，企业需提交研发费用专项审计报告。" # 干扰项：都有研发、企业，但一个是申报一个是认定
    })

    # --- B. 加载生成的黄金数据 ---
    if os.path.exists(EVAL_FILE):
        print(f"发现评测集: {EVAL_FILE}，正在加载...")
        with open(EVAL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 只取第一个正例和第一个最硬的负例进行对比
                    if len(data['pos']) > 0 and len(data.get('neg', [])) > 0:
                        cases.append({
                            "query": data['query'],
                            "pos": data['pos'][0],
                            "neg": data['neg'][0] 
                        })
                except:
                    continue
        print(f"总计加载 {len(cases)} 条测试数据 (含 2 条固定演示数据)")
    else:
        print("未找到 eval_golden.jsonl，仅使用固定演示案例。")
    
    return cases

def compare_models():
    print("\n" + "="*60)
    print("正在初始化模型 ...")
    print("="*60)
    
    print(f"加载基座: {BASE_MODEL_NAME} ...")
    try:
        base_model = CrossEncoder(BASE_MODEL_NAME, max_length=512, device=DEVICE)
    except Exception as e:
        print(f"基座模型加载失败: {e}")
        return

    print(f"加载微调: {FT_MODEL_PATH} ...")
    if not os.path.exists(FT_MODEL_PATH):
        print(f"找不到微调模型路径: {FT_MODEL_PATH}，请先运行训练脚本！")
        return
    
    try:
        # 微调模型加载不需要传 max_length，它会读取 config.json
        ft_model = CrossEncoder(FT_MODEL_PATH, device=DEVICE)
    except Exception as e:
        print(f"微调模型加载失败: {e}")
        print("提示：如果报错 Unrecognized configuration，请检查 train 脚本是否成功生成了 config.json")
        return

    # 加载数据
    test_cases = load_eval_cases()
    
    print("\n" + "="*60)
    print("Rerank 模型巅峰对决")
    print("="*60)

    for i, case in enumerate(test_cases):
        # 限制只展示前 5 条 + 后 2 条，防止刷屏（如果数据很多）
        if i >= 5 and i < len(test_cases) - 2:
            continue
            
        query = case['query']
        pos = case['pos']
        neg = case['neg']
        
        # 截断长文本用于显示
        pos_display = (pos[:40] + '...') if len(pos) > 40 else pos
        neg_display = (neg[:40] + '...') if len(neg) > 40 else neg
        
        # 构造输入对
        pair_pos = [query, pos]
        pair_neg = [query, neg]
        
        # --- A. 基座打分 ---
        scores_base = base_model.predict([pair_pos, pair_neg])
        score_pos_base = scores_base[0]
        score_neg_base = scores_base[1]
        gap_base = score_pos_base - score_neg_base
        
        # --- B. 微调打分 ---
        scores_ft = ft_model.predict([pair_pos, pair_neg])
        score_pos_ft = scores_ft[0]
        score_neg_ft = scores_ft[1]
        gap_ft = score_pos_ft - score_neg_ft
        
        # --- C. 打印报告 ---
        print(f"\n[Case {i+1}] 提问: {query}")
        print(f"   正确: {pos_display}")
        print(f"   干扰: {neg_display}")
        print("-" * 40)
        
        # 格式化输出基座结果
        status_base = "危险" if gap_base < 1.0 else "尚可"
        if score_neg_base > score_pos_base: status_base = "排序错误"
        print(f"基座模型 | 正例: {score_pos_base:7.4f} | 负例: {score_neg_base:7.4f} | Gap: {gap_base:7.4f} [{status_base}]")
        
        # 格式化输出微调结果
        status_ft = "优秀" if gap_ft > gap_base else "持平"
        if score_neg_ft > score_pos_ft: status_ft = "翻车"
        print(f"微调模型 | 正例: {score_pos_ft:7.4f} | 负例: {score_neg_ft:7.4f} | Gap: {gap_ft:7.4f} [{status_ft}]")
        
        # 结论判定
        if score_neg_ft < score_neg_base - 1.0:
            print("   结论: 成功识别并压制了干扰项！")
        elif gap_ft > gap_base + 1.0:
             print("   结论: 显著拉开了正负例差距！")

if __name__ == "__main__":
    compare_models()