import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "final_intent_model/final_model")

INTENTS = ["order_query", "addr_modify", "refund_apply", "oos"]

class IntentPredictor:
    def __init__(self, model_path):
        print(f"📦 正在加载模型: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def predict(self, text, threshold=0.5):
        # 1. 预处理
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(self.model.device)

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 3. 多标签核心：使用 Sigmoid 转化为概率
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # 4. 根据阈值筛选意图
        results = []
        for i, p in enumerate(probs):
            if p >= threshold:
                results.append({
                    "intent": INTENTS[i],
                    "confidence": f"{p:.4f}"
                })
        
        # 5. 兜底逻辑：如果所有概率都很低，取最高的一个（可选）
        if not results:
            max_idx = np.argmax(probs)
            results.append({
                "intent": INTENTS[max_idx], 
                "confidence": f"{probs[max_idx]:.4f} (兜底建议)"
            })
            
        return results

if __name__ == "__main__":
    predictor = IntentPredictor(MODEL_PATH)

    # 测试用例：涵盖单意图、多意图、以及模糊意图
    test_cases = [
        # --- 1. 单意图测试 (验证基础准确性) ---
        "快递到哪了？帮我查下单子", "我想看看我的订单物流信息", "单号是多少？货发了吗？",
        "帮我改下收货地址，写错了", "我想把配送地址换成公司", "电话写错了，能帮我改一下吗？",
        "我不想要了，申请退款", "这件衣服质量不好，我要退货退款", "怎么退款啊？操作一下",
        "你们这还有货吗？", "什么时候补货？", "这款还有蓝色的吗？",
        "物流太慢了，帮我催一下", "地址改到北京朝阳区", "我要退钱，不买了",

        # --- 2. 双意图融合 (验证多标签核心能力) ---
        "帮我查下快递到哪了，顺便改个地址", "物流信息还没更新，我想申请退款了",
        "这个没货了吗？如果有货的话帮我改下地址发走", "我要退款，顺带问下那个没货的什么时候到",
        "地址写错了能改吗？顺便查查我另外一个单子发货没", "退款已经申请了，帮我看看还要多久到账",
        "我想买那个缺货的，还有能把现在这个单子地址改了吗？", "查一下物流，要是还没发货我就想退款了",
        "地址改到上海，顺便问下这款还有货吗？", "刚才退款成功了，那我的地址信息还会保存吗？",
        "快递还没到，我想改地址，如果不让改我就退款", "帮我看看单子，顺便改下收件人电话",
        "这款断货了？那帮我把之前那个单子退了吧", "查物流，再帮我改个地址", "退款流程怎么走？顺带看看发货没",

        # --- 3. 三意图及以上复杂情况 (压力测试) ---
        "我想查下物流，如果还没发货帮我改个地址，或者干脆退款算了",
        "这个蓝色没货了吗？那帮我把黑色的地址改一下，顺便看看那个红色的到哪了",
        "我要退款，因为地址填错了改不了，而且我看你们一直没补货",
        "帮我看看物流，顺便改下电话，另外问下这款还有大码吗？",
        "之前那个单子退款了没？顺便改下新单子的地址，再看看物流",
        "没货就直说，我好申请退款，顺便把那个地址改到老家去",
        "快递到哪了？我想改地址，要是改不了我就不要了，直接退款",
        "这款补货了吗？我想买，另外查查我昨天的单子，顺便改个联系方式",
        "地址帮我改下，顺便看看发货没，要是断货了记得提醒我退款",
        "退款、改地址、查物流，这三个功能在哪里操作？",

        # --- 4. 干扰项 (验证 OOS/闲聊 识别能力) ---
        "你好，有人在吗？", "今天天气不错啊", "你们几点下班？",
        "能不能给我便宜点？", "你是机器人吗？", "我想听首歌",
        "哈哈哈哈太逗了", "投诉你们物流太慢！", "转人工服务",
        "谢谢你，帮了大忙了", "你们公司在哪？", "再见，拜拜",
        "给我推荐个好吃的", "这多少钱？", "我心情不好"
    ]

    print("\n" + "="*50)
    print(" ModernBERT 多意图识别测试开始")
    print("="*50)

    for text in test_cases:
        res = predictor.predict(text)
        print(f"\n 输入: {text}")
        for r in res:
            print(f"   识别到意图: 【{r['intent']}】 (置信度: {r['confidence']})")

    print("\n" + "="*50)