import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IntentPredictor:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.id2label = {0: "order_query", 1: "refund_apply", 2: "addr_modify", 3: "human_service", 4: "chitchat"}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred_id = torch.max(probs, dim=-1)
        
        return {
            "intent": self.id2label[pred_id.item()],
            "confidence": conf.item(),
            "text": text
        }

# 测试环节
predictor = IntentPredictor("./final_intent_model")

test_cases = [
    "那个订单我想退了",     # 典型的退款意图
    "地址填错了怎么改？",    # 修改地址
    "帮我转接人工，快点",   # 人工服务
    "今天天气真不错"        # 闲聊
]

for case in test_cases:
    res = predictor.predict(case)
    print(f"输入: {res['text']} | 识别意图: {res['intent']} | 置信度: {res['confidence']:.4f}")