import torch
from transformers import BertTokenizer, BertModel

class NBert(nn.Module):

    def __init__(self):
        model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)


    def get_bert_embeddings(self, input):
        # 加载预训练的BERT模型和分词器
        model_name = 'bert-base-uncased'

        # 对句子进行编码
        inputs = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 将模型设置为评估模式
        self.model.eval()

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs
