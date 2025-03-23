import streamlit as st
import json
import warnings
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np

# 屏蔽警告信息
warnings.filterwarnings("ignore")

# 定义标签映射

label1_list = ["I10.x00x032", "I20.000", "I20.800x007", "I21.401", "I50.900x018"]
label12id = {label: idx for idx, label in enumerate(label1_list)}
id2label1 = {idx: label for label, idx in label12id.items()}

label2_list = [
    "E04.101", "E04.102", "E11.900", "E14.900x001", "E72.101", "E78.500", "E87.600",
    "I10.x00x023", "I10.x00x024", "I10.x00x027", "I10.x00x028", "I10.x00x031", "I10.x00x032",
    "I20.000", "I25.102", "I25.103", "I25.200", "I31.800x004", "I38.x01", "I48.x01", "I48.x02",
    "I49.100x001", "I49.100x002", "I49.300x001", "I49.300x002", "I49.400x002", "I49.400x003",
    "I49.900", "I50.900x007", "I50.900x008", "I50.900x010", "I50.900x014", "I50.900x015",
    "I50.900x016", "I50.900x018", "I50.907", "I63.900", "I67.200x011", "I69.300x002",
    "I70.203", "I70.806", "J18.900", "J98.414", "K76.000", "K76.807", "N19.x00x002",
    "N28.101", "Q24.501", "R42.x00x004", "R91.x00x003", "Z54.000x033", "Z95.501", "Z98.800x612"
]
label22id = {label: idx for idx, label in enumerate(label2_list)}
id2label2 = {idx: label for label, idx in label22id.items()}

# 定义 MultiTaskBERT 模型

class MultiTaskBERT(nn.Module):
    def __init__(self, pretrained_model_name, num_labels1, num_labels2):
        super().__init__()
        base_model = BertModel.from_pretrained(pretrained_model_name)
        # 使用前6层作为共享部分
        self.shared_bert = nn.ModuleList(base_model.encoder.layer[:6])
        # 后6层分别作为单标签和多标签任务的专用部分
        self.single_bert = nn.ModuleList([nn.ModuleList(base_model.encoder.layer[6:])[i] for i in range(6)])
        self.multi_bert = nn.ModuleList([nn.ModuleList(base_model.encoder.layer[6:])[i] for i in range(6)])
        self.embeddings = base_model.embeddings
        self.dropout = nn.Dropout(0.3)
        # 单标签任务
        self.single_head = nn.Linear(768, num_labels1)
        # 多标签任务
        self.multi_head = nn.Linear(768, num_labels2)
        self.log_sigma1 = nn.Parameter(torch.tensor([-1.0]))
        self.log_sigma2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        shared_output = embedding_output
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        for layer in self.shared_bert:
            shared_output = layer(shared_output, extended_attention_mask)[0]
        # 单任务分支
        single_output = shared_output
        for layer in self.single_bert:
            single_output = layer(single_output, extended_attention_mask)[0]
        # 多任务分支
        multi_output = shared_output
        for layer in self.multi_bert:
            multi_output = layer(multi_output, extended_attention_mask)[0]
        single_cls = self.dropout(single_output[:, 0, :])
        multi_cls = self.dropout(multi_output[:, 0, :])
        single_logits = self.single_head(single_cls)
        multi_logits = self.multi_head(multi_cls)
        return single_logits, multi_logits
    
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask
        return attention_mask

# 界面初始化及模型加载

st.title("测试集病历预测系统")

# 加载 tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 设置运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
num_labels1 = len(label1_list)
num_labels2 = len(label2_list)
model = MultiTaskBERT("bert-base-chinese", num_labels1, num_labels2)
model.to(device)

# 加载训练好的模型参数
model_path = "./best_model/best_model.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    st.success(f"模型参数加载成功：{model_path}")
except Exception as e:
    st.error(f"加载模型失败：{e}")

# 加载测试数据

test_data_path = "./test_data/test_data.json"
try:
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    st.success(f"成功加载测试数据，共 {len(test_data)} 条记录")
except Exception as e:
    st.error(f"加载测试数据失败：{e}")
    test_data = []

# 展示测试集病历列表，并选择进行预测

st.header("测试集病历列表")
if test_data:
    # 显示每条记录的 text1 部分前50个字符作为简要描述
    sample_options = {f"样本 {i+1}: {item['text1'][:50]}..." : i for i, item in enumerate(test_data)}
    selected_key = st.selectbox("选择一条病历记录进行预测", list(sample_options.keys()))
    selected_index = sample_options[selected_key]
    
    sample = test_data[selected_index]
    st.write("### 病历详情")
    st.write("**病案标识:**", sample['ID'])
    st.write("**文本1:**", sample['text1'])
    st.write("**文本2:**", sample['text2'])
    st.write("**主要诊断编码:**", sample['label1'])
    st.write("**其他诊断编码:**", sample['label2'])
    
    # 定义针对单条样本预测的函数
    def predict_sample(sample):
        text1 = sample["text1"]
        text2 = sample["text2"]
        encoding = tokenizer(
            text1,
            text2,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # 将各项tensor移动到对应设备上
        for key in encoding:
            encoding[key] = encoding[key].to(device)
        model.eval()
        with torch.no_grad():
            logits1, logits2 = model(encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"])
            # 主要诊断预测：取概率最大的标签
            pred_label1_idx = torch.argmax(logits1, dim=1).item()
            pred_label1 = id2label1[pred_label1_idx]
            # 其他诊断预测：sigmoid后阈值0.5判断多标签
            pred_logits2 = torch.sigmoid(logits2)
            pred_label2_binary = (pred_logits2 > 0.5).int().squeeze(0).cpu().numpy()
            pred_label2 = [id2label2[i] for i, val in enumerate(pred_label2_binary) if val == 1]
        return pred_label1, pred_label2

    if st.button("进行预测"):
        pred1, pred2 = predict_sample(sample)
        st.write("### 预测结果")
        st.write("**预测主要诊断 (label1):**", pred1)
        st.write("**预测其他诊断 (label2):**", pred2)
else:
    st.write("没有加载到测试数据。")