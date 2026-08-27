# -*- coding: utf-8 -*-
"""
模型定义 - MatSciBERT (纯BERT，无CRF)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from config import NUM_LABELS, MODEL_NAME, DROPOUT_RATE


class MatSciBERTCRF(nn.Module):
    """
    BERT 模型用于命名实体识别（无CRF层）
    
    结构:
        1. BERT: 提取上下文特征
        2. Dropout: 防止过拟合
        3. Linear: 将BERT输出映射到标签空间
    """
    
    def __init__(self, num_labels=NUM_LABELS, model_name=MODEL_NAME, dropout_rate=DROPOUT_RATE):
        super(MatSciBERTCRF, self).__init__()
        
        self.num_labels = num_labels
        
        # 加载预训练的MatSciBERT
        # add_pooling_layer=False 避免加载不需要的pooler层，消除警告
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类层
        self.classifier = nn.Linear(self.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len), 可选，用于计算损失
        
        Returns:
            如果提供labels: 返回(loss, logits)
            否则: 返回logits
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        # 映射到标签空间
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        
        if labels is not None:
            # 训练模式：计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 只计算非padding位置的损失
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            return loss, logits
        else:
            # 推理模式：返回logits
            return logits
    
    def get_logits(self, input_ids, attention_mask):
        """获取logits分数（用于调试）"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        emissions = self.classifier(sequence_output)
        return emissions


def get_model(device='cuda'):
    """创建并返回模型实例"""
    model = MatSciBERTCRF()
    
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    
    model = model.to(device)
    return model, device


if __name__ == '__main__':
    # 测试模型
    print("创建模型...")
    model, device = get_model()
    
    print(f"\n模型结构:")
    print(f"  BERT hidden size: {model.hidden_size}")
    print(f"  Number of labels: {model.num_labels}")
    print(f"  Device: {device}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    seq_len = 128
    
    dummy_input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(device)
    dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)
    dummy_labels = torch.randint(0, NUM_LABELS, (batch_size, seq_len)).to(device)
    
    # 训练模式
    model.train()
    loss = model(dummy_input_ids, dummy_attention_mask, dummy_labels)
    print(f"  Loss: {loss.item():.4f}")
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input_ids, dummy_attention_mask)
    print(f"  Predictions shape: {len(predictions)} x {len(predictions[0])}")
    
    print("\n模型测试通过!")
