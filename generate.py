import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # ==========================================
        # 维度符号定义 (Dimension Legend):
        # B = batch_size (批次大小)
        # T = current sequence length (当前上下文实际拥有的序列长度，每轮循环后会 +1)
        # K = context_length (模型允许的最大记忆/滑窗窗口长度)
        # V = vocab_size (全词表大小 / 总唯一字符数)
        # ==========================================

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        
        # 用于累积生成字符的字符串
        generated_text = ""
        print (context)
        
        # 初始输入的 context 形状: (B, T)
        for i in range(new_chars):
            # 1. 裁剪上下文 (防止自注意力机制的序列长度突破模型训练上限 K)
            # cropped_context 形状: (B, min(T, K))
            # - 如果 T <= K: 形状维持 (B, T)，原封不动
            # - 如果 T > K: 形状死死卡在 (B, K)，滑动窗口永远只取最尾部最新的 K 个词
            cropped_context = context[:, -context_length:]
            
            # 2. 前向传播获取全局预测分数
            # logits 形状: (B, min(T, K), V)
            logits = model(cropped_context)
            
            # 提取最后一个时间步的置信度得分 (我们只关心基于当前最后一个词，预测其后面的“下一位新成员”)
            # last_token_logits 形状: (B, V)
            last_token_logits = logits[:, -1, :]  
            print(last_token_logits.shape)
            
            # 转化为合法的归一化概率分布 (让整行加起来严格等于 1)
            # probs 形状: (B, V)
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # 3. 依据概率分布进行多项式随机抽样 (摇号掷骰子)
            # next_token 形状: (B, 1)
            next_token = torch.multinomial(probs, 1, generator=generator)

            # 评测系统固定代码：重置随机状态以确保评测结果可复现
            generator.set_state(initial_state)
            
            # 4. 将新生成的 Token 在时间轴（列维度 dim=1）上拼接到已有序列的末尾
            # 拼接操作: (B, T) 和 (B, 1) 进行横向拼接
            # 拼接后 context 的新形状: (B, T + 1)
            context = torch.cat((context, next_token), dim=1)
            
            # 5. 将包含单个元素的 PyTorch 张量剥离为纯粹的 Python 原生整数
            # token_id 形状: 0-D 标量 (Scalar / 纯 Python int)
            token_id = next_token.item()
            
            # 将整数映射回字符并累加
            generated_text += int_to_char[token_id]

        return generated_text