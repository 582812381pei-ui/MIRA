import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """ 一个标准的多头注意力模块 """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        # 1. 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 拆分为多头
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用 Softmax 得到注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 5. 加权求和
        context = torch.matmul(attention_weights, v)

        # 6. 合并多头
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 7. 最终线性投影
        return self.out_proj(context)


class FeedForward(nn.Module):
    """ 一个简单的 FFN 模块 """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))


class MultimodalAspectFusion(nn.Module):
    def __init__(self, d_model: int = 1152, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()

        # 阶段一：方面引导交叉注意力 (图像 Q, 方面词 K/V)
        self.guidance_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 阶段二：情感提取交叉注意力 (方面词 Q, 引导后的图像 K/V)
        self.extraction_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 一个标准的前馈网络来增强表示能力
        self.ffn = FeedForward(d_model, d_model * 4, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, image_hidden, aspect_pool):
        """
        Args:
            image_hidden (torch.Tensor): 图像 patch 特征, shape [batch, 576, 1152]
            aspect_pool (torch.Tensor): 方面词池化特征, shape [batch, 1152]
        """
        # --- 准备工作 ---
        # Cross-Attention 需要序列格式的输入, 所以为 aspect_pool 增加一个序列长度维度 (1)
        # [batch, 1152] -> [batch, 1, 1152]
        aspect_hidden = aspect_pool.unsqueeze(1)

        # --- 阶段一: 方面引导 (生成方面感知的图像表示) ---
        # Q: 图像的每个 patch (image_hidden)
        # K, V: 方面词的全局表示 (aspect_hidden)
        # 含义: "图像的每个部分，都来咨询一下方面词，看看自己和它有多相关"
        guided_img = self.guidance_cross_attn(query=image_hidden, key=aspect_hidden, value=aspect_hidden)

        # 添加残差连接和层归一化 (标准 Transformer 操作)
        image_aware = self.norm1(image_hidden + self.dropout(guided_img))

        # --- 阶段二: 情感提取 (从引导后的图像中提取方面相关的情感) ---
        # Q: 方面词的全局表示 (aspect_hidden)
        # K, V: 刚刚生成的、方面感知的图像表示 (image_aware)
        # 含义: "方面词，现在请审视这幅已经被你'高亮'过的图像，并提取出核心信息"
        extracted_visual_feature = self.extraction_cross_attn(query=aspect_hidden, key=image_aware, value=image_aware)

        # 再次添加残差连接和层归一化
        fused_feature = self.norm2(aspect_hidden + self.dropout(extracted_visual_feature))

        # --- FFN 增强 ---
        # 通过一个前馈网络进一步处理融合后的特征
        ffn_output = self.ffn(fused_feature)
        final_output = self.norm3(fused_feature + self.dropout(ffn_output))

        # --- 输出 ---
        # 去掉我们之前添加的序列维度，得到最终的融合特征向量
        # [batch, 1, 1152] -> [batch, 1152]
        return final_output.squeeze(1)


# --- 示例用法 ---
if __name__ == '__main__':
    BATCH_SIZE = 4
    NUM_PATCHES = 576
    D_MODEL = 1152

    # 创建符合您描述的伪数据
    image_features = torch.randn(BATCH_SIZE, NUM_PATCHES, D_MODEL)
    aspect_features = torch.randn(BATCH_SIZE, D_MODEL)

    # 初始化我们的融合模块
    fusion_module = MultimodalAspectFusion(d_model=D_MODEL, num_heads=12)

    # 执行前向传播
    final_fused_feature = fusion_module(image_features, aspect_features)

    # 检查输出形状
    print("输入图像特征形状:", image_features.shape)
    print("输入方面词特征形状:", aspect_features.shape)
    print("---")
    print("最终融合特征形状:", final_fused_feature.shape)

    # 这个 final_fused_feature 就是可以送入最终分类器进行情感判断的特征向量
    # 例如：
    # classifier = nn.Linear(D_MODEL, 3) # 3分类：正面/负面/中性
    # logits = classifier(final_fused_feature)
    # print("分类器输出形状:", logits.shape)
