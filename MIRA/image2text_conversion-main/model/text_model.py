
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel,AutoModel
# Construct the joint Caption & Sentiment Classifier Model

class OnlyTextModel_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(self.cfg.model) # bert model
        # self.bert = BertModel.from_pretrained(self.cfg.model) # bert model
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        # self.project = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        # 分类器接收拼接后的向量 (V_target + D_agg)
        self.project = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 6, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            # nn.Linear(self.bert_hidden_size, 3) # 假设3分类
        )

        self.project_1 = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            # nn.Linear(self.bert_hidden_size, 3) # 假设3分类
        )

        # 门控，控制候选信息的注入强度（保守初始化）
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.fuse_gate[-2].weight, 0.0)  # 使初期 gate 偏小
        nn.init.constant_(self.fuse_gate[-2].bias, -2.0)


        self.dropout = nn.Dropout(self.cfg.dropout)

        # self.attention = TransformerLayer(self.bert.config.hidden_size, num_heads=2, dropout=cfg.dropout,max_relative_position=4)
        self._init_weights(self.out)
        self._init_weights(self.project)
        self._init_weights(self.project_1)
        self._init_weights(self.norm)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,input_ids,attention_mask,candidate_input_ids,candidate_attention_mask):

        outputs_target = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        v_target = outputs_target.pooler_output
        if candidate_input_ids.ndim != 2:

            # --- 2a: 编码所有候选方面词 ---
            batch_size, len_candidates, max_len = candidate_input_ids.shape

            # 变形 -> (batch_size * num_candidates, max_len)
            reshaped_candidate_ids = candidate_input_ids.view(-1, max_len)
            reshaped_candidate_mask = candidate_attention_mask.view(-1, max_len)

            outputs_interferes = self.bert(input_ids=reshaped_candidate_ids, attention_mask=reshaped_candidate_mask)

            # 获取池化输出并恢复形状
            # pooler_output_interferes 形状: (batch_size * num_candidates, hidden_size)
            pooler_output_interferes = outputs_interferes.pooler_output
            # v_interferes 形状: (batch_size, num_candidates, hidden_size)
            v_interferes = pooler_output_interferes.view(batch_size, len_candidates, -1)

            # --- 2b: 计算差分向量 ---
            # 使用广播机制: v_target.unsqueeze(1) -> (batch_size, 1, hidden_size)
            # diff_vectors 形状: (batch_size, num_candidates, hidden_size)
            diff_vectors = v_target.unsqueeze(1) - v_interferes
            diff_squared = diff_vectors ** 2

            candidate_features = torch.cat([diff_vectors, diff_squared, v_interferes], dim=-1)
            # --- 2c: 使用MeanPooling聚合差分向量 ---
            # d_agg 形状: (batch_size, hidden_size)
            # d_agg = torch.mean(diff_vectors, dim=1)
            d_agg_max, _ = torch.max(candidate_features, dim=1)
            d_agg_mean = torch.mean(candidate_features, dim=1)

            d_agg = torch.cat([d_agg_max, d_agg_mean], dim=-1)

            d_agg = self.project(d_agg)


        else:
            # --- 2d: 如果没有干扰项，则总差分向量为零向量 ---
            d_agg = torch.zeros_like(v_target)

        # --- 步骤 3: 融合与分类 ---
        # 拼接目标向量和聚合后的总差分向量
        # v_fused 形状: (batch_size, hidden_size * 2)
        v_fused = torch.cat([v_target, d_agg], dim=-1)

        v_fused = self.project_1(v_fused)

        gate = self.fuse_gate(v_fused)

        v_fused = self.norm(gate * v_fused  + v_target )

        # gate = self.fuse_gate(v_fused)

        # v_fused = self.norm(v_fused  + v_target )

        logits = self.out(self.dropout(v_fused))

        return logits, v_fused # 返回logits和融合后的特征



class EnhancedFilterModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        gate = self.gate(x)
        out = gate * x
        return out
class OnlyTextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(self.cfg.model_large, add_pooling_layer=False)  # 🚫 不加载自带 pooler
        # self.bert = BertModel.from_pretrained(self.cfg.model) # bert model
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, 3) # 3 classes
        # self.project = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        # 分类器接收拼接后的向量 (V_target + D_agg)
        self.project = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 6, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            # nn.Linear(self.bert_hidden_size, 3) # 假设3分类
        )

        self.project_1 = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            # nn.Linear(self.bert_hidden_size, 3) # 假设3分类
        )

        self.self_gate = EnhancedFilterModule(hidden_size=self.bert.config.hidden_size)

        # 推荐的最终方案
        # 门控，控制候选信息的注入强度（保守初始化）
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

        # 初始化现在指向正确的层 (index 0)，并且逻辑上也是正确的
        nn.init.constant_(self.fuse_gate[0].weight, 0.0)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)


        self.dropout = nn.Dropout(self.cfg.dropout)

        # ✅ 手动实现 pooler（BERT 原版是一个 Linear + Tanh）
        self.pooler = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()

        # self.attention = TransformerLayer(self.bert.config.hidden_size, num_heads=2, dropout=cfg.dropout,max_relative_position=4)
        self._init_weights(self.out)
        self._init_weights(self.project)
        self._init_weights(self.project_1)
        self._init_weights(self.norm)
        self._init_weights(self.pooler)
        self.self_gate.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,input_ids,attention_mask,candidate_input_ids,candidate_attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # ✅ 手动提取 CLS 向量
        cls_token = last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # ✅ 手动 Pooler：Linear + Tanh（与 BERT 原始实现一致）
        tweet_cls = self.activation(self.pooler(cls_token))  # [batch, hidden_size]

        # Dropout + 分类
        v_target = self.dropout(tweet_cls)
        if candidate_input_ids.ndim != 2:

            # --- 2a: 编码所有候选方面词 ---
            batch_size, len_candidates, max_len = candidate_input_ids.shape

            # 变形 -> (batch_size * num_candidates, max_len)
            reshaped_candidate_ids = candidate_input_ids.view(-1, max_len)
            reshaped_candidate_mask = candidate_attention_mask.view(-1, max_len)

            outputs_interferes = self.bert(input_ids=reshaped_candidate_ids, attention_mask=reshaped_candidate_mask)
            last_hidden_state_interferes = outputs_interferes.last_hidden_state  # [batch, seq_len, hidden]
            # ✅ 手动提取 CLS 向量
            cls_token_interferes = last_hidden_state_interferes[:, 0, :]  # [batch, hidden_size]

            # ✅ 手动 Pooler：Linear + Tanh（与 BERT 原始实现一致）
            tweet_cls_interferes = self.activation(self.pooler(cls_token_interferes))  # [batch, hidden_size]

            # Dropout + 分类
            pooler_output_interferes = self.dropout(tweet_cls_interferes)

            # 获取池化输出并恢复形状
            # pooler_output_interferes 形状: (batch_size * num_candidates, hidden_size)

            # v_interferes 形状: (batch_size, num_candidates, hidden_size)
            v_interferes = pooler_output_interferes.view(batch_size, len_candidates, -1)

            # --- 2b: 计算差分向量 ---
            # 使用广播机制: v_target.unsqueeze(1) -> (batch_size, 1, hidden_size)
            # diff_vectors 形状: (batch_size, num_candidates, hidden_size)
            diff_vectors = v_target.unsqueeze(1) - v_interferes
            diff_squared = diff_vectors ** 2

            candidate_features = torch.cat([diff_vectors, diff_squared, v_interferes], dim=-1)
            # --- 2c: 使用MeanPooling聚合差分向量 ---
            # d_agg 形状: (batch_size, hidden_size)
            # d_agg = torch.mean(diff_vectors, dim=1)
            d_agg_max, _ = torch.max(candidate_features, dim=1)
            d_agg_mean = torch.mean(candidate_features, dim=1)

            d_agg = torch.cat([d_agg_max, d_agg_mean], dim=-1)

            d_agg = self.project(d_agg)


        else:
            # --- 2d: 如果没有干扰项，则总差分向量为零向量 ---
            d_agg = torch.zeros_like(v_target)

        # --- 步骤 3: 融合与分类 ---
        # 拼接目标向量和聚合后的总差分向量
        # v_fused 形状: (batch_size, hidden_size * 2)

        d_agg = self.self_gate(d_agg)

        v_fused = d_agg + v_target

        gate = self.fuse_gate(v_fused)

        v_fused = self.norm(gate * v_fused  + v_target)

        logits = self.out(self.dropout(v_fused))

        return logits, v_fused # 返回logits和融合后的特征


