import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertModel, AutoModel


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

# ...existing code...
class FiLMFromState(nn.Module):
    """
    用前一时刻的“推理状态 S”调制基础查询 Q，生成动态查询。
    """
    def __init__(self, D_text, D_state, input_dropout=0.1):
        super().__init__()
        self.param_generator = nn.Sequential(
            nn.Linear(D_state, D_text * 2),
            nn.ReLU(),
            nn.Linear(D_text * 2, D_text * 2)
        )
        self.D_text = D_text
        self.in_drop = nn.Dropout(input_dropout)

        # init: 第一层 Xavier，最后一层 0，使初期 gamma≈0、beta≈0
        nn.init.xavier_uniform_(self.param_generator[0].weight)
        nn.init.zeros_(self.param_generator[0].bias)
        nn.init.zeros_(self.param_generator[-1].weight)
        nn.init.zeros_(self.param_generator[-1].bias)

    def forward(self, base_query, state):
        params = self.param_generator(self.in_drop(state))
        gamma, beta = params[:, :self.D_text], params[:, self.D_text:]
        gamma = torch.tanh(gamma) * 0.5
        return (1 + gamma) * base_query + beta

class GatedCrossAttentionAlign(nn.Module):
    def __init__(self, D_model, n_heads=8, use_norm=True, proj_dim=None, attn_dropout=0.1, res_dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=D_model, num_heads=n_heads, batch_first=True, dropout=attn_dropout
        )
        self.use_norm = use_norm
        self.norm_q = nn.LayerNorm(D_model) if use_norm else nn.Identity()
        self.norm_attn = nn.LayerNorm(D_model) if use_norm else nn.Identity()

        self.proj_dim = proj_dim or D_model
        self.proj_q = nn.Linear(D_model, self.proj_dim)
        self.proj_attn = nn.Linear(D_model, self.proj_dim)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.proj_dim * 2, self.proj_dim // 2),
            nn.ReLU(),
            nn.Linear(self.proj_dim // 2, 1),
            nn.Sigmoid()
        )
        # 将门控最后一层(线性)的偏置初始化为负值，使 gate 初期更小(≈0.12)
        nn.init.constant_(self.gate_mlp[-2].weight, 0.0)
        nn.init.constant_(self.gate_mlp[-2].bias, -2.0)

        self.dropout = nn.Dropout(res_dropout)
        self.norm1 = nn.LayerNorm(D_model)
        self.ffn = nn.Sequential(
            nn.Linear(D_model, D_model * 4), nn.GELU(), nn.Linear(D_model * 4, D_model)
        )
        self.norm2 = nn.LayerNorm(D_model)

    def forward(self, Q, V):
        # print(Q.shape,V.shape)

        attn_output, _ = self.multihead_attn(query=Q, key=V, value=V)
        Q_proj = self.proj_q(self.norm_q(Q))
        A_proj = self.proj_attn(self.norm_attn(attn_output))
        gate_input = torch.cat([Q_proj, A_proj], dim=-1)
        gate = self.gate_mlp(gate_input)
        gated_addition = Q + gate * attn_output
        gated_output = self.norm1(self.dropout(gated_addition))
        ffn_output = self.ffn(gated_output)
        output = self.norm2(gated_output + self.dropout(ffn_output))
        return output.squeeze(1)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network, as seen in Llama, PaLM, and other modern models.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # 根据 Llama 等模型的实践，中间层的维度通常设置为 d_model 的 4 倍再乘以 2/3
        hidden_dim = int(2 * (d_model * 4) / 3)
        # 确保 hidden_dim 是一个可以被硬件优化的值，例如 256 的倍数
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 核心操作
        gate = self.w_gate(x)
        up = self.w_up(x)
        fused = F.silu(gate) * up

        # 投影回原始维度
        output = self.w_down(fused)

        return self.dropout(output)

class CrossAttentionAlign(nn.Module):
    def __init__(self, D_model, n_heads=8, use_norm=True, proj_dim=None, attn_dropout=0.3, res_dropout=0.3):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=D_model, num_heads=n_heads, batch_first=True, dropout=attn_dropout
        )
        self.use_norm = use_norm

        self.proj_dim = proj_dim or D_model


        self.dropout = nn.Dropout(res_dropout)
        self.norm1 = nn.LayerNorm(D_model)
        self.ffn = SwiGLUFFN(d_model=D_model, dropout=res_dropout)
        self.norm2 = nn.LayerNorm(D_model)

    def forward(self, Q, V):
        # print(Q.shape,V.shape)

        attn_output, _ = self.multihead_attn(query=Q, key=V, value=V)

        gated_addition = Q + attn_output
        gated_output = self.norm1(self.dropout(gated_addition))
        # 3. FFN
        ffn_output = self.ffn(gated_output)

        # 4. Add & Norm (Post-LN style) for FFN
        # 注意：因为 SwiGLUFFN 内部已经包含了 dropout，所以这里不再需要 self.dropout(ffn_output)
        output = self.norm2(gated_output + ffn_output)
        return output.squeeze(1)



class ReasoningStepCellVar(nn.Module):
    """
    单步：动态查询 -> 门控跨模态对齐 -> GRUCell 更新状态 -> 状态正则（LN/Dropout）
    """
    def __init__(self, D_model, D_state, n_heads=8, state_dropout=0.1, use_layernorm=True):
        super().__init__()
        self.query_generator = FiLMFromState(D_model, D_state)
        self.aligner = CrossAttentionAlign(D_model, n_heads)
        self.state_updater = nn.GRUCell(input_size=D_model, hidden_size=D_state)
        # GRU 正交/零偏置初始化，稳定收敛
        for name, p in self.state_updater.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        self.use_layernorm = use_layernorm
        self.state_norm = nn.LayerNorm(D_state) if use_layernorm else nn.Identity()
        self.state_dropout = nn.Dropout(state_dropout) if state_dropout > 0 else nn.Identity()

    def forward(self, prev_state, V, base_query):
        dyn_q = self.query_generator(base_query, prev_state).unsqueeze(1)
        feat_t = self.aligner(dyn_q, V)
        next_state = self.state_updater(feat_t, prev_state)
        next_state = self.state_norm(next_state)
        next_state = self.state_dropout(next_state)
        return next_state, feat_t

class DenseFeedbackFeatureExtractor(nn.Module):
    """
    仅提取多步对齐特征，可配置聚合策略（mean/max/last/attn/concat）。
    返回:
        {
            "fused": 聚合后的特征 (默认拼接方面向量),
            "step_features": 每一步的特征栈 [B, steps, D_model],
            "final_state": 最后一步的推理状态 [B, D_state]
        }
    """

    def __init__(self,
                 D_text_global, D_text_aspect, D_vision, D_model, D_state,
                 n_heads=8,
                 max_steps=6,
                 fuse_mode="max",  # mean / max / last / attn / concat
                 state_dropout=0.3,
                 use_state_layernorm=True,
                 project_concat=True  # concat 模式是否额外线性映射回 D_model
                 ):
        super().__init__()

        self.text_proj_aspect = nn.Sequential(
            nn.Linear(D_text_aspect, D_model),
            nn.GELU(),
            nn.LayerNorm(D_model),
            nn.Dropout(state_dropout),
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(D_vision, D_model),
            nn.GELU(),
            nn.LayerNorm(D_model),
            nn.Dropout(state_dropout),
        )

        self.cell = ReasoningStepCellVar(
            D_model, D_state, n_heads,
            state_dropout=state_dropout,
            use_layernorm=use_state_layernorm,

        )

        self.max_steps = max_steps
        self.fuse_mode = fuse_mode.lower()
        if self.fuse_mode == "concat":
            out_dim = D_model * max_steps + D_model
            self.concat_proj = nn.Linear(out_dim, D_model) if project_concat else None
        elif self.fuse_mode == "attn":
            self.attn_score = nn.Sequential(
                nn.Linear(D_model, D_model // 2),
                nn.ReLU(),
                nn.Linear(D_model // 2, 1)
            )
    def forward(self, image_features, global_text_features, aspect_text_features):
        V = self.vision_proj(image_features)
        T_aspect = self.text_proj_aspect(aspect_text_features)

        Q_global = global_text_features

        state = T_aspect
        feats = []
        for t in range(self.max_steps):
            base_query = Q_global if (t % 2 == 0) else T_aspect
            state, feat_t = self.cell(state, V, base_query)
            feats.append(feat_t)

        feat_stack = torch.stack(feats, dim=1)
        fused = self._fuse_features(feat_stack, Q_global)
        return fused
# ...existing code...
    def _fuse_features(self, feat_stack, Q_global):
        """
        feat_stack: [B, steps, D_model]
        aspect_vec: [B, D_model]
        """
        if self.fuse_mode == "mean":
            fused = feat_stack.mean(dim=1)

        elif self.fuse_mode == "max":
            fused, _ = feat_stack.max(dim=1)


        elif self.fuse_mode == "attn":
            scores = self.attn_score(feat_stack)  # [B,steps,1]
            weights = torch.softmax(scores, dim=1)  # [B,steps,1]
            fused = (weights * feat_stack).sum(dim=1)  # [B,Dm]
        elif self.fuse_mode == "concat":
            concat_feat = torch.cat(
                [feat_stack.reshape(feat_stack.size(0), -1), Q_global],
                dim=-1
            )  # [B, steps*Dm + Dm]
            fused = F.relu(self.concat_proj(concat_feat)) if hasattr(self, "concat_proj") and self.concat_proj else concat_feat
        else:
            raise ValueError(f"Unsupported fuse_mode: {self.fuse_mode}")
        return fused

class ClipModel(nn.Module):
    def __init__(self,args,clip_base_model):
        super(ClipModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(self.args.model_large, add_pooling_layer=False)  # 🚫 不加载自带 pooler

        self.clip_base_model=clip_base_model

        self.extractor = DenseFeedbackFeatureExtractor(
        D_text_global=1024,
        D_text_aspect=1152,
        D_vision=1152,
        D_model=1024,
        D_state=1024,
        n_heads=2,
        max_steps=2,
        fuse_mode="concat",      # 可换成 mean/max/last/concat
        state_dropout=self.args.dropout,
        use_state_layernorm=True
        )

        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.self_gate = EnhancedFilterModule(hidden_size=self.bert.config.hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.dropout)
        )

        # 推荐的最终方案
        # 门控，控制候选信息的注入强度（保守初始化）
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

        # 初始化现在指向正确的层 (index 0)，并且逻辑上也是正确的
        nn.init.constant_(self.fuse_gate[0].weight, 0.0)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)

        self.dropout = nn.Dropout(self.args.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3)  # 3 classes

        # ✅ 手动实现 pooler（BERT 原版是一个 Linear + Tanh）
        self.pooler = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()


        self._init_weights(self.norm)
        self._init_weights(self.fusion)
        self._init_weights(self.out)
        self.self_gate.apply(self._init_weights)
        self.extractor.apply(self._init_weights)
        self._init_weights(self.pooler)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)





    def forward(self,clip_image, clip_text,clip_aspect,input_ids,attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # ✅ 手动提取 CLS 向量
        cls_token = last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # ✅ 手动 Pooler：Linear + Tanh（与 BERT 原始实现一致）
        tweet_cls = self.activation(self.pooler(cls_token))  # [batch, hidden_size]

        # Dropout + 分类
        pooler_output = self.dropout(tweet_cls)


        image_pool,image_hidden = self.clip_base_model.encode_image(clip_image,normalize=True)  # 257 维度 第一个是cls
        # text_pool= self.clip_base_model.encode_text(clip_text,normalize=True)
        aspect_pool=self.clip_base_model.encode_text(clip_aspect,normalize=True)
        # image_features=F.relu(self.image_Layer(image_features))
        # text_features=F.relu(self.text_layer(text_features))
        # aspect_features=F.relu(self.aspect_Layer(aspect_features))


        image_outputs = self.extractor(
            image_features=image_hidden,
            global_text_features=pooler_output.detach(),
            aspect_text_features=aspect_pool
        )

        image_outputs = self.self_gate(image_outputs)

        combined_features = image_outputs+pooler_output


        gate = self.fuse_gate(combined_features)

        # 4️⃣ 计算相似度
        similarity = (image_pool * aspect_pool).sum(dim=-1, keepdim=True)  # [B, 1]

        text_probs = torch.sigmoid(
            similarity * self.clip_base_model.logit_scale.exp() + self.clip_base_model.logit_bias
        )

        combined_features = text_probs * gate * combined_features + pooler_output

        outputs = self.out(self.dropout(combined_features))

        return outputs, image_outputs


class ClipModel(nn.Module):
    def __init__(self,args,clip_base_model):
        super(ClipModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(self.args.model_large, add_pooling_layer=False)  # 🚫 不加载自带 pooler

        self.clip_base_model=clip_base_model

        self.extractor = DenseFeedbackFeatureExtractor(
        D_text_global=1024,
        D_text_aspect=1152,
        D_vision=1152,
        D_model=1024,
        D_state=1024,
        n_heads=2,
        max_steps=2,
        fuse_mode="concat",      # 可换成 mean/max/last/concat
        state_dropout=self.args.dropout,
        use_state_layernorm=True
        )

        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.self_gate = EnhancedFilterModule(hidden_size=self.bert.config.hidden_size)

        self.fusion = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.dropout)
        )

        # 推荐的最终方案
        # 门控，控制候选信息的注入强度（保守初始化）
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1),
            nn.Sigmoid()
        )

        # 初始化现在指向正确的层 (index 0)，并且逻辑上也是正确的
        nn.init.constant_(self.fuse_gate[0].weight, 0.0)
        nn.init.constant_(self.fuse_gate[0].bias, -2.0)

        self.dropout = nn.Dropout(self.args.dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 3)  # 3 classes

        # ✅ 手动实现 pooler（BERT 原版是一个 Linear + Tanh）
        self.pooler = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()


        self._init_weights(self.norm)
        self._init_weights(self.fusion)
        self._init_weights(self.out)
        self.self_gate.apply(self._init_weights)
        self.extractor.apply(self._init_weights)
        self._init_weights(self.pooler)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)





    def forward(self,clip_image, clip_text,clip_aspect,input_ids,attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden]
        # ✅ 手动提取 CLS 向量
        cls_token = last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # ✅ 手动 Pooler：Linear + Tanh（与 BERT 原始实现一致）
        tweet_cls = self.activation(self.pooler(cls_token))  # [batch, hidden_size]

        # Dropout + 分类
        pooler_output = self.dropout(tweet_cls)


        image_pool,image_hidden = self.clip_base_model.encode_image(clip_image,normalize=True)  # 257 维度 第一个是cls
        # text_pool= self.clip_base_model.encode_text(clip_text,normalize=True)
        aspect_pool=self.clip_base_model.encode_text(clip_aspect,normalize=True)
        # image_features=F.relu(self.image_Layer(image_features))
        # text_features=F.relu(self.text_layer(text_features))
        # aspect_features=F.relu(self.aspect_Layer(aspect_features))


        image_outputs = self.extractor(
            image_features=image_hidden,
            global_text_features=pooler_output.detach(),
            aspect_text_features=aspect_pool
        )

        image_outputs = self.self_gate(image_outputs)

        combined_features = image_outputs+pooler_output


        gate = self.fuse_gate(combined_features)

        # 4️⃣ 计算相似度
        similarity = (image_pool * aspect_pool).sum(dim=-1, keepdim=True)  # [B, 1]

        text_probs = torch.sigmoid(
            similarity * self.clip_base_model.logit_scale.exp() + self.clip_base_model.logit_bias
        )

        combined_features = text_probs * gate * combined_features + pooler_output

        outputs = self.out(self.dropout(combined_features))

        return outputs, image_outputs

class ClipModel_base(nn.Module):
    def __init__(self,args,clip_base_model):
        super(ClipModel_base, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(self.args.model)  # bert model

        self.clip_base_model=clip_base_model

        self.extractor = DenseFeedbackFeatureExtractor(
        D_text_global=768,
        D_text_aspect=1152,
        D_vision=1152,
        D_model=768,
        D_state=768,
        n_heads=2,
        max_steps=2,
        fuse_mode="concat",      # 可换成 mean/max/last/concat
        state_dropout=self.args.dropout,
        use_state_layernorm=True
        )

        self.norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.self_gate = EnhancedFilterModule(hidden_size=768)

        self.fusion = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(),
            nn.Dropout(self.args.dropout)
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

        self.dropout = nn.Dropout(self.args.dropout)
        self.out = nn.Linear(768, 3)  # 3 classes

        self._init_weights(self.norm)
        self._init_weights(self.fusion)
        self._init_weights(self.out)
        self.self_gate.apply(self._init_weights)
        self.extractor.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)





    def forward(self,clip_image, clip_text,clip_aspect,input_ids,attention_mask):

        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)

        pooler_output = outputs.pooler_output

        image_pool,image_hidden = self.clip_base_model.encode_image(clip_image,normalize=True)  # 257 维度 第一个是cls
        # text_pool= self.clip_base_model.encode_text(clip_text,normalize=True)
        aspect_pool=self.clip_base_model.encode_text(clip_aspect,normalize=True)
        # image_features=F.relu(self.image_Layer(image_features))
        # text_features=F.relu(self.text_layer(text_features))
        # aspect_features=F.relu(self.aspect_Layer(aspect_features))

        image_outputs = self.extractor(
            image_features=image_hidden,
            global_text_features=pooler_output.detach(),
            aspect_text_features=aspect_pool
        )

        image_outputs = self.self_gate(image_outputs)

        combined_features = torch.cat([image_outputs, pooler_output.detach()], dim=-1)

        combined_features = self.fusion(combined_features)

        gate = self.fuse_gate(combined_features)

        # 4️⃣ 计算相似度
        similarity = (image_pool * aspect_pool).sum(dim=-1, keepdim=True)  # [B, 1]

        text_probs = torch.sigmoid(
            similarity * self.clip_base_model.logit_scale.exp() + self.clip_base_model.logit_bias
        )

        combined_features = self.norm(text_probs * gate * combined_features + pooler_output)

        outputs = self.out(self.dropout(combined_features))

        return outputs, image_outputs

