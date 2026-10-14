
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# class StackedFusionModel_2(nn.Module):
#
#     def __init__(self):
#         super(StackedFusionModel_2, self).__init__()
#
#         self.num_classes = 3
#         initial_weights = torch.tensor([[0.2, 0.7,0.1],  # 类别 0: 文本权重 0.3, 多模态权重 0.7
#                                         [0.2, 0.7,0.1],  # 类别 1: 文本权重 0.3, 多模态权重 0.7
#                                         [0.2, 0.7,0.1]],  # 类别 2: 文本权重 0.3, 多模态权重 0.7
#                                        dtype=torch.float32)
#
#         # 将初始权重张量注册为可学习的参数
#         self.class_weights = nn.Parameter(initial_weights)
#
#         self.class_weights.requires_grad = False  # 阻止更新 class_weights
#
#         # Activation functions
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x1, x2):
#         """
#         Args:
#             x1: Tensor of shape [batch_size, num_classes] (logits from model 1)
#             x2: Tensor of shape [batch_size, num_classes] (logits from model 2)
#         Returns:
#             fus_output: Tensor of shape [batch_size, num_classes] (fused logits)
#             class_weights_numpy: NumPy array of shape [num_classes, 2] (detached class weights)
#         """
#         prob1 = self.softmax(x1)
#         prob2 = self.softmax(x2)
#
#         batch_size = x1.size(0)
#
#         # 扩展类别权重以匹配批次大小
#         expanded_class_weights = self.class_weights.unsqueeze(0).repeat(batch_size, 1, 1) # [batch, num_classes, 2]
#
#         # 扩展概率以进行逐元素相乘
#         prob1_expanded = prob1.unsqueeze(-1) # [batch, num_classes, 1]
#         prob2_expanded = prob2.unsqueeze(-1) # [batch, num_classes, 1]
#
#         # 堆叠概率
#         probs_stacked = torch.cat([prob1_expanded, prob2_expanded], dim=-1) # [batch, num_classes, 2]
#
#         # 进行类别特定的加权平均
#         fused_probs = torch.sum(expanded_class_weights * probs_stacked, dim=-1) # [batch, num_classes]
#
#         fus_output = torch.log(fused_probs + 1e-6)
#
#         class_weights_numpy = self.class_weights.detach().cpu().numpy()
#
#         text_weights = class_weights_numpy[:, 0]  # 获取所有类别的文本权重
#         average_text_weight = np.mean(text_weights)  # 计算平均值
#
#         return fus_output, average_text_weight # 返回模型1的权重作为示例

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List



class StackedFusionModel_2(nn.Module):

    def __init__(self):
        super(StackedFusionModel_2, self).__init__()

        self.num_classes = 3
        initial_weights = torch.tensor([[0.8, 0.8],  # 类别 0: 文本权重 0.3, 多模态权重 0.7
                                        [0.8, 0.8],  # 类别 1: 文本权重 0.3, 多模态权重 0.7
                                        [0.8, 0.8]],  # 类别 2: 文本权重 0.3, 多模态权重 0.7
                                       dtype=torch.float32)

        # 将初始权重张量注册为可学习的参数
        self.class_weights = nn.Parameter(initial_weights)

        self.class_weights.requires_grad = False  # 阻止更新 class_weights

        # Activation functions
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor of shape [batch_size, num_classes] (logits from model 1)
            x2: Tensor of shape [batch_size, num_classes] (logits from model 2)
        Returns:
            fus_output: Tensor of shape [batch_size, num_classes] (fused logits)
            class_weights_numpy: NumPy array of shape [num_classes, 2] (detached class weights)
        """
        prob1 = self.softmax(x1)
        prob2 = self.softmax(x2)

        batch_size = x1.size(0)

        # 扩展类别权重以匹配批次大小
        expanded_class_weights = self.class_weights.unsqueeze(0).repeat(batch_size, 1, 1) # [batch, num_classes, 2]

        # 扩展概率以进行逐元素相乘
        prob1_expanded = prob1.unsqueeze(-1) # [batch, num_classes, 1]
        prob2_expanded = prob2.unsqueeze(-1) # [batch, num_classes, 1]

        # 堆叠概率
        probs_stacked = torch.cat([prob1_expanded, prob2_expanded], dim=-1) # [batch, num_classes, 2]

        # 进行类别特定的加权平均
        fused_probs = torch.sum(expanded_class_weights * probs_stacked, dim=-1) # [batch, num_classes]

        fused_probs = torch.clamp(fused_probs, min=1e-12, max=1.0)

        fus_output = torch.log(fused_probs)

        # class_weights_numpy = self.class_weights.detach().cpu().numpy()

        # text_weights = class_weights_numpy[:, 0]  # 获取所有类别的文本权重
        # average_text_weight = np.mean(text_weights)  # 计算平均值

        return fus_output


class FusedModel_loss_1_1(nn.Module):

    def __init__(self, only_text_model,clip_model):
        super().__init__()
        self.only_text_model=only_text_model
        self.clip_model=clip_model
        self.out = nn.Linear(768, 3)  # 3 classes
        self.weight1 = nn.Parameter(torch.tensor(0.8,dtype=torch.float32))  # 可训练的权重1
        self.weight2 = nn.Parameter(torch.tensor(0.2,dtype=torch.float32))  # 可训练的权重1
        # self.primaryGatedFusionModel=PrimaryGatedFusionModel(768,3,768,3)
        # self.attention_fusion = AttentionFusion(hidden_size=3)
        self.stack_fus = StackedFusionModel_2()


    def forward(self,input_ids, attention_mask,
                        input_ids_all, attention_mask_all,
                        clip_image, clip_aspect,clip_tweet,
                        candidate_input_ids,candidate_attention_mask
                        ):

        only_text_output,only_text_pooler=self.only_text_model(input_ids_all, attention_mask_all,candidate_input_ids,candidate_attention_mask)

        clip_output,clip_pooler=self.clip_model(clip_image,clip_aspect,clip_tweet,input_ids_all, attention_mask_all)


        fus_output = self.stack_fus(clip_output,only_text_output)


        # prob_list = [F.softmax(only_text_output, dim=-1), F.softmax(text_image_output, dim=-1),
        #              F.softmax(clip_output, dim=-1)]
        # fus_output = self.attention_fusion(prob_list)

        return  only_text_output,clip_output,fus_output


