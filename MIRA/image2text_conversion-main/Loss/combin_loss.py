from math import e
import re
from sympy import im
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class kl_divergence(nn.Module):



    def __init__(self, temperature=1.0):
        super().__init__()
        # 确保温度是正数
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        self.temperature = temperature

    def forward(self, p_logits, q_logits):


        # 1. 使用温度 T 对 logits 进行缩放
        p_soft = p_logits / self.temperature
        q_soft = q_logits / self.temperature

        # 2. 计算软化后 p 的概率分布和 q 的对数概率分布
        #    p_soft_probs 对应 KL(p||q) 中的 p
        #    q_soft_log_probs 对应 KL(p||q) 中的 log(q)
        p_soft_probs = F.softmax(p_soft, dim=-1)
        q_soft_log_probs = F.log_softmax(q_soft, dim=-1)

        # 3. 使用 PyTorch 的 kl_div 函数计算 KL 散度
        #    F.kl_div(input, target) 计算的是 sum(target * (log(target) - input))
        #    因此，F.kl_div(q_soft_log_probs, p_soft_probs) 计算的是
        #    sum(p_soft_probs * (log(p_soft_probs) - q_soft_log_probs))
        #    这正是 KL(p || q) 的定义。
        kl_loss = F.kl_div(q_soft_log_probs, p_soft_probs.detach(), reduction='batchmean')

        # 4. 乘以 T^2 进行校正
        #    这是知识蒸馏论文中提出的标准做法，用于确保蒸馏损失的梯度大小
        #    与不带温度的学生模型损失（如交叉熵）大致处于同一量级。
        loss = kl_loss * (self.temperature ** 2)

        return loss





class CombinedLoss_3_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = kl_divergence(temperature=1.0)
        self.NLLLoss = torch.nn.NLLLoss()

    def forward(self, only_text_output, text_image_output, fus_output, labels,epoch,output_image,output_text):


        if epoch<1:
            return self.ce_loss(only_text_output,labels) + self.ce_loss(text_image_output,labels) + self.kl_loss(output_text,only_text_output) + self.kl_loss(output_image,text_image_output)

        else:
            kl_loss_1 = self.kl_loss(text_image_output, only_text_output) * 0.5
            kl_loss_2 = self.kl_loss(only_text_output, text_image_output) * 0.5
            kl_loss_3 = self.kl_loss(output_image,text_image_output) * 0.1
            kl_loss_4 = self.kl_loss(output_text,only_text_output) * 0.1
            all_loss = self.NLLLoss(fus_output,labels)
            total_loss = kl_loss_3 + all_loss + kl_loss_4 + kl_loss_1 + kl_loss_2
            return total_loss
class kl_divergence(nn.Module):



    def __init__(self, temperature=1.0):
        super().__init__()
        # 确保温度是正数
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")
        self.temperature = temperature

    def forward(self, p_logits, q_logits):


        # 1. 使用温度 T 对 logits 进行缩放
        p_soft = p_logits / self.temperature
        q_soft = q_logits / self.temperature

        # 2. 计算软化后 p 的概率分布和 q 的对数概率分布
        #    p_soft_probs 对应 KL(p||q) 中的 p
        #    q_soft_log_probs 对应 KL(p||q) 中的 log(q)
        p_soft_probs = F.softmax(p_soft, dim=-1)
        q_soft_log_probs = F.log_softmax(q_soft, dim=-1)

        # 3. 使用 PyTorch 的 kl_div 函数计算 KL 散度
        #    F.kl_div(input, target) 计算的是 sum(target * (log(target) - input))
        #    因此，F.kl_div(q_soft_log_probs, p_soft_probs) 计算的是
        #    sum(p_soft_probs * (log(p_soft_probs) - q_soft_log_probs))
        #    这正是 KL(p || q) 的定义。
        kl_loss = F.kl_div(q_soft_log_probs, p_soft_probs.detach(), reduction='batchmean')

        # 4. 乘以 T^2 进行校正
        #    这是知识蒸馏论文中提出的标准做法，用于确保蒸馏损失的梯度大小
        #    与不带温度的学生模型损失（如交叉熵）大致处于同一量级。
        loss = kl_loss * (self.temperature ** 2)

        return loss





class CombinedLoss_3_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = kl_divergence(temperature=1.0)
        self.NLLLoss = torch.nn.NLLLoss()

    def forward(self, only_text_output, text_image_output, fus_output, labels,epoch,output_image,output_text):


        if epoch<1:
            return self.ce_loss(only_text_output,labels) + self.ce_loss(text_image_output,labels) + self.kl_loss(output_text,only_text_output) + self.kl_loss(output_image,text_image_output)

        else:
            kl_loss_1 = self.kl_loss(text_image_output, only_text_output) * 0.5
            kl_loss_2 = self.kl_loss(only_text_output, text_image_output) * 0.5
            kl_loss_3 = self.kl_loss(output_image,text_image_output) * 0.1
            kl_loss_4 = self.kl_loss(output_text,only_text_output) * 0.1
            all_loss = self.NLLLoss(fus_output,labels)
            total_loss = kl_loss_3 + all_loss + kl_loss_4 + kl_loss_1 + kl_loss_2
            return total_loss







