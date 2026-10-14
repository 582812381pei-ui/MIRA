import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from torch.utils.hipify.hipify_python import value
from transformers import RobertaModel, BertModel, AutoModel



class TextImageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(self.cfg.model)  # bert model
        self.dropout = nn.Dropout(self.cfg.dropout)
        # self.layer1=nn.Linear(self.bert.config.hidden_size,self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, 3)  # 3 classes
        self._init_weights(self.out)

    def _init_weights(self, module):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            torch.nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):

        # inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)  #[batch,128,768]

        # # Prediction Module
        outputs_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)


        tweet_cls = outputs_text.pooler_output


        outputs = self.out(tweet_cls)

        return outputs, tweet_cls

