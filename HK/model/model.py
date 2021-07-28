"""
"""
import math
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import f1_score



class Summarizer(nn.Module):

    def __init__(self, model_name):
        """
        """
        super(Summarizer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        print('sents_vec.shape:', sents_vec.shape)
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores


class SummarizerWithSpt(nn.Module):
    
    def __init__(self, model_name):
        """
        """
        super(SummarizerWithSpt, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        # self.config.vocab_size += 1
        # self.config.type_vocab_size += 1
        # self.new_token_id = self.config.vocab_size
        self.encoder = transformers.AutoModel.from_pretrained(model_name, config=self.config)
        self.query_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.key_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.value_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        batch_size = x.size(0)
        # x 맨 앞에 cls token id 추가 (B * L+1)
        new_token = torch.tensor([[2]]).repeat(batch_size, 1).to('cuda')
        x = torch.cat((new_token, x), 1)
        # segs 맨 앞에 1 추가 (B * L+1)
        new_token = torch.tensor([[1]]).repeat(batch_size, 1).to('cuda')
        segs = torch.cat((new_token, segs), 1)
        # clss 맨 앞에 -1 추가 (B * 51)
        new_token = torch.tensor([[-1]]).repeat(batch_size, 1).to('cuda')
        clss = torch.cat((new_token, clss), 1)
        # mask에 True 추가 (B * L+1)
        new_token = torch.tensor([[True]]).repeat(batch_size, 1).to('cuda')
        mask = torch.cat((new_token, mask), 1)
        # mask_clss 맨 앞에 False 추가 (B * 51)
        new_token = torch.tensor([[False]]).repeat(batch_size, 1).to('cuda')
        mask_clss = torch.cat((new_token, mask_clss), 1)

        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]

        # attn 연산
        query = top_vec[:,:1,:]
        query = self.query_layer(query)
        key = self.key_layer(sents_vec)
        value = self.value_layer(sents_vec)
        attn_rate = torch.matmul(key, torch.transpose(query, 1, 2))
        attn_rate = attn_rate / math.sqrt(self.config.hidden_size)
        attn_rate = F.softmax(attn_rate, 1)
        sent_vec = attn_rate * value

        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        sent_scores = sent_scores[:,1:]
        return sent_scores


class SummarizerWithSumm(nn.Module):
    
    def __init__(self, model_name):
        """
        """
        super(SummarizerWithSumm, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.summ_encoder = transformers.AutoModel.from_pretrained(model_name, config=self.config)
        self.sentence_encoder = transformers.AutoModel.from_pretrained(model_name, config=self.config)
        self.query_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.key_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.value_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, segs, clss, mask, mask_clss, summ_src, summ_segs, summ_mask):
        """
        """
        summ_vec = self.summ_encoder(input_ids = summ_src.long(), attention_mask = summ_mask.float(),  token_type_ids = summ_segs.long()).pooler_output.unsqueeze(1)
        top_vec = self.sentence_encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]

        # attn 연산
        query = self.query_layer(summ_vec)
        key = self.key_layer(sents_vec)
        value = self.value_layer(sents_vec)
        attn_rate = torch.matmul(key, torch.transpose(query, 1, 2))
        attn_rate = attn_rate / (math.sqrt(self.config.hidden_size) * 10)
        attn_rate = F.softmax(attn_rate, 1)
        sent_vec = attn_rate * value

        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores