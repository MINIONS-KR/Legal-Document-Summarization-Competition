import torch
import transformers
from torch import nn
from sklearn.metrics import f1_score
from transformers import AutoConfig, AutoModel


class Summarizer(nn.Module):

    def __init__(self, args):
        """
        """
        super(Summarizer, self).__init__()
        self.args = args
        self.device = args.device
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        
        self.encoder = transformers.AutoModel.from_pretrained(args.model_name)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(), token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores
    
    
class TopKBertSumExt(nn.Module):
    def __init__(self, args):
            super(TopKBertSumExt, self).__init__()
            self.args = args
            self.device = args.device
            self.hidden_dim = self.args.hidden_dim
            self.n_layers = self.args.n_layers
            self.n_heads = self.args.n_heads
            self.bert = transformers.AutoModel.from_pretrained(args.model_name)
            self.encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_heads, batch_first=True),
                                                num_layers=self.n_layers)
            self.fc = nn.Linear(self.hidden_dim, 1)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, segs, clss, mask, mask_clss):
        top_vec = self.bert(input_ids = x.long(), attention_mask = mask.float(), token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        #===============================================#
        topk = torch.topk(sent_scores, 25, axis=1).indices.tolist()
        new_mask_clss = torch.zeros_like(mask_clss)
        for idx, k in enumerate(topk):
          sent_num = mask_clss.tolist()[idx].count(True)
          if sent_num < 8:
            new_mask_clss[idx, :, None] = mask_clss[idx, :, None]
          else:
            new_mask_clss[idx, k[:sent_num//2], None] = True
        new_sents_vec = sents_vec * new_mask_clss[:, :, None].float()
        top_k_sents = self.encoder(src = new_sents_vec.float(), src_key_padding_mask = new_mask_clss.float())
        top_k_sents = top_k_sents * new_mask_clss[:, :, None].float()
        h2 = self.fc2(top_k_sents).squeeze(-1)
        final_sent_scores = self.sigmoid(h2) * new_mask_clss.float()
        
        return final_sent_scores, new_mask_clss
