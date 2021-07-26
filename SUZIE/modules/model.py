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