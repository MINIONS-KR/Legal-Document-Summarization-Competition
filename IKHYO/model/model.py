"""
"""
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from sklearn.metrics import f1_score


class Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(Summarizer, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained('klue/bert-base')
        self.encoder = transformers.AutoModel.from_pretrained('klue/bert-base', config = self.config)
        
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 768, bidirectional = True, batch_first=True)

        self.fc = nn.Linear(768*2, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        """
        input_ids = input_ids.squeeze(0).transpose(0, 1)
        token_type_ids = token_type_ids.squeeze(0).transpose(0, 1)
        attention_mask = attention_mask.squeeze(0).transpose(0, 1)

        sents_vec = self.encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state
        lstm_input = sents_vec[:, 0, :].unsqueeze(1).transpose(0, 1)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(lstm_input)

        h = self.sigmoid(self.fc(enc_hiddens.squeeze(0)).squeeze(-1))
        
        h = torch.cat([h, torch.zeros(50 - len(input_ids)).to("cuda:0")], dim=0)
       
        return h.unsqueeze(0)


