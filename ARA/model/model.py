"""
"""
import torch
from torch import nn
import transformers
from sklearn.metrics import f1_score

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class SentsAvg_Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(SentsAvg_Summarizer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained("kykim/bert-kor-base")

        self.sents_layer = nn.Linear(768, 768)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.xavier_uniform_(self.sents_layer.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state

        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        new_sents_vec = []
        for b in range(len(x)): # batch size
            batch_sents_vec = []
            sents_end = torch.where(x[b] == 3)[0]
            for i in range(len(sents_end)):
                batch_sents_vec.append(self.sents_layer(torch.mean(top_vec[b, clss[b][i].long():sents_end[i]], dim=0)))
            
            for i in range(len(clss[b])-len(sents_end)):
                batch_sents_vec.append(top_vec[b, -1])
            
            new_sents_vec.append(torch.stack(batch_sents_vec, dim=0))
        
        new_sents_vec = torch.stack(new_sents_vec, dim=0)
        new_sents_vec = new_sents_vec * mask_clss[:, :, None].float()

        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        # sents_vec = sents_vec * mask_clss[:, :, None].float()

        h = self.fc(new_sents_vec).squeeze(-1)

        sent_scores = self.sigmoid(h) * mask_clss.float()

        return sent_scores


class CLSSentsAvg_Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(CLSSentsAvg_Summarizer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained("kykim/bert-kor-base")
        
        
        self.cls_layer = nn.Linear(768, 768//2)
        self.sents_layer = nn.Linear(768, 768//2)
        self.fc = nn.Linear(768, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.xavier_uniform_(self.cls_layer.weight)
        torch.nn.init.xavier_uniform_(self.sents_layer.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
    
    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state

        new_sents_vec = []
        for b in range(len(x)): # batch size
            batch_sents_vec = []
            sents_end = torch.where(x[b] == 3)[0]
            for i in range(len(sents_end)):
                sents_embedding = self.sents_layer(torch.mean(top_vec[b, clss[b][i].long()+1:sents_end[i]], dim=0))
                cls_embedding = self.cls_layer(top_vec[b, clss[b][i].long()])
                
                batch_sents_vec.append(torch.cat([cls_embedding, sents_embedding]))
            
            for i in range(len(clss[b])-len(sents_end)):
                batch_sents_vec.append(top_vec[b, -1])
            
            new_sents_vec.append(torch.stack(batch_sents_vec, dim=0))
        
        new_sents_vec = torch.stack(new_sents_vec, dim=0)
        new_sents_vec = new_sents_vec * mask_clss[:, :, None].float()

        h = self.fc(new_sents_vec).squeeze(-1)

        sent_scores = self.sigmoid(h) * mask_clss.float()

        return sent_scores

class BERT_Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(BERT_Summarizer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained("kykim/bert-kor-base")
        
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids=x.long(), attention_mask = mask.float(), token_type_ids=segs.long()).last_hidden_state
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        
        return sent_scores

class Electra_Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(Electra_Summarizer, self).__init__()
        self.encoder = transformers.AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        
        self.lstm = nn.LSTM(input_size = 768, 
                            hidden_size = 768, 
                            num_layers = 3,
                            dropout=0.1, 
                            bidirectional = False, 
                            batch_first = True)
        
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.xavier_uniform_(self.fc.weight)


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        
        top_vec = self.encoder(input_ids=x.long(), attention_mask = mask.float(), token_type_ids=segs.long())[0]
        top_vec, (last_h, last_c) = self.lstm(top_vec)
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        
        return sent_scores
