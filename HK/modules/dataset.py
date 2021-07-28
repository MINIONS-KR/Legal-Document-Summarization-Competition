"""
"""

import os
import time
import pandas as pd
import torch
import modules.utils as utils
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain
from sklearn.model_selection import train_test_split




class CustomDataset(Dataset):
    def __init__(self, data_dir, mode, model_name):
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        if 'summ' in self.data_dir:
            print('Loading ' + self.mode + '_summ dataset..')
            file_path = os.path.join(self.data_dir[:-5], self.mode, self.mode + '.json')
            df = utils.load_json(file_path)
            file_path = os.path.join(self.data_dir, self.mode, 'summ_'+self.mode+'.csv')
            summ_df = utils.load_csv(file_path)
        else:
            print('Loading ' + self.mode + ' dataset..')
            file_path = os.path.join(self.data_dir, self.mode, self.mode + '.json')
            df = utils.load_json(file_path)

        train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)
        if self.mode == 'train': 
            df = train_df
        elif self.mode == 'val': 
            df = valid_df

        inputs = pd.DataFrame(columns=['src'])
        labels = pd.DataFrame(columns=['trg'])
        inputs['src'] =  df['article_original']
        if self.mode != 'test':
            labels['trg'] =  df['extractive']
        # Preprocessing
        if 'summ' in self.data_dir:
            inputs, labels = self.preprocessing(inputs, labels, summ_df.loc[list(df.index)])
        else:
            inputs, labels = self.preprocessing(inputs, labels)
        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs, labels, summ_df=None):
        if summ_df is not None:
            print('Preprocessing ' + self.mode + '_summ dataset..')
        else:
            print('Preprocessing ' + self.mode + ' dataset..')

        #Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)),  add_special_tokens=True, truncation=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        # ##Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))
        
        if summ_df is not None:
            input_ids = []
            token_type_ids = []
            attention_mask = []
            for idx, row in tqdm(summ_df.iterrows(), total=len(summ_df)):
                tokenized_text = self.tokenizer(row['summ_text'], max_length=256, truncation=True, padding='max_length', return_tensors='pt')
                input_ids.append(tokenized_text['input_ids'].squeeze())
                token_type_ids.append(tokenized_text['token_type_ids'].squeeze())
                attention_mask.append(tokenized_text['attention_mask'].squeeze())

            inputs['summ_src'] = input_ids
            inputs['summ_segs'] = token_type_ids
            inputs['summ_mask'] = attention_mask

        # #Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        if self.mode != 'test':
            labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))
        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        num = len(self.inputs[0])
        if self.mode != 'test':
            return [self.inputs[index][i] for i in range(num)], self.labels[index]
        else:
            return [self.inputs[index][i] for i in range(num)]
