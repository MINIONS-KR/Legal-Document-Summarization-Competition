import os
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain

from pprint import pprint


class CustomDataset(Dataset):
    def __init__(self, data, data_dir, mode):
        self.data = data
        self.data_dir = data_dir
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        if self.mode != "test":
            self.inputs, self.labels = self.data_loader()
        else:
            self.inputs = self.data_loader()
            
    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')

        df = self.data
        if self.mode != "test":
            labels = pd.DataFrame(columns=['trg'])
            labels["trg"] = df['extractive'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(50)]))

        num_sentence = []
        for sentence in df["article_original"]:
            num_sentence.append(len(sentence))

        tokenize_sentence = self.tokenizer(
            list(chain.from_iterable(df["article_original"])),
            padding=True,
            truncation=True,
            max_length=120,
            add_special_tokens=True,
            return_tensors="pt",
        )

        inputs = pd.DataFrame(columns=["input_ids", "token_type_ids", "attention_mask"])
        idx = 0
        for length in num_sentence:

            input_ids = torch.stack([tokenize_sentence["input_ids"][idx + i] for i in range(length)], dim=1)
            token_type_ids = torch.stack([tokenize_sentence["token_type_ids"][idx + i] for i in range(length)], dim=1)
            attention_mask = torch.stack([tokenize_sentence["attention_mask"][idx + i] for i in range(length)], dim=1)

            inputs = inputs.append({'input_ids' : torch.tensor(input_ids) , 'token_type_ids' : torch.tensor(token_type_ids), 'attention_mask' : torch.tensor(attention_mask)} , ignore_index=True)
            idx += length
        
        if self.mode != "test":
            return inputs, labels
        else:
            return inputs

    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        if self.mode == 'test':
            return self.inputs["input_ids"][index], self.inputs["token_type_ids"][index], self.inputs["attention_mask"][index]
        else:
            return self.inputs["input_ids"][index], self.inputs["token_type_ids"][index], self.inputs["attention_mask"][index], self.labels["trg"][index]




