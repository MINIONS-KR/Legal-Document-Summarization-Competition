import os
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from pprint import pprint
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def make_word_index_dict(tokens):
    word_index = {}
    word = ''
    index = []
    
    for i, t in enumerate(tokens):
        if (t == '[CLS]') or (t == '[SEP]'):
            continue
        if not t.startswith('##'):
            if word:
                word_index[word.replace('##', '')] = index
                word = ''
                index = []
            word += t
            index.append(i)
        if t.startswith('##'):
            word += t
            index.append(i)
                
    return word_index


class CustomDataset(Dataset):
    def __init__(self, args, data, mode):
        self.data = data
        self.data_dir = args.data_dir
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode + '_Y.pt'))

        else:
            df = self.data
            inputs = pd.DataFrame(columns=['src'])
            labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']

            if self.mode != "test":
                labels['trg'] =  df['extractive']

            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            print("preprocessing")

            # Save data
            torch.save(inputs, os.path.join(self.data_dir, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode + '_Y.pt'))

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)), dtype=torch.int64)]))
        return padded_data
    
    def tokenize(self, x):
        result = [self.tokenizer.encode(x[i], add_special_tokens=True, truncation=True) for i in range(len(x))]
        result_concat = list(chain.from_iterable(result))

        if len(result_concat) <= 512:
            return torch.tensor(result_concat)
        
        else:
            return torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 * len(result[i]) / len(result_concat)) if int(512 * len(result[i]) / len(result_concat)) >= 3 else 3, add_special_tokens=True, truncation=True) for i in range(len(x))])))        
    
    def mask_to_random_tokens(self, inputs, k=2):
        mask_token_id = self.tokenizer.mask_token_id

        for src in inputs:
            index_to_mask = []

            tokens = self.tokenizer.convert_ids_to_tokens(src)
            word_dict = make_word_index_dict(tokens)
            candidates = [key for key in list(word_dict.keys()) if (3 < len(key)) and (len(key)<=6)]

            if len(candidates) >= k:
                for i in range(k):
                    rand_num = random.randint(0, len(candidates)-1)
                    index_to_mask.extend(word_dict[candidates[rand_num]])
            else:
                for i in range(len(candidates)):
                    index_to_mask.extend(word_dict[candidates[i]])

            for idx in index_to_mask:
                src[idx] = mask_token_id

        return inputs
    
    def mask_to_long_tokens(self, inputs, k=2):
        mask_token_id = self.tokenizer.mask_token_id

        for src in inputs:
            index_to_mask = []

            tokens = self.tokenizer.convert_ids_to_tokens(src)
            word_dict = make_word_index_dict(tokens)
            candidates = sorted(list(word_dict.keys()), reverse=True, key=len)

            for i in range(k):
                index_to_mask.extend(word_dict[candidates[i]])

            for idx in index_to_mask:
                src[idx] = mask_token_id

        return inputs

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        # Encoding original text
        inputs['src'] = inputs['src'].map(self.tokenize)
        if self.mode == 'train':
            inputs['src'] = self.mask_to_random_tokens(inputs['src'])
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        
        # Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}
        if self.mode != 'test':
            labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))

        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if self.mode == 'test':
            return [self.inputs[index][i] for i in range(5)]
        else:
            return [self.inputs[index][i] for i in range(5)], self.labels[index]


def get_train_loaders(args):
    """
        define train/validation pytorch dataset & loader

        Returns:
            train_loader: pytorch data loader for train data
            val_loader: pytorch data loader for validation data
    """
    # get data from json
    with open(os.path.join(args.data_dir, "train.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    train_df = pd.DataFrame(data)

    if args.partial_dataset != 'None':
        print(args.partial_dataset)
        small_idx, large_idx = [], []
        for idx, article in enumerate(list(train_df['article_original'])):
            if len(article) > 6:
                large_idx.append(idx)
            else:
                small_idx.append(idx)
        if args.partial_dataset == 'small':
            train_df = train_df.iloc[small_idx, :].reset_index()
        if args.partial_dataset == 'large':
            train_df = train_df.iloc[large_idx, :].reset_index()
    
    if args.train_kfold:
        kf = KFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            if args.fold != fold:
                continue
            train_data = train_df.iloc[train_idx]
            val_data = train_df.iloc[val_idx]        
    else:
        train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=args.seed)
    
    # get train & valid dataset from dataset.py
    train_dataset = CustomDataset(args, train_data, mode='train')
    val_dataset = CustomDataset(args, val_data, mode='valid')

    # define data loader based on each dataset
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=False,
                                shuffle=False)

    return train_dataloader, val_dataloader
    
    
def get_test_loader(args):
    # Get data from json
    with open(os.path.join(args.data_dir, "test.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    test_df = pd.DataFrame(data)
    
    # Load dataset & dataloader
    test_dataset = CustomDataset(args, test_df, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 shuffle=False)
    
    return test_dataloader