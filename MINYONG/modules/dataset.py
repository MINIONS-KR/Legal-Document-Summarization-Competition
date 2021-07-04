import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain

class KLUEBERT_base_Dataset(Dataset):
    def __init__(self, data, data_dir, mode, data_aug=False, add_mask=False):
        self.data = data
        self.data_dir = data_dir
        self.mode = mode
        self.data_aug = data_aug
        self.add_mask = add_mask
        self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')

        df = self.data
        inputs = pd.DataFrame(columns=['src'])
        labels = pd.DataFrame(columns=['trg'])

        # data augmentation
        if self.data_aug:
            df = self.data_augmentation(df)

        inputs['src'] =  df['article_original']
        if self.mode != "test":
            labels['trg'] =  df['extractive']

        # Preprocessing
        inputs, labels = self.preprocessing(inputs, labels)
        print("preprocessing")

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data


    def data_augmentation(self, df):
        aug_df = df.copy()
        aug_df["sent_count"] = aug_df.article_original.apply(lambda x: len(x))

        for id, article, extractive, sent_count in zip(aug_df.id, aug_df.article_original, aug_df.extractive, aug_df.sent_count):
            if sent_count >= 11:
                df = df.append(pd.DataFrame([(id, extractive, article)], columns=["id", "extractive", "article_original"]), ignore_index=True)
        
        return df


    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        # Encoding original text
        # inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)), add_special_tokens=True) for i in range(len(x))]))))

        def tokenize(x):
            result = [self.tokenizer.encode(x[i], add_special_tokens=True) for i in range(len(x))]
            result_concat = list(chain.from_iterable(result))

            if len(result_concat) <= 512:
                return torch.tensor(result_concat)
            
            else:
                return torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], truncation=True, max_length = int(512 * len(result[i]) / len(result_concat)) if int(512 * len(result[i]) / len(result_concat)) >= 3 else 3, add_special_tokens=True) for i in range(len(x))])))

        def custom_to_mask(src):
            mask_token = self.tokenizer.mask_token_id
            special_token = [self.tokenizer.sep_token_id, self.tokenizer.cls_token_id, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id]

            mask_idxs = set()
            while len(mask_idxs) < 5:
                ids = random.randrange(1, len(src) - 1)
                if src[ids] not in special_token:
                    mask_idxs.add(ids)

            for mask_idx in list(mask_idxs):
                src[mask_idx] = mask_token

            return src

        inputs['src'] = inputs['src'].map(tokenize)
        if self.add_mask:
            inputs['src'] = inputs['src'].map(custom_to_mask)
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