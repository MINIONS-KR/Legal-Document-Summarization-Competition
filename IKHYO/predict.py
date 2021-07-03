import warnings
warnings.filterwarnings('ignore')

from modules.dataset import CustomDataset
from modules.utils import seed_everything, get_test_config

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from importlib import import_module
import pickle
import os
import random
import json
import argparse


class CFG:
    # Project Environment
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/DATA/Final_DATA/task05_test'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    df = pickle.load(f)
    f.close()
    return df

if __name__ == '__main__':
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU ě´ëŚ: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser(description="AIOnlineCompetition")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'test config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config from custom config.json file
    get_test_config(CFG, os.path.join(CFG.PROJECT_DIR, 'configs', 'inference', args.config))

    # Set random seed
    seed_everything(CFG.seed)

    #get data from json
    with open(os.path.join(CFG.DATA_DIR, "test.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
        
    test_df = pd.DataFrame(data)


    # Load dataset & dataloader
    test_dataset = CustomDataset(test_df, data_dir=CFG.DATA_DIR, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=CFG.batch_size,
                                 num_workers=CFG.num_workers,
                                 pin_memory=CFG.pin_memory,
                                 drop_last=CFG.drop_last,
                                 shuffle=False)
    

    # Load Model
        
    models = [torch.load(f"models/Ik_fold{fold}.pt") for fold in range(5)]

    # make predictions
    for i in range(len(models)):
        models[i].eval()
        pred_lst = []
        with torch.no_grad():
            for batch_index, (input_ids, token_type_ids, attention_mask) in enumerate(test_dataloader):
                input_ids = input_ids.to(CFG.device)
                token_type_ids = token_type_ids.to(CFG.device)
                attention_mask = attention_mask.to(CFG.device)

                sent_score = models[i](input_ids, token_type_ids, attention_mask)

                pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            
        with open(os.path.join(CFG.DATA_DIR, "sample_submission.json"), "r", encoding="utf-8-sig") as f:
            sample_submission = json.load(f)
        
        for row, pred in zip(sample_submission, pred_lst):
            row['summary_index1'] = pred[0]
            row['summary_index2'] = pred[1]
            row['summary_index3'] = pred[2]
        
        with open(f"submissions/ikhyo{i}.json", "w") as f:
            json.dump(sample_submission, f, separators=(',', ':'))
        
