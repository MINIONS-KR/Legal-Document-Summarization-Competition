import warnings
warnings.filterwarnings('ignore')

from modules.dataset import get_test_loader
from modules.utils import seed_everything
from modules.trainer import get_model
from args import parse_args
from modules.model import *

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from importlib import import_module
import os
import random
import json
import argparse


def load_model(args):
    model_path = os.path.join(args.model_dir, args.test_model_name)
    load_state = torch.load(model_path)
    model = get_model(args)

    model.load_state_dict(load_state['state_dict'], strict=True)
    
    print("Loading Model from:", model_path, "...Finished.")
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    seed_everything(args.seed)
    
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{args.device}]")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    
    test_loader = get_test_loader(args)
    model = load_model(args)
    model.eval()
    pred_lst = []
    
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            src = data[0].to(args.device)
            clss = data[1].to(args.device)
            segs = data[2].to(args.device)
            mask = data[3].to(args.device)
            mask_clss = data[4].to(args.device)

            sent_score = model(src, segs, clss, mask, mask_clss)
            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
        
    with open(os.path.join(args.submission_dir, "sample_submission.json"), "r", encoding="utf-8-sig") as f:
        sample_submission = json.load(f)
    
    for row, pred in zip(sample_submission, pred_lst):
        row['summary_index1'] = pred[0]
        row['summary_index2'] = pred[1]
        row['summary_index3'] = pred[2]
    
    with open(os.path.join(args.submission_dir, args.submission_name), "w") as f:
        json.dump(sample_submission, f, separators=(',', ':'))

        
if __name__ == '__main__':
    args = parse_args(mode='train')
    main(args)
