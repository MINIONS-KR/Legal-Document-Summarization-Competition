import warnings
warnings.filterwarnings('ignore')


from modules.model import Summarizer
from modules.dataset import get_train_loaders
from modules.utils import get_logger, make_directory, seed_everything, save_json
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder

import wandb
from args import parse_args
from modules import trainer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from importlib import import_module
import os
import json
import random
import argparse
from sklearn.model_selection import KFold



def main(args):
    wandb.login()
    
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{args.device}]")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    if args.train_kfold:
        for i in range(5):
            print(f"[K-Fold] Fold {i} starts!")
            args.fold = i
            wandb.init(project='AI-Online-Competition', config=vars(args), entity="ohsuz", name=f"{args.run_name}_{args.fold}")
            trainer.run(args)
            wandb.finish()
    else:
        wandb.init(project='AI-Online-Competition', config=vars(args), entity="ohsuz", name=args.run_name)
        trainer.run(args)
        wandb.finish()


if __name__ == "__main__":
    args = parse_args(mode='train')
    main(args)