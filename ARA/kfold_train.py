import os
import sys
import random
import json
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import transformers

from torch.utils.data import DataLoader
from datetime import datetime, timezone, timedelta
from sklearn.model_selection import train_test_split

from model.model import *
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.metrics import Hitrate
from modules.recorders import PerformanceRecorder

# DEBUG
DEBUG = False

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = '../DATA/Final_DATA/task05_train'
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']

# PERFORMANCE RECORD
MODEL_NAME = "kobert"
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'model', MODEL_NAME)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

# 1. Setting
# Set random seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                           
# Dataset
with open(f"{DATA_DIR}/train.json", "r", encoding="utf-8-sig") as f:
    data = pd.read_json(f) 
df = pd.DataFrame(data)

for fold_idx in range(5):
    with open(f"{DATA_DIR}/{MODEL_NAME}-{fold_idx}-train.pkl", 'rb') as f:
    train_index = pickle.load(f)
    train_data = df.iloc[train_index]

    with open(f"{DATA_DIR}/{MODEL_NAME}-{fold_idx}-valid.pkl", 'rb') as f:
        valid_index = pickle.load(f)
    valid_data = df.iloc[valid_index]
    
    train_dataset = CustomDataset(train_data, mode='train', model_name=MODEL_NAME)
    validation_dataset = CustomDataset(valid_data, mode='valid', model_name=MODEL_NAME)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model
    model = Summarizer().to(device)
    
    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=50, steps_per_epoch=len(train_dataloader))
    loss_fn = torch.nn.BCELoss(reduction='none')
    
    # Set metrics
    metric_fn = Hitrate
    
    # Set trainer
    trainer = Trainer(model, device, loss_fn, metric_fn, optimizer, scheduler)
    
    # Set performance recorder
    key_column_value_list = [
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

        
    criterion = 0
    for epoch_index in range(EPOCHS):
        trainer.train_epoch(train_dataloader, validation_dataloader, performance_recorder, f"{MODEL_NAME}{fold_idx}", epoch_index=epoch_index)
        trainer.validate_epoch(validation_dataloader, epoch_index=epoch_index)

        if trainer.validation_score > criterion:
            criterion = trainer.validation_score
            performance_recorder.weight_path = os.path.join("/Minions/models", f"{MODEL_NAME}{fold_idx}.pt')
            performance_recorder.save_weight()
    
    torch.cuda.empty_cache()