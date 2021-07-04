import os
import json
import pandas as pd
import numpy as np
import random
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.model import *
from modules.dataset import TestCustomDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.metrics import Hitrate
from modules.recorders import PerformanceRecorder

from datetime import datetime, timezone, timedelta
from sklearn.model_selection import train_test_split

# CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = '/DATA/Final_DATA/task05_test'
SAVE_DATA_DIR = os.path.join(PROJECT_DIR, 'data')
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')
config = load_yaml(TRAIN_CONFIG_PATH)

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# TRAIN
EPOCHS = config['TRAIN']['num_epochs']
BATCH_SIZE = config['TRAIN']['batch_size']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)

args = parser.parse_args()

# PERFORMANCE RECORD
MODEL_NAME = args.model_name
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'model', MODEL_NAME)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

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


with open(f"{DATA_DIR}/test.json", "r", encoding="utf-8-sig") as f:
    data = pd.read_json(f) 
df = pd.DataFrame(data)

test_dataset = TestCustomDataset(df, mode='test', model_name=MODEL_NAME)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model
model = torch.load(os.path.join("./models", f"{MODEL_NAME}.pt"))
model.to(device)
model.eval()

# Set optimizer, scheduler, loss function, metric function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=50, steps_per_epoch=len(test_dataloader))
loss_fn = torch.nn.BCELoss(reduction='none')

# Set metrics
metric_fn = Hitrate

# Set trainer
trainer = Trainer(model, device, loss_fn, metric_fn, optimizer, scheduler)

predict = trainer.test_epoch(test_dataloader)

with open("./submissions/sample_submission.json", "r", encoding="utf-8-sig") as f:
    sample_submission = json.load(f)
    
for row, pred in zip(sample_submission, predict):
    row['summary_index1'] = pred[0]
    row['summary_index2'] = pred[1]
    row['summary_index3'] = pred[2]
    
with open(os.path.join(f"./submissions/{MODEL_NAME}.json"), "w") as f:
    json.dump(sample_submission, f, separators=(',', ':'))