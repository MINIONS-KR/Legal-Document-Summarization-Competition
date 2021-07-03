import os
import json
import pickle
import random
import torch
import numpy as np
import pandas as pd

from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from sklearn.model_selection import KFold

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
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']

# PERFORMANCE RECORD
MODEL_NAME = "kobert"
PERFORMANCE_RECORD_DIR = os.path.join(PROJECT_DIR, 'results', MODEL_NAME)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


# Set random seed
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"{DATA_DIR}/train.json")
with open(f"{DATA_DIR}/train.json", "r", encoding="utf-8-sig") as f:
    data = pd.read_json(f) 
df = pd.DataFrame(data)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
fold_idx = 0
for train_idx, test_idx in kf.split(df):
    train_data, valid_data = df.iloc[train_idx], df.iloc[test_idx]

    print(f"Fold {fold_idx} -- Train Set: {len(train_data)}, Valid Set: {len(valid_data)}")

    with open(f"{DATA_DIR}/{MODEL_NAME}-{fold_idx}-train.pkl", 'wb') as f:
        pickle.dump(train_idx, f)
        
    with open(f"{DATA_DIR}/{MODEL_NAME}-{fold_idx}-valid.pkl", 'wb') as f:
        pickle.dump(test_idx, f)

    fold_idx += 1
