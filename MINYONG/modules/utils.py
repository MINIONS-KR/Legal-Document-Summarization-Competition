""" 공용 함수
    * File I/O
    * Model Load / Save
    * Seed
    * System
"""

from typing import Callable
import torch.utils.data
import torchvision

import os
import random
import pprint
import json
import pickle
import yaml
import pandas as pd
import numpy as np
import torch
import logging


def get_train_config(CFG, config):
    '''
        train config file parser
    '''

    with open(config) as f:
        config = json.load(f)
    
    # SEED
    CFG.seed = config["SEED"]["seed"]

    # DATALOADER
    CFG.num_workers = config["DATALOADER"]["num_workers"]
    CFG.pin_memory = config["DATALOADER"]["pin_memory"]
    CFG.drop_last = config["DATALOADER"]["drop_last"]

    # TRAIN
    CFG.epochs = config['TRAIN']['epochs']
    CFG.batch_size = config['TRAIN']['batch_size']
    CFG.grad_accum = config['TRAIN']['grad_accum']
    CFG.data_aug = config['TRAIN']['data_aug']
    CFG.add_mask = config['TRAIN']['add_mask']
    CFG.learning_rate = config['TRAIN']['learning_rate']
    CFG.backbone_ratio = config['TRAIN']['backbone_ratio']
    CFG.max_grad_norm = config['TRAIN']['max_grad_norm']
    CFG.valid_step = config['TRAIN']['valid_step']
    CFG.early_stopping_patience = config['TRAIN']['early_stopping_patience']
    CFG.model = config['TRAIN']['model']
    CFG.model_params = config['TRAIN']['model_params']
    CFG.dataset = config['TRAIN']['dataset']
    CFG.optimizer = config["TRAIN"]["optimizer"]
    CFG.optimizer_params = config["TRAIN"]["optimizer_params"]
    CFG.scheduler = config["TRAIN"]["scheduler"]
    CFG.scheduler_params = config["TRAIN"]["scheduler_params"]
    CFG.criterion = config["TRAIN"]["criterion"]
    CFG.criterion_params = config["TRAIN"]["criterion_params"]

    # PERFORMANCE RECORD
    CFG.DEBUG = config['PERFORMANCE_RECORD']['DEBUG']
    CFG.PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']
    CFG.PERFORMANCE_RECORD_DIR = os.path.join(CFG.PROJECT_DIR, 'results', 'train', CFG.RESULT_DIR)

    # pprint.pprint(CFG.__dict__)


def get_test_config(CFG, config):
    '''
        train config file parser
    '''

    with open(config) as f:
        config = json.load(f)
    
    # SEED
    CFG.seed = config["SEED"]["seed"]

    # DATALOADER
    CFG.num_workers = config["DATALOADER"]["num_workers"]
    CFG.pin_memory = config["DATALOADER"]["pin_memory"]
    CFG.drop_last = config["DATALOADER"]["drop_last"]

    # PREDICT
    CFG.train_serial = config['PREDICT']['train_serial']
    CFG.model = config['PREDICT']['model']
    CFG.model_params = config['PREDICT']['model_params']
    CFG.dataset = config['PREDICT']['dataset']
    CFG.batch_size = config['PREDICT']['batch_size']


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU


def make_directory(directory: str) -> str:
    """경로가 없으면 생성
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지
    """

    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"

        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"

    return msg


def get_logger(name: str, file_path: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger




"""
File I/O
"""
def load_csv(path: str):
    return pd.read_csv(path)

def load_json(path):
    return pd.read_json(path, orient='records', encoding='utf-8-sig')

def load_jsonl(path):
    with open(path, encoding='UTF8') as f:
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']
        df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
        return df

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def save_csv(path: str, obj: dict, index=False):
    try:
        obj.to_csv(path, index=index)
        message = f'csv saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message

def save_json(path: str, obj:dict):
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, indent=4, sort_keys=False)
        message = f'Json saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message

def save_pkl(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_yaml(path, obj):
    try:
        with open(path, 'w') as f:
            yaml.dump(obj, f, sort_keys=False)
        message = f'Json saved {path}'
    except Exception as e:
        message = f'Failed to save : {e}'
    print(message)
    return message

def count_csv_row(path):
    """
    CSV 열 수 세기
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        n_row = sum(1 for row in reader)

