""" 추론 코드
"""

import os
import random
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from model.model import *
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.metrics import Hitrate
from modules.utils import load_yaml, save_yaml, get_logger, make_directory

if __name__ == '__main__':


    # Set random seed
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    DATA_DIR = './data'
    BATCH_SIZE = 4
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yml')
    config = load_yaml(PREDICT_CONFIG_PATH)
    TRAINED_MODEL_PATH = config['PREDICT']['trained_model_path']
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset & dataloader
    test_dataset = CustomDataset(data_dir=DATA_DIR, mode='test', model_name=config['PREDICT']['model'])
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Model
    model = SummarizerWithSpt(config['PREDICT']['model']).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    # Set metrics & Loss function
    metric_fn = Hitrate
    loss_fn = torch.nn.BCELoss(reduction='none')

    # Set trainer
    trainer = Trainer(model, device, loss_fn, metric_fn)

    # Predict
    predict = trainer.test_epoch(test_dataloader)

    with open(os.path.join(DATA_DIR, 'test/sample_submission.json'), "r", encoding="utf-8-sig") as f:
        sample_submission = json.load(f)
        
    for row, pred in zip(sample_submission, predict):
        row['summary_index1'] = pred[0]
        row['summary_index2'] = pred[1]
        row['summary_index3'] = pred[2]
        
    with open(os.path.join("./output", config['PREDICT']['output']), "w") as f:
        json.dump(sample_submission, f, separators=(',', ':'))