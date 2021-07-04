import warnings
warnings.filterwarnings('ignore')


from model.model import Summarizer
from modules.dataset import CustomDataset
from modules.trainer import Trainer
from modules.utils import get_logger, make_directory, get_train_config, seed_everything, save_json
from modules.criterion import create_criterion
from modules.optimizer import create_optimizer
from modules.scheduler import create_scheduler
from modules.earlystoppers import LossEarlyStopper
from modules.metrics import Hitrate
from modules.recorders import PerformanceRecorder

import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from importlib import import_module
from sklearn.model_selection import KFold
import os
import json
import random
import argparse


class CFG:
    # Project Environment
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = '/DATA/Final_DATA/task05_train'

    # TRAIN SERIAL
    KST = timezone(timedelta(hours=9))
    TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")

    # DEVICE
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    df = pickle.load(f)
    f.close()
    return df

def get_data_utils():
    """
        define train/validation pytorch dataset & loader

        Returns:
            train_loader: pytorch data loader for train data
            val_loader: pytorch data loader for validation data
    """
    # get data from json
    with open(os.path.join(CFG.DATA_DIR, "train.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    train_df = pd.DataFrame(data)

    train_dataset = CustomDataset(train_df, data_dir=CFG.DATA_DIR, mode='train')

    return train_dataset


def get_model(train_dataloader):
    '''
        get defined model from model.py
        
        Returns:
            model: pytorch model that would be trained
            optimizer: pytorch optimizer for gradient descent
            scheduler: pytorch lr scheduler
            criterion: loss function
    '''

    # Load Model
    model_module = getattr(import_module("model.model"), CFG.model)
    model = model_module()

    CFG.system_logger.info('===== Review Model Architecture =====')
    CFG.system_logger.info(f'{model} \n')

    # get optimizer from optimizer.py
    optimizer = create_optimizer(
        CFG.optimizer, 
        params = [
            {"params": model.encoder.parameters(), "lr": CFG.learning_rate*0.1},
            {"params": model.lstm.parameters(), "lr": CFG.learning_rate},
            {"params": model.fc.parameters(), "lr": CFG.learning_rate}, 
        ], 
        lr = CFG.learning_rate, 
        **CFG.optimizer_params)

    # get scheduler from scheduler.py

    # get criterion from criterion.py
    criterion = create_criterion(
        CFG.criterion,
        **CFG.criterion_params)

    return model, optimizer, criterion


def train(model, optimizer, criterion, train_dataloader, validation_dataloader, tokenizer, fold):
    # Set metrics
    metric_fn = Hitrate

    # Set trainer
    trainer = Trainer(model, CFG.device, criterion, metric_fn, tokenizer, optimizer, logger=CFG.system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=CFG.early_stopping_patience, verbose=True, logger=CFG.system_logger)

    # Set performance recorder
    key_column_value_list = [
        CFG.TRAIN_SERIAL,
        CFG.TRAIN_TIMESTAMP,
        CFG.early_stopping_patience,
        CFG.batch_size,
        CFG.epochs,
        CFG.learning_rate,
        CFG.seed]

    best_score = 0
    for epoch_index in range(CFG.epochs):
        trainer.train_epoch(train_dataloader, epoch_index=epoch_index)
        trainer.validate_epoch(validation_dataloader, epoch_index=epoch_index)

        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

        if early_stopper.stop:
            print('Early stopped')
            break

        if trainer.validation_score >= best_score:
            best_score = trainer.validation_score
            torch.save(model, f"models/Ik_fold{fold}.pt")


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU ě´ëŚ: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser(description="AIOnlineCompetition")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'train config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config from custom config.json file
    get_train_config(CFG, os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config))

    # set every random seed
    seed_everything(CFG.seed)

    # Set train result directory
    make_directory(CFG.PERFORMANCE_RECORD_DIR)

    # Save config json file
    with open(os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config)) as f:
        config = json.load(f)
    save_json(os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_config.json'), config)

    # Set system logger
    CFG.system_logger = get_logger(name='train',
                                   file_path=os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # set pytorch dataset & loader
    data_set = get_data_utils()

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    
    kfold = KFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_set)):
        print(fold)
        # get model, optimizer, criterion(not for this task), and scheduler
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_dataloader = DataLoader(dataset=data_set, batch_size=CFG.batch_size, sampler=train_subsampler)
        validation_dataloader = DataLoader(dataset=data_set, batch_size=CFG.batch_size, sampler=val_subsampler)

        model, optimizer, criterion = get_model(train_dataloader)
        model.to(CFG.device)
        # train
        train(model, optimizer, criterion, train_dataloader, validation_dataloader, tokenizer, fold)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()