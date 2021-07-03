import warnings
warnings.filterwarnings('ignore')

from modules.utils import get_logger, make_directory, get_train_config, seed_everything, save_json
from modules.criterion import create_criterion
from modules.optimizer import create_optimizer
from modules.scheduler import create_scheduler
from modules.earlystoppers import ScoreEarlyStopper
from modules.metrics import Hitrate
from modules.recorders import PerformanceRecorder

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

from datetime import datetime, timezone, timedelta
from importlib import import_module
import os
import json
import argparse


class CFG:
    # Project Environment
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = '/DATA/Final_DATA/task05_train'
    MODEL_DIR = './models'

    # TRAIN SERIAL
    RESULT_DIR = "BERTSUM_FINAL"

    # DEVICE
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    # split train & test data
    # (1) random split
    # train_data, valid_data = train_test_split(train_df, test_size=0.1, random_state=CFG.seed)
    
    
    # (2) random KFold
    kf = KFold(n_splits=5, random_state=CFG.seed, shuffle=True)
    for train_index, test_index in kf.split(train_df):
        train_data = train_df.iloc[train_index]
        valid_data = train_df.iloc[test_index]


    # (3) Stratified KFold with the number of sentence
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
    # sen_count = list(train_df.article_original.apply(lambda x: len(x)))
    # for train_index, test_index in skf.split(train_df, sen_count):
    #     train_data = train_df.iloc[train_index]
    #     valid_data = train_df.iloc[test_index]


    # get train & valid dataset from dataset.py
    dataset = getattr(import_module("modules.dataset"), CFG.dataset)
    train_dataset = dataset(train_data, data_dir=CFG.DATA_DIR, mode='train', data_aug=CFG.data_aug, add_mask=CFG.add_mask)
    validation_dataset = dataset(valid_data, data_dir=CFG.DATA_DIR, mode='valid')

    # define data loader based on each dataset
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=CFG.batch_size,
                                  num_workers=CFG.num_workers,
                                  pin_memory=CFG.pin_memory,
                                  drop_last=CFG.drop_last,
                                  shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=CFG.batch_size,
                                       num_workers=CFG.num_workers,
                                       pin_memory=CFG.pin_memory,
                                       drop_last=CFG.drop_last,
                                       shuffle=False)

    return train_dataloader, validation_dataloader


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
    model = model_module(**CFG.model_params).to(CFG.device)
    model.train()

    # CFG.system_logger.info('===== Review Model Architecture =====')
    # CFG.system_logger.info(f'{model} \n')

    # get optimizer from optimizer.py
    if CFG.model == 'BertExt':
            optimizer = create_optimizer(
            CFG.optimizer,
            params = [
                {"params": model.bert.parameters(), "lr": CFG.learning_rate * CFG.backbone_ratio},
                {"params": model.ext_layer.parameters()},
            ],
            lr = CFG.learning_rate,
            **CFG.optimizer_params)
    
    else:
        optimizer = create_optimizer(
            CFG.optimizer,
            params = [
                {"params": model.encoder.parameters(), "lr": CFG.learning_rate * CFG.backbone_ratio},
                {"params": model.fc.parameters()},
            ],
            lr = CFG.learning_rate,
            **CFG.optimizer_params)


    # get scheduler from scheduler.py
    scheduler = create_scheduler(
        CFG.scheduler,
        optimizer = optimizer,
        num_warmup_steps= CFG.epochs * len(train_dataloader) / 5,
        num_training_steps= CFG.epochs * len(train_dataloader))
        # **CFG.scheduler_params)

    # get criterion from criterion.py
    criterion = create_criterion(
        CFG.criterion,
        **CFG.criterion_params)

    return model, optimizer, scheduler, criterion



def validation(model, dataloader, criterion, metric_fn):
    model.eval()
    val_total_loss = 0
    pred_lst = []
    target_lst = []

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(dataloader):
            src = data[0].to(CFG.device)
            clss = data[1].to(CFG.device)
            segs = data[2].to(CFG.device)
            mask = data[3].to(CFG.device)
            mask_clss = data[4].to(CFG.device)
            target = target.float().to(CFG.device)

            sent_score = model(src, segs, clss, mask, mask_clss)
            loss = criterion(sent_score, target)
            loss = (loss * mask_clss.float()).sum()
            val_total_loss += loss.item()

            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

        val_mean_loss = val_total_loss / len(dataloader)
        validation_score = metric_fn(y_true=target_lst, y_pred=pred_lst)
    
    model.train()
    
    return val_mean_loss, validation_score


def train(model, optimizer, scheduler, criterion, train_dataloader, validation_dataloader, fold):
    # Set metrics
    metric_fn = Hitrate

    # Set earlystopper
    early_stopper = ScoreEarlyStopper(patience=CFG.early_stopping_patience, verbose=True, logger=CFG.system_logger)

    # Set performance recorder
    key_column_value_list = [
        CFG.early_stopping_patience,
        CFG.batch_size,
        CFG.epochs,
        CFG.learning_rate,
        CFG.seed]

    performance_recorder = PerformanceRecorder(column_name_list=CFG.PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=CFG.PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=CFG.system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)


    # train phase
    best_score = 0
    global_epoch = 0

    for epoch_index in range(CFG.epochs):
        seg = "---------------------------------------------------------------------------------------------"
        start_msg = f"Epoch {epoch_index}"
        print(seg)
        print(start_msg)
        CFG.system_logger.info(seg) if CFG.system_logger else print(seg)
        CFG.system_logger.info(start_msg) if CFG.system_logger else print(start_msg)

        train_total_loss = 0
        pred_lst = []
        target_lst = []

        for step, (data, target) in enumerate(train_dataloader):
            src = data[0].to(CFG.device)
            clss = data[1].to(CFG.device)
            segs = data[2].to(CFG.device)
            mask = data[3].to(CFG.device)
            mask_clss = data[4].to(CFG.device)

            target = target.float().to(CFG.device)

            sent_score = model(src, segs, clss, mask, mask_clss)
            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

            loss = criterion(sent_score, target)
            loss = (loss * mask_clss.float()).sum()
            train_total_loss += loss.item()
            loss = loss / CFG.grad_accum

            loss.backward()

            if ((step + 1) % CFG.grad_accum == 0) or (step == len(train_dataloader) - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            scheduler.step()


            if ((step + 1) % CFG.valid_step == 0):
                val_mean_loss, validation_score = validation(model, validation_dataloader, criterion, metric_fn)
                
                # early_stopping check
                early_stopper.check_early_stopping(score=validation_score)

                if early_stopper.stop:
                    print('Early stopped')
                    return

                if validation_score > best_score:
                    best_score = validation_score
                    performance_recorder.weight_path = os.path.join(CFG.MODEL_DIR, f'bertsum{fold}.pt')
                    performance_recorder.save_weight()
            
                train_mean_loss = train_total_loss / CFG.valid_step
                train_score = metric_fn(y_true=target_lst, y_pred=pred_lst)
                train_msg = f'Step {step + 1}, Train, loss: {train_mean_loss}, Score: {train_score}'
                val_msg = f'Step {step + 1}, Valid, loss: {val_mean_loss}, Score: {validation_score}'
                print(train_msg)
                print(val_msg)
                print()

                pred_lst = []
                target_lst = []
                train_total_loss = 0

                CFG.system_logger.info(train_msg) if CFG.system_logger else print(train_msg)
                CFG.system_logger.info(val_msg) if CFG.system_logger else print(val_msg)


                # Performance record - csv & save elapsed_time
                performance_recorder.add_row(epoch_index=global_epoch,
                                            train_loss=train_mean_loss,
                                            validation_loss=val_mean_loss,
                                            train_score=train_score,
                                            validation_score=validation_score)

                # Performance record - plot
                performance_recorder.save_performance_plot(final_epoch=global_epoch)

                global_epoch += 1


def main():
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # select config file
    parser = argparse.ArgumentParser(description="AIOnlineCompetition")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'train config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config from custom config.json file
    get_train_config(CFG, os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config))

    # set every random seed
    seed_everything(CFG.seed)

    # Set train result directory
    make_directory(CFG.PERFORMANCE_RECORD_DIR)

    # Save train config json file
    with open(os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config)) as f:
        config = json.load(f)
    save_json(os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_config.json'), config)

    # Set system logger
    CFG.system_logger = get_logger(name='train', file_path=os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_log.log'))


    # KFOLD Ensemble
    # get data from json
    with open(os.path.join(CFG.DATA_DIR, "train.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    train_df = pd.DataFrame(data)
    

    # (2) random KFold
    kf = KFold(n_splits=5, random_state=CFG.seed, shuffle=True)
    for fold, (train_index, test_index) in enumerate(kf.split(train_df)):
        train_data = train_df.iloc[train_index]
        valid_data = train_df.iloc[test_index]

        # get train & valid dataset from dataset.py
        dataset = getattr(import_module("modules.dataset"), CFG.dataset)
        train_dataset = dataset(train_data, data_dir=CFG.DATA_DIR, mode='train', data_aug=CFG.data_aug, add_mask=CFG.add_mask)
        validation_dataset = dataset(valid_data, data_dir=CFG.DATA_DIR, mode='valid')

        # define data loader based on each dataset
        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=CFG.batch_size,
                                    num_workers=CFG.num_workers,
                                    pin_memory=CFG.pin_memory,
                                    drop_last=CFG.drop_last,
                                    shuffle=True)
        validation_dataloader = DataLoader(dataset=validation_dataset,
                                        batch_size=CFG.batch_size,
                                        num_workers=CFG.num_workers,
                                        pin_memory=CFG.pin_memory,
                                        drop_last=CFG.drop_last,
                                        shuffle=False)

        # get model, optimizer, criterion(not for this task), and scheduler
        model, optimizer, scheduler, criterion = get_model(train_dataloader)

        # fold msg
        msg = f'Fold {fold + 1}'
        print(msg)
        print()
        CFG.system_logger.info(msg) if CFG.system_logger else print(msg)

        # train process
        train(model, optimizer, scheduler, criterion, train_dataloader, validation_dataloader, fold)


if __name__ == "__main__":
    main()