"""Trainer 클래스 정의
"""

import torch
import random
from tqdm import tqdm

class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    
    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
    """

    def __init__(self, model, device, criterion, metric_fn, tokenizer, optimizer=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.logger = logger
        self.tokenizer= tokenizer

    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        sample_num = 0
        pred_lst = []
        target_lst = []
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)

        self.model.zero_grad()
        for step, (input_ids, token_type_ids, attention_mask, target) in pbar:
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            target = target.float().to(self.device)

            sent_score = self.model(input_ids, token_type_ids, attention_mask)

            loss = self.criterion(sent_score, target)
            self.train_total_loss += loss
            sample_num += target.shape[0]

            loss.backward()
            if (step + 1) % 4 == 0 or step == len(dataloader) - 1 :
                self.optimizer.step()
                self.model.zero_grad()

            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

            description = f"loss: {self.train_total_loss/sample_num:.8f}"
            pbar.set_description(description)
            
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
        msg = f'          Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)


    def validate_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        pred_lst = []
        target_lst = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True)

        with torch.no_grad():
            for batch_index, (input_ids, token_type_ids, attention_mask, target) in pbar:
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                target = target.float().to(self.device)

                sent_score = self.model(input_ids, token_type_ids, attention_mask)
                loss = self.criterion(sent_score, target)
                self.val_total_loss += loss

                pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
                target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
            msg = f'          Epoch {epoch_index}, Validation, loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)



