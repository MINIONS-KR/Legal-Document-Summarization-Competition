"""PerformanceRecorder
    * 성능 기록

TODO:
    * best.pt, last,pt 저장

NOTES:

REFERENCE:
    * MNC 코드 템플릿 modules/recorders.py

UPDATED:
"""

import os
import csv
import numpy as np
import logging
from matplotlib import pyplot as plt

import torch

from modules.utils import make_directory, count_csv_row


class PerformanceRecorder():

    def __init__(self,
                 column_name_list: list,
                 record_dir: str,
                 key_column_value_list: list,
                 model: 'model',
                 optimizer: 'optimizer',
                 scheduler: 'scheduler',
                 logger: logging.RootLogger=None):
        """Recorder 초기화
            
        Args:
            column_name_list (list(str)):
            record_dir (str):
            key_column_value_list (list)

        Note:
        """

        self.column_name_list = column_name_list
        
        self.record_dir = record_dir
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')
        self.best_record_filepath = os.path.join('/'.join(self.record_dir.split('/')[:-1]),'train_best_record.csv')
        self.weight_path = os.path.join(record_dir, 'model.pt')

        self.row_counter = 0
        self.key_column_value_list = key_column_value_list

        self.train_loss_list = list()
        self.validation_loss_list= list()
        self.train_score_list = list()
        self.validation_score_list = list()

        self.loss_plot = None
        self.score_plot = None

        self.min_loss = np.Inf
        self.best_record = None

        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.key_column_value_list = key_column_value_list

    def set_model(self, model: 'model'):
        self.model = model

    def set_logger(self, logger: logging.RootLogger):
        self.logger = logger

    def create_record_directory(self):
        """
        record 경로 생성
        """
        make_directory(self.record_dir)
        msg = f"Create directory {self.record_dir}"
        self.logger.info(msg) if self.logger else None

    def add_row(self,
                epoch_index: int,
                train_loss: float,
                validation_loss: float,
                train_score: float,
                validation_score: float):
        """Epoch 단위 성능 적재
        
        최고 성능 Epoch 모니터링
        모든 Epoch 종료 이후 최고 성능은 train_best_records.csv 에 적재

        Args:
            row (list): 

        """
        self.train_loss_list.append(train_loss)
        self.validation_loss_list.append(validation_loss)
        self.train_score_list.append(train_score)
        self.validation_score_list.append(validation_score)

        row = self.key_column_value_list + [epoch_index, train_loss, validation_loss, train_score, validation_score]
        
        # Update scheduler learning rate
        learning_rate = self.scheduler.get_last_lr()[0]
        row[9] = learning_rate

        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.writer(f)

            if self.row_counter == 0:
                writer.writerow(self.column_name_list)

            writer.writerow(row)
            msg = f"Write row {self.row_counter}"
            self.logger.info(msg) if self.logger else None

        self.row_counter += 1

        # Update best record & Save check point
        if validation_loss < self.min_loss:
            msg = f"Update best record row {self.row_counter}, checkpoints {self.min_loss} -> {validation_loss}"
            
            self.min_loss = validation_loss
            self.best_record = row
            self.save_weight()

            self.logger.info(msg) if self.logger else None

    def add_best_row(self):
        """
        모든 Epoch 종료 이후 최고 성능에 해당하는 row을 train_best_records.csv 에 적재
        """

        n_row = count_csv_row(self.best_record_filepath)

        with open(self.best_record_filepath, newline='', mode='a') as f:
            writer = csv.writer(f)

            if n_row == 0:
               writer.writerow(self.column_name_list)
            
            writer.writerow(self.best_record)

        msg = f"Save best record {self.best_record_filepath}"
        self.logger.info(msg) if self.logger else None

    def save_weight(self)-> None:
        """Weight 저장

        Args:
            loss (float): validation loss
            model (`model`): model
        
        """
        check_point = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(check_point, self.weight_path)
        #torch.save(self.model.state_dict(), self.weight_path)
        msg = f"Model saved: {self.weight_path}"
        self.logger.info(msg) if self.logger else None

    def save_performance_plot(self, final_epoch: int):
        """Epoch 단위 loss, score plot 생성 후 저장

        """
        self.loss_plot = self.plot_performance(epoch=final_epoch+1,
                                          train_history=self.train_loss_list,
                                          validation_history=self.validation_loss_list,
                                          target='loss')
        self.score_plot = self.plot_performance(epoch=final_epoch+1,
                                           train_history=self.train_score_list,
                                           validation_history=self.validation_score_list,
                                           target='score')

        self.loss_plot.savefig(os.path.join(self.record_dir, 'loss.png'))
        self.score_plot.savefig(os.path.join(self.record_dir, 'score.jpg'))
        plt.close('all')


        msg = f"Save performance plot {self.record_dir}"
        self.logger.info(msg) if self.logger else None

    def plot_performance(self,
                         epoch: int,
                         train_history:list,
                         validation_history:list,
                         target:str)-> plt.figure:
        """loss, score plot 생성

        """
        fig = plt.figure(figsize=(12, 5))
        epoch_range = list(range(epoch))

        plt.plot(epoch_range, train_history, marker='.', c='red', label="train")
        plt.plot(epoch_range, validation_history, marker='.', c='blue', label="validation")

        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel(target)

        # plt.xticks(epoch_range, [str(i) for i in epoch_range])

        return fig