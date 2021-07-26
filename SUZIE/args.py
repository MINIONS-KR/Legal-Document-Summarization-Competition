import os
import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')

    # Directory Setting
    parser.add_argument('--data_dir', default='/content/drive/MyDrive/Legal-Document-Summarization-Competition/data', type=str, help='data directory')
    parser.add_argument('--model_dir', default='/content/drive/MyDrive/Legal-Document-Summarization-Competition/model', type=str, help='model directory')
    parser.add_argument('--submission_dir', default='/content/drive/MyDrive/Legal-Document-Summarization-Competition/submission', type=str, help='submission directory')
                    
    # Name Setting
    parser.add_argument('--train_file_name', default='train.json', type=str, help='train file name')
    parser.add_argument('--test_file_name', default='test.json', type=str, help='test file name')
    parser.add_argument('--test_model_name', default='suz.pt', type=str, help='test file name')
    parser.add_argument('--submission_name', default='0626_submission.json', type=str, help='submission file name')
    parser.add_argument('--run_name', default='suz', type=str, help='wandb run name')    
    
    # Model Setting
    parser.add_argument('--model', default='base', type=str, help='model type')
    parser.add_argument('--model_name', default='klue/bert-base', type=str, help='pretrained model name')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='step', type=str, help='scheduler type')
    parser.add_argument('--max_seq_len', default=10, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=5, type=int, help='number of workers')
    parser.add_argument('--hidden_dim', default=768, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=3, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=4, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop out rate')
    parser.add_argument('--criterion', default='bce', type=str, help='criterion name')
    
    # Trainer Setting
    parser.add_argument('--n_epochs', default=3, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--accum_steps', default=4, type=int, help='accumulation steps') # 16 * 4 = 64 batch size의 효과..?
    parser.add_argument('--clip_grad', default=1, type=int, help='clip grad')
    parser.add_argument('--patience', default=5, type=int, help='for early stopping')

    # ETC
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')
    parser.add_argument('--fold', default=0, type=int, help='fold number')
    parser.add_argument('--train_kfold', default=False, type=bool, help='train with kfold')
    parser.add_argument('--inference_kfold', default=False, type=bool, help='inference with kfold')
    parser.add_argument('--partial_dataset', default='None', type=str, help='train with partial dataset')
    
    args = parser.parse_args()

    return args