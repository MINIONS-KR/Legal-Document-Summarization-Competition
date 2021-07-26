from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, mode='max', verbose=True)
    elif args.scheduler == 'linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'onecycle':
        #scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=1, steps_per_epoch=len(train_dataloader))
        scheduler = OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001)
        
    return scheduler