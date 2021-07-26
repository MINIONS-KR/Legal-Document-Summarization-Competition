from torch.optim import Adam, AdamW


def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    optimizer.zero_grad()
    
    return optimizer