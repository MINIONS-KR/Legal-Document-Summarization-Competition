import torch
import torch.nn as nn

# name과 optimizer class를 묶는 entrypoint
_optimizer_entrypoints = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
}


# optimizer 이름을 가지는 class return
def optimizer_entrypoint(optimizer_name):
    return _optimizer_entrypoints[optimizer_name]


# optimizer list에 전달된 이름이 존재하는지 판단
def is_optimizer(optimizer_name):
    return optimizer_name in _optimizer_entrypoints


def create_optimizer(optimizer_name, **kwargs):
    # 정의한 optimizer list내에 이름이 있으면
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name) # class를 가져와서
        optimizer = create_fn(**kwargs) # 전달된 인자를 사용해 선언
        
    else:
        raise RuntimeError('Unknown optimizer (%s)' % optimizer_name) # 없을 시, Error raise
        
    return optimizer # defined criterion return