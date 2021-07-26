import numpy as np


def get_metric(targets, preds):
    """ Metric 함수 반환하는 함수

    Returns:
        metric_fn (Callable)
    """
    hitrate = np.array([len(list(set(ans).intersection(targets[i])))/3 for i, ans in enumerate(preds)])
    score = np.mean(hitrate)
    return score


