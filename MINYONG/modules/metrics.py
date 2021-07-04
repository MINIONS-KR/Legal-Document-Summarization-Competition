
import numpy as np


def Hitrate(y_true, y_pred):
    """ Metric 함수 반환하는 함수

    Returns:
        metric_fn (Callable)
    """
    hitrate = np.array([len(list(set(ans).intersection(y_true[i])))/3 for i, ans in enumerate(y_pred)])
    score = np.mean(hitrate)
    return score


