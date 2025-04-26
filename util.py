import numpy as np

def rmse(y, yhat):
    if yhat.ndim == 3:
        yhat = np.mean(yhat, 0)
    return np.sqrt(np.mean(np.square(y-yhat),0))

def coverage(y, yhat):
    yhat_mean = np.mean(yhat, 0)
    yhat_sd = np.std(yhat, 0)
    c = np.logical_and(yhat_mean-2*yhat_sd <= y, yhat_mean+2*yhat_sd >= y)
    return np.mean(c, 0)

def width(yhat):
    yhat_sd = np.std(yhat, 0)
    return np.mean(2*yhat_sd,0)

def interval_score(y,yhat):
    yhat_mean = np.mean(yhat, 0)
    yhat_sd = np.std(yhat, 0)
    w = np.mean(width(yhat))
    p1 = 2/0.95 * np.sum(y < yhat_mean-2*yhat_sd)
    p2 = 2/0.95 * np.sum(y > yhat_mean+2*yhat_sd)
    return w + p1 + p2