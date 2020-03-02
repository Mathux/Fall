import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def make_scoring(is_classification):
    # Scoring
    if is_classification:
        def score(y, pred):
            # Find the closest to 0 or 1
            if len(pred.shape) == 1:
                pred = pred[:, np.newaxis]
            ypred = np.argmin(np.hstack((np.abs(pred - 0),
                                         np.abs(pred - 1))),
                              axis=1)
            return accuracy_score(y, ypred)
        scoring = make_scorer(score)
        scoring.best = np.argmax
        scoring.first = "max"
        scoring.name = "Accuracy"
        scoring.direction = "maximize"
    else:
        scoring = make_scorer(mean_squared_error,
                              greater_is_better=True)
        scoring.best = np.argmin
        scoring.first = "min"
        scoring.name = "MSE"
        scoring.direction = "minimize"
    return scoring
