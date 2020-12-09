from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifierCV
import numpy as np


def _ridgeclassifiercv(*, train, test, x_predict=None, metrics, alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False,
        scoring=None, cv=None, class_weight=None, store_cv_values=False):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV
    """

    model = RidgeClassifierCV(alphas=alphas, fit_intercept=fit_intercept, normalize=normalize, scoring=scoring, cv=cv,
            class_weight=class_weight, store_cv_values=store_cv_values)
    model.fit(train[0], train[1])
    model_name = 'RidgeClassifierCV'
    y_hat = model.predict(test[0])

    if metrics == 'f1_score':
        accuracy = f1_score(test[1], y_hat)
    if metrics == 'jaccard_score':
        accuracy = jaccard_score(test[1], y_hat)
    if metrics == 'accuracy_score':
        accuracy = accuracy_score(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
    