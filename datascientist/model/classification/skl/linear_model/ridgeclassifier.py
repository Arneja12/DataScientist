from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifier
import numpy as np


def _ridgeclassifier(*, train, test, x_predict=None, metrics, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
        max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier
    """

    model = RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, max_iter=max_iter,
        tol=tol, class_weight=class_weight, solver=solver, random_state=random_state)
    model.fit(train[0], train[1])
    model_name = 'RidgeClassifier'
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
    