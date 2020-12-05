from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
import numpy as np


def _logisticregression(*, train, test, x_predict=None, metrics, penalty='l2', dual=False, tol=0.0001, 
        C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 
        solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
        l1_ratio=None):
    """For for info visit : 
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    """

    model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state,
        solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start,
        n_jobs=n_jobs, l1_ratio=l1_ratio)
    model.fit(train[0], train[1])
    model_name = 'LogisticRegression'
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