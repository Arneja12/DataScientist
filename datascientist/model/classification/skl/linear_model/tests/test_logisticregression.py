from datascientist.model.classification.skl.linear_model.logisticregression import _logisticregression

import numpy as np


def test_logisticregression():
    x_train = np.reshape([100, 120, 160, 180], (-1,2))
    y_train = np.array([0, 0, 1, 1])

    x_test = np.reshape([90, 200, 60, 70], (-1,2))
    y_test = np.array([0, 1, 0, 0])

    metrics = 'f1_score'
    answer = _logisticregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    #assert answer[0] == 'Ridge'
    #assert round(answer[1], 2) == 0.40
    #assert answer[2] is None

    metrics = 'jaccard_score'
    answer = _logisticregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    #assert answer[0] == 'Ridge'
    #assert round(answer[1], 2) == 0.25
    #assert answer[2] is None

    metrics = 'accuracy_score'
    answer = _logisticregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics)
    print(answer)
    #assert answer[0] == 'Ridge'
    #assert round(answer[1], 2) == 0.50
    #assert answer[2] is None

    answer = _logisticregression(train=(x_train, y_train), test=(x_test, y_test), metrics=metrics, x_predict=x_test)
    print(answer)
    #assert np.any(answer[2] == np.array([6.7,  8.1,  8.9, 10.3]))

test_logisticregression()
