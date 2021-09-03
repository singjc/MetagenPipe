from sklearn import metrics


def AUROC_SCORE(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: macro average of ROC AUC score (AUROC)
    """
    return metrics.roc_auc_score(y_true, y_pred, average='macro')


def SENSITIVITY_SCORE(y_true, y_pred):
    """
    Note: This function does not work with LogisticRegressionCV, as there is some internal conversion of 0, 1 to -1, 1,
            but, for an input of 0, 1, LogisticRegressionCV will output 0, 1 for final prediction
    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: sensitivity, assuming a positive label of 1.
    """
    return metrics.recall_score(y_true, y_pred, average='binary', pos_label=1)


def SPECIFICITY_SCORE(y_true, y_pred):
    """
    Note: This function does not work with LogisticRegressionCV, as there is some internal conversion of 0, 1 to -1, 1,
            but, for an input of 0, 1, LogisticRegressionCV will output 0, 1 for final prediction
    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: specificity, assuming a positive label of 1.
    """
    return metrics.recall_score(y_true, y_pred, average='binary', pos_label=0)


def PPV_SCORE(y_true, y_pred):
    """
    Note: This function does not work with LogisticRegressionCV, as there is some internal conversion of 0, 1 to -1, 1,
            but, for an input of 0, 1, LogisticRegressionCV will output 0, 1 for final prediction
    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: positive predictive value (precision), assuming a positive label of 1.
    """
    return metrics.precision_score(y_true, y_pred, average='binary', pos_label=1)


def NPV_SCORE(y_true, y_pred):
    """
    Note: This function does not work with LogisticRegressionCV, as there is some internal conversion of 0, 1 to -1, 1,
            but, for an input of 0, 1, LogisticRegressionCV will output 0, 1 for final prediction
    :param y_true: true values of y
    :param y_pred: predicted values of y
    :return: negative predictive value (precision of negative label), assuming a positive label of 1.
    """
    return metrics.precision_score(y_true, y_pred, average='binary', pos_label=0)
