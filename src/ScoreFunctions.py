from sklearn import metrics
import numpy as np

def AUROC_SCORE(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: y_pred: predicted probabilities of y
    :return: macro average of ROC AUC score (AUROC), or just AUROC if y_true is binary
    """
    return metrics.roc_auc_score(y_true, y_pred, average='macro')

def AUROC_NEG(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: y_pred: predicted probabilities of y
    :return: AUROC for negative class. expect binary input for y_true and y_pred.
    """
    assert np.all(np.in1d(y_true, [0, 1]))
    assert len(y_true.shape) == 1
    assert len(y_pred.shape) == 1
    y_true = np.array([1], dtype='int32') - y_true
    y_pred = np.array([1], dtype='float32') - y_pred
    return metrics.roc_auc_score(y_true, y_pred)


def AUPRC_SCORE(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: predicted probabilities of y
    :return: macro average of AUPRC curve, or just AUPRC if y_true is binary
    """
    return metrics.average_precision_score(y_true, y_pred, average='macro')

def AUPRC_NEG(y_true, y_pred):
    """

    :param y_true: true values of y
    :param y_pred: y_pred: predicted probabilities of y
    :return: AUPRC for negative class. expect binary input for y_true and y_pred.
    """
    assert np.all(np.in1d(y_true, [0, 1]))
    assert len(y_true.shape) == 1
    assert len(y_pred.shape) == 1
    y_true = np.array([1], dtype='int32') - y_true
    y_pred = np.array([1], dtype='float32') - y_pred
    return metrics.average_precision_score(y_true, y_pred)


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
