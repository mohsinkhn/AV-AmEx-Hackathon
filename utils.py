import math
import pandas as pd
import numpy as np


#@author: Mohsin
def woe(X, y, cont=True):
    '''
    Calculate weights of evidence and information value for checking a feature importance in classification tasks.
    For continous features, we first bin them in 255 bins

    :param X: (array) : numpy array of feature to check.
    :param y: (array) : Target array (should be 0 and 1 values)
    :param cont: (bool) : Whether feature is continous or not
    :return: (Series, Series, float): Returns series for WOE and IV, along with IV value
    '''
    tmp = pd.DataFrame()
    tmp["variable"] = X
    if cont:
        tmp["variable"] = pd.qcut(tmp["variable"], 255, duplicates="drop")
    tmp["target"] = y
    var_counts = tmp.groupby("variable")["target"].count()
    var_events = tmp.groupby("variable")["target"].sum()
    var_nonevents = var_counts - var_events
    tmp["var_counts"] = tmp.variable.map(var_counts)
    tmp["var_events"] = tmp.variable.map(var_events)
    tmp["var_nonevents"] = tmp.variable.map(var_nonevents)
    events = sum(tmp["target"] == 1)
    nonevents = sum(tmp["target"] == 0)
    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))
    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)
    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]
    iv = tmp.groupby("variable")["iv"].last().sum()
    return tmp["woe"], tmp["iv"], iv


#Taken from user: BlueSurfer , https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = math.e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)

    return ent


