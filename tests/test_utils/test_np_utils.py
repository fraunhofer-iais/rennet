"""
@motjuste
Created: 08-02-2017

Test the Numpy Utilities
"""
import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from rennet.utils import np_utils as nu

# pylint: disable=redefined-outer-name

## TRUE LABELS OF 1D AND PREDICTIONS AND CONFUSIONS ###########################

labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1])

predictions = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],  # Exact
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],  # No 2_0
    [0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1],  # No 1_0
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # No 2
    [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # No 0
    [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0],  # None correct
    [4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0]  # Extra class
])

confusion = np.array([
    [[4, 0, 0], [0, 6, 0], [0, 0, 4]],
    [[2, 2, 0], [2, 2, 2], [0, 3, 1]],
    [[2, 0, 2], [0, 4, 2], [2, 0, 2]],
    [[1, 3, 0], [2, 4, 0], [2, 2, 0]],
    [[0, 3, 1], [0, 4, 2], [0, 4, 0]],
    [[0, 3, 1], [2, 0, 4], [1, 3, 0]],
])

confrecall = np.array([
    [[1.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 1.00]],
    [[0.50, 0.50, 0.00], [0.33, 0.33, 0.33], [0.00, 0.75, 0.25]],
    [[0.50, 0.00, 0.50], [0.00, 0.67, 0.33], [0.50, 0.00, 0.50]],
    [[0.25, 0.75, 0.00], [0.33, 0.67, 0.00], [0.50, 0.50, 0.00]],
    [[0.00, 0.25, 0.75], [0.00, 0.67, 0.33], [0.00, 1.00, 0.00]],
    [[0.00, 0.75, 0.25], [0.33, 0.00, 0.67], [0.25, 0.75, 0.00]],
])

confprec = np.array([
    [[1.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 1.00]],
    [[0.50, 0.28, 0.00], [0.50, 0.28, 0.67], [0.00, 0.43, 0.33]],
    [[0.50, 0.00, 0.50], [0.00, 1.00, 0.00], [0.50, 0.00, 0.50]],
    [[0.20, 0.33, np.nan], [0.40, 0.44, np.nan], [0.40, 0.22, np.nan]],
    [[np.nan, 0.27, 0.33], [np.nan, 0.36, 0.67], [np.nan, 0.36, 0.00]],
    [[0.00, 0.50, 0.25], [0.67, 0.00, 0.75], [0.33, 0.50, 0.00]],
])

## CONFUSION MATRIX TESTS #####################################################


@pytest.fixture(scope='module', params=list(range(len(confusion))))
def normal_preds_confusion(request):
    return {
        "labels": labels,
        "predictions": predictions[request.param, ...],
        "confusion": confusion[request.param, ...],
    }


def test_normal_confusion_matrix(normal_preds_confusion):
    labels = normal_preds_confusion['labels']
    preds = normal_preds_confusion['predictions']
    true_confusion = normal_preds_confusion['confusion']

    assert_almost_equal(true_confusion, nu.confusion_matrix(labels, preds))


@pytest.fixture(scope='module', params=[6, ])
def extra_class_preds_confusion(request):
    return {
        "labels": labels,
        "predictions": predictions[request.param, ...],
    }


def test_extra_pred_label_raises(extra_class_preds_confusion):
    labels = extra_class_preds_confusion['labels']
    preds = extra_class_preds_confusion['predictions']

    # NOTE: The Exception is raised while converting the preds to categorical
    with pytest.raises(RuntimeError):
        nu.confusion_matrix(labels, preds)


def test_missing_label_warns(extra_class_preds_confusion):
    preds = extra_class_preds_confusion['labels']  # has extra class
    labels = extra_class_preds_confusion['predictions']

    # NOTE: The Exception is raised while converting the preds to categorical
    with pytest.raises(RuntimeWarning):
        nu.confusion_matrix(labels, preds, warn=True)

    # Should not raise warning
    nu.confusion_matrix(labels, preds)


## CONF-PRECISION AND -RECALL TESTS ###########################################


@pytest.fixture(scope='module', params=[0, 1,])# 2, 5])
def normal_preds_conf_prec_rec(request):
    return {
        "confusion": confusion[request.param, ...],
        "confprec": confprec[request.param, ...],
        "confrecall": confrecall[request.param, ...],
    }


def test_individual_preds_conf_prec_rec(normal_preds_conf_prec_rec):
    confusion = normal_preds_conf_prec_rec['confusion']
    true_confprec = normal_preds_conf_prec_rec['confprec']
    true_confrecall = normal_preds_conf_prec_rec['confrecall']

    confprec, confrecall = nu.normalize_confusion_matrix(confusion)
    assert_almost_equal(true_confprec, confprec, decimal=2)
    # assert_almost_equal(true_confrecall, confrecall)
