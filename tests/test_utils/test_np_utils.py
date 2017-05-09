"""
@motjuste
Created: 08-02-2017

Test the Numpy Utilities
"""
from __future__ import division, print_function
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.metrics import confusion_matrix as ext_confusionmatrix
from keras.utils.np_utils import to_categorical as ext_tocategorical

from rennet.utils import np_utils as nu

# pylint: disable=redefined-outer-name


@pytest.fixture(scope='module')
def base_labels_cls3():
    """ sample array of class labels
    that can be recombined to act as predicitons of each other
    """

    return np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1],  # Exact
        [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],  # No 2_0
        [0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1],  # No 1_0
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # No 2
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # No 0
        [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0],  # None correct
        [4, 4, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0],  # Extra class
    ])


## FIXTURES AND TESTS FOR TO_CATEGORICAL ###########################################


@pytest.fixture(
    scope='module',
    params=list(range(len(base_labels_cls3()))),
    ids=lambda i: "TPU-(1, B, 1, 3)-{}".format(i)  #pylint: disable=unnecessary-lambda
)
def pred1_batB_seqL1_cls3_trues_preds_user_cat(request, base_labels_cls3):
    """ y and Y (categorical) in user expected format
    trues and preds look the same from the user's perspective when
    there is only one predictor
    """
    i = request.param
    y = base_labels_cls3[i]
    nclasses = max(y) + 1
    Y = ext_tocategorical(y, num_classes=nclasses)

    return {
        'y': y,
        'Y': Y,
        'nclasses': nclasses,
    }


def test_tocategorical_trues_preds_user(
        pred1_batB_seqL1_cls3_trues_preds_user_cat):
    y, Y, nc = [
        pred1_batB_seqL1_cls3_trues_preds_user_cat[k]
        for k in ['y', 'Y', 'nclasses']
    ]

    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=list(range(len(base_labels_cls3()))),
    ids=lambda i: "TG-(B, 1, 3)-{}".format(i)  #pylint: disable=unnecessary-lambda
)
def batB_seqL1_cls3_trues_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format expected by the generic function
    The generic representation expects:
    (Predictions, Batchsize, SequenceLength)

    For Trues, there is no Predictions dimension
    """
    i = request.param
    y = base_labels_cls3[i]
    nclasses = max(y) + 1
    Y = ext_tocategorical(y, num_classes=nclasses)

    return {
        'nclasses': nclasses,

        # Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': y[:, np.newaxis],

        # Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': Y[:, np.newaxis, :],
    }


def test_tocategorical_trues_generic(batB_seqL1_cls3_trues_generic_cat):
    y, Y, nc = [
        batB_seqL1_cls3_trues_generic_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=list(range(len(base_labels_cls3()))),
    ids=lambda i: "PG-(B, 1, 3)-{}".format(i)  #pylint: disable=unnecessary-lambda
)
def pred1_batB_seqL1_cls3_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format expected by the generic function
    The generic representation expects:
    (Predictions, Batchsize, SequenceLength)

    Here, there is only one prediction
    """
    i = request.param
    y = base_labels_cls3[i]
    nclasses = max(y) + 1
    Y = ext_tocategorical(y, num_classes=nclasses)

    return {
        'nclasses': nclasses,

        # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': y[np.newaxis, :, np.newaxis],

        # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': Y[np.newaxis, :, np.newaxis, :],
    }


def test_tocategorical_pred1_generic(pred1_batB_seqL1_cls3_generic_cat):
    y, Y, nc = [
        pred1_batB_seqL1_cls3_generic_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # P: number of predictions
    ids=lambda i: "PU-({}, B, 1, 3)".format(i)  #pylint: disable=unnecessary-lambda
)
def predP_batB_seqL1_cls3_user_cat(request, base_labels_cls3):
    """ y and Y (categorical) in user expected format
    there are P predictions
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, num_classes=nclasses) for yy in y]

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


def test_tocategorical_predP_user(predP_batB_seqL1_cls3_user_cat):
    y, Y, nc = [
        predP_batB_seqL1_cls3_user_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # P: number of predictions
    ids=lambda i: "PU-({}, B, 1, 3)".format(i)  #pylint: disable=unnecessary-lambda
)
def predP_batB_seqL1_cls3_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in user expected format
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, num_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Predictor=P, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': np.array(y)[..., np.newaxis],

        # Predictor=P, Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[..., np.newaxis, :],
    }


def test_tocategorical_predsP_generic(predP_batB_seqL1_cls3_generic_cat):
    y, Y, nc = [
        predP_batB_seqL1_cls3_generic_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    print(y.shape, Y.shape, nu.to_categorical(y, nc).shape)
    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # B: batchsize
    ids=lambda i: "TU-({}, Q, 3)".format(i)  #pylint: disable=unnecessary-lambda
)
def pred1_batB_seqlQ_cls3_trues_preds_user_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format the user expects
    this is exactly like predP_batB_seqL1_cls3_user_cat
    but these are trues
    non-zero sequence length trues and preds look similar for single predictor
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, num_classes=nclasses) for yy in y]

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


def test_tocategorical_batB_seqLQ_user(
        pred1_batB_seqlQ_cls3_trues_preds_user_cat):
    y, Y, nc = [
        pred1_batB_seqlQ_cls3_trues_preds_user_cat[k]
        for k in ['y', 'Y', 'nclasses']
    ]

    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # B: batchsize
    ids=lambda i: "TU-({}, Q, 3)".format(i)  #pylint: disable=unnecessary-lambda
)
def batB_seqlQ_cls3_trues_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format the user expects
    this is exactly like predP_batB_seqL1_cls3_user_cat
    but these are trues
    non-zero sequence length trues and preds look similar for single predictor
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, num_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y),

        # Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y),
    }


def test_tocategorical_batB_seqlQ_trues_generic(
        batB_seqlQ_cls3_trues_generic_cat):
    y, Y, nc = [
        batB_seqlQ_cls3_trues_generic_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    print(y.shape, Y.shape, nu.to_categorical(y, nc).shape)
    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # B: batchsize
    ids=lambda i: "TU-({}, Q, 3)".format(i)  #pylint: disable=unnecessary-lambda
)
def pred1_batB_seqlQ_cls3_preds_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format the user expects
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, num_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Predictor=1, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y)[np.newaxis, ...],

        # Predictor=1, Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[np.newaxis, ...],
    }


def test_tocategorical_pred1_batB_seqlQ_preds_generic(
        pred1_batB_seqlQ_cls3_preds_generic_cat):
    y, Y, nc = [
        pred1_batB_seqlQ_cls3_preds_generic_cat[k]
        for k in ['y', 'Y', 'nclasses']
    ]

    print(y.shape, Y.shape, nu.to_categorical(y, nc).shape)
    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[[list(range(2)), list(range(5))]],  # P: Predictions, B: batchsize
    ids=lambda i: "PU-({}, {}, Q, 3)".format(*i)  #pylint: disable=unnecessary-lambda
)
def predP_batB_seqlQ_cls3_preds_user_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format the user expects
    """
    p, b = request.param
    y = []
    nclasses = 0
    for _ in p:
        yy = []
        for i in b:
            yy.append(base_labels_cls3[i])
            nclasses = max(nclasses, max(base_labels_cls3[i]) + 1)
        y.append(yy)

    Y = []
    for yp in y:
        YY = []
        for yb in yp:
            YY.append(ext_tocategorical(yb, num_classes=nclasses))
        Y.append(YY)

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


def test_tocategorical_predP_batB_seqLQ_user(
        predP_batB_seqlQ_cls3_preds_user_cat):
    y, Y, nc = [
        predP_batB_seqlQ_cls3_preds_user_cat[k]
        for k in ['y', 'Y', 'nclasses']
    ]

    print(y.shape, Y.shape, nu.to_categorical(y, nc).shape)
    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


@pytest.fixture(
    scope='module',
    params=[[list(range(2)), list(range(5))]],  # P: Predictions, B: batchsize
    ids=lambda i: "PG-({}, {}, Q, 3)".format(*i)  #pylint: disable=unnecessary-lambda
)
def predP_batB_seqlQ_cls3_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format the user expects
    """
    p, b = request.param
    y = []
    nclasses = 0
    for _ in p:
        yy = []
        for i in b:
            yy.append(base_labels_cls3[i])
            nclasses = max(nclasses, max(base_labels_cls3[i]) + 1)
        y.append(yy)

    Y = []
    for yp in y:
        YY = []
        for yb in yp:
            YY.append(ext_tocategorical(yb, num_classes=nclasses))
        Y.append(YY)

    return {
        'nclasses': nclasses,

        # Predictor=P, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y)[...],

        # Predictor=P, Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[...],
    }


def test_tocategorical_predP_batB_seqLQ_generic(
        predP_batB_seqlQ_cls3_generic_cat):
    y, Y, nc = [
        predP_batB_seqlQ_cls3_generic_cat[k] for k in ['y', 'Y', 'nclasses']
    ]

    print(y.shape, Y.shape, nu.to_categorical(y, nc).shape)
    assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)

    if y.max() == nc:
        assert_almost_equal(nu.to_categorical(y), Y)

    assert True


## FIXTURES AND TESTS FOR CONFUSION MATRIX CALCULATIONS #######################


@pytest.fixture(
    scope='module',
    params=list(range(5)),
    ids=lambda i: "T={}".format(i),  #pylint: disable=unnecessary-lambda
)
def batB_seql1_cls3_trues_confmat(request, base_labels_cls3):
    i = request.param

    y_user = base_labels_cls3[i]
    nclasses = 3
    Y_user = nu.to_categorical(y_user, nclasses)

    # Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
    y_generic = y_user[:, np.newaxis]
    Y_generic = nu.to_categorical(y_generic, nclasses)

    return {
        'yt': y_user,
        'Yt': Y_user,
        'nclasses': nclasses,
        'ytg': y_generic,
        'Ytg': Y_generic
    }


@pytest.fixture(
    scope='module',
    params=list(range(5)),
    ids=lambda i: "P={}".format(i),  #pylint: disable=unnecessary-lambda
)  #pylint: disable=too-many-locals
def pred1_batB_seql1_cls3_preds_confmat(request, base_labels_cls3,
                                        batB_seql1_cls3_trues_confmat):
    i = request.param

    yp = base_labels_cls3[i]

    nclasses = batB_seql1_cls3_trues_confmat['nclasses']
    Yp = nu.to_categorical(yp, nclasses)

    yt = batB_seql1_cls3_trues_confmat['yt']
    Yt = batB_seql1_cls3_trues_confmat['Yt']
    confmat = ext_confusionmatrix(yt, yp, labels=np.arange(nclasses))

    confrecall = confmat / (confmat.sum(axis=1))[:, np.newaxis]
    confprecision = (confmat.T / (confmat.sum(axis=0))[:, np.newaxis]).T

    # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
    ytg = batB_seql1_cls3_trues_confmat['ytg']
    Ytg = batB_seql1_cls3_trues_confmat['Ytg']
    ypg = yp[np.newaxis, :, np.newaxis]
    Ypg = nu.to_categorical(ypg, nclasses)

    # Predictor=1, Batchsize=1(sumaxis), SequenceLength=1, ClassLabel=(nclasses, nclasses)(implicit)
    confmatg = confmat[np.newaxis, np.newaxis, np.newaxis, ...]
    confrecallg = confrecall[np.newaxis, np.newaxis, np.newaxis, ...]
    confprecisiong = confprecision[np.newaxis, np.newaxis, np.newaxis, ...]

    return {
        'yt': yt,
        'Yt': Yt,
        'yp': yp,
        'Yp': Yp,
        'confmat': confmat,
        'confrecall': confrecall,
        'confprecision': confprecision,
        'ytg': ytg,
        'Ytg': Ytg,
        'ypg': ypg,
        'Ypg': Ypg,
        'confmatg': confmatg,
        'confrecallg': confrecallg,
        'confprecisiong': confprecisiong,
    }


@pytest.mark.confmat
def test_pred1_batB_seql1_user_confmat(pred1_batB_seql1_cls3_preds_confmat):
    Yt, Yp, confmat = [
        pred1_batB_seql1_cls3_preds_confmat[k]
        for k in ['Yt', 'Yp', 'confmat']
    ]
    print(Yt.shape, Yp.shape)
    assert_almost_equal(nu.confusion_matrix_forcategorical(Yt, Yp), confmat)

    yt, yp = [pred1_batB_seql1_cls3_preds_confmat[k] for k in ['yt', 'yp']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(nu.confusion_matrix(yt, yp, nclasses), confmat)

    assert True


@pytest.mark.confmat
def test_pred1_batB_seql1_generic_confmat(pred1_batB_seql1_cls3_preds_confmat):
    Yt, Yp, confmat = [
        pred1_batB_seql1_cls3_preds_confmat[k]
        for k in ['Ytg', 'Ypg', 'confmatg']
    ]
    print("TEST", Yt.shape, Yp.shape)

    assert_almost_equal(
        nu.confusion_matrix_forcategorical(
            Yt, Yp, keepdims=True), confmat)

    yt, yp = [pred1_batB_seql1_cls3_preds_confmat[k] for k in ['ytg', 'ypg']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(
        nu.confusion_matrix(
            yt, yp, nclasses, keepdims=True), confmat)

    assert True


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # P: number of predictions
    ids=lambda i: "P={}".format(i),  #pylint: disable=unnecessary-lambda
)  #pylint: disable=too-many-locals
def predP_batB_seql1_cls3_preds_confmat(request, base_labels_cls3,
                                        batB_seql1_cls3_trues_confmat):
    i = request.param

    yp = [base_labels_cls3[ii] for ii in i]

    nclasses = batB_seql1_cls3_trues_confmat['nclasses']
    Yp = nu.to_categorical(yp, nclasses)

    yt = batB_seql1_cls3_trues_confmat['yt']
    Yt = batB_seql1_cls3_trues_confmat['Yt']

    confmat, confrecall, confprecision = [], [], []
    for ypp in yp:
        _confmat = ext_confusionmatrix(yt, ypp, labels=np.arange(nclasses))

        _confrecall = _confmat / (_confmat.sum(axis=1))[:, np.newaxis]
        _confprecision = (_confmat.T / (_confmat.sum(axis=0))[:, np.newaxis]).T

        confmat.append(_confmat)
        confrecall.append(_confrecall)
        confprecision.append(_confprecision)

    confmat = np.array(confmat)
    confrecall = np.array(confrecall)
    confprecision = np.array(confprecision)
    yp = np.array(yp)

    # Predictor=P, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
    ytg = batB_seql1_cls3_trues_confmat['ytg']
    Ytg = batB_seql1_cls3_trues_confmat['Ytg']
    ypg = yp[..., np.newaxis]
    Ypg = nu.to_categorical(ypg, nclasses)

    # Predictor=P, Batchsize=1(sumaxis), SequenceLength=1, ClassLabel=(nclasses, nclasses)(implicit)
    confmatg = confmat[:, np.newaxis, np.newaxis, ...]
    confrecallg = confrecall[:, np.newaxis, np.newaxis, ...]
    confprecisiong = confprecision[:, np.newaxis, np.newaxis, ...]

    return {
        'yt': yt,
        'Yt': Yt,
        'yp': yp,
        'Yp': Yp,
        'confmat': confmat,
        'confrecall': confrecall,
        'confprecision': confprecision,
        'ytg': ytg,
        'Ytg': Ytg,
        'ypg': ypg,
        'Ypg': Ypg,
        'confmatg': confmatg,
        'confrecallg': confrecallg,
        'confprecisiong': confprecisiong,
    }


@pytest.mark.confmat
def test_predP_batB_seql1_user_confmat(predP_batB_seql1_cls3_preds_confmat):
    Yt, Yp, confmat = [
        predP_batB_seql1_cls3_preds_confmat[k]
        for k in ['Yt', 'Yp', 'confmat']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(nu.confusion_matrix_forcategorical(Yt, Yp), confmat)

    yt, yp = [predP_batB_seql1_cls3_preds_confmat[k] for k in ['yt', 'yp']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(nu.confusion_matrix(yt, yp, nclasses), confmat)

    assert True


@pytest.mark.confmat
def test_predP_batB_seql1_generic_confmat(predP_batB_seql1_cls3_preds_confmat):
    Yt, Yp, confmat = [
        predP_batB_seql1_cls3_preds_confmat[k]
        for k in ['Ytg', 'Ypg', 'confmatg']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(
        nu.confusion_matrix_forcategorical(
            Yt, Yp, keepdims=True), confmat)

    yt, yp = [predP_batB_seql1_cls3_preds_confmat[k] for k in ['ytg', 'ypg']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(
        nu.confusion_matrix(
            yt, yp, nclasses, keepdims=True), confmat)

    assert True


@pytest.fixture(
    scope='module',
    params=[[b + q for q in range(2)]
            for b in range(3)],  # B: batchsize, Q: SequenceLength
    ids=lambda i: "T={}".format(i),  #pylint: disable=unnecessary-lambda
)
def batB_seqlQ_cls3_trues_confmat(request, base_labels_cls3):
    i = request.param

    y_user = [base_labels_cls3[ii] for ii in i]
    nclasses = 3
    y_user = np.array(y_user)
    Y_user = nu.to_categorical(y_user, nclasses)

    # Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
    y_generic = y_user
    Y_generic = nu.to_categorical(y_generic, nclasses)

    return {
        'yt': y_user,
        'Yt': Y_user,
        'nclasses': nclasses,
        'ytg': y_generic,
        'Ytg': Y_generic
    }


@pytest.fixture(
    scope='module',
    params=[list(range(2))],  # B: batchsize, Q: SequenceLength
    ids=lambda i: "P={}".format(i),  #pylint: disable=unnecessary-lambda
)  #pylint: disable=too-many-locals
def pred1_batB_seqlQ_cls3_preds_confmat(request, base_labels_cls3,
                                        batB_seqlQ_cls3_trues_confmat):
    i = request.param

    yp = [base_labels_cls3[ii] for ii in i]

    nclasses = batB_seqlQ_cls3_trues_confmat['nclasses']
    Yp = nu.to_categorical(yp, nclasses)

    yt = batB_seqlQ_cls3_trues_confmat['yt']
    Yt = batB_seqlQ_cls3_trues_confmat['Yt']

    confmat, confrecall, confprecision = [], [], []
    for b, ybb in enumerate(yp):
        _confmat = ext_confusionmatrix(
            yt[b, ...], ybb, labels=np.arange(nclasses))

        _confrecall = _confmat / (_confmat.sum(axis=1))[:, np.newaxis]
        _confprecision = (_confmat.T / (_confmat.sum(axis=0))[:, np.newaxis]).T

        confmat.append(_confmat)
        confrecall.append(_confrecall)
        confprecision.append(_confprecision)

    confmat = np.array(confmat)
    confrecall = np.array(confrecall)
    confprecision = np.array(confprecision)
    yp = np.array(yp)

    # Predictor=1, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
    ytg = batB_seqlQ_cls3_trues_confmat['ytg']
    Ytg = batB_seqlQ_cls3_trues_confmat['Ytg']
    ypg = yp[np.newaxis, ...]
    Ypg = nu.to_categorical(ypg, nclasses)

    # Predictor=1, Batchsize=B, SequenceLength=1(sumaxis), ClassLabel=(nclasses, nclasses)(implicit)
    confmatg = confmat[np.newaxis, :, np.newaxis, ...]
    confrecallg = confrecall[np.newaxis, :, np.newaxis, ...]
    confprecisiong = confprecision[np.newaxis, :, np.newaxis, ...]

    return {
        'yt': yt,
        'Yt': Yt,
        'yp': yp,
        'Yp': Yp,
        'confmat': confmat,
        'confrecall': confrecall,
        'confprecision': confprecision,
        'ytg': ytg,
        'Ytg': Ytg,
        'ypg': ypg,
        'Ypg': Ypg,
        'confmatg': confmatg,
        'confrecallg': confrecallg,
        'confprecisiong': confprecisiong,
    }


@pytest.mark.confmat
def test_pred1_batB_seqlQ_user_confmat(pred1_batB_seqlQ_cls3_preds_confmat):
    Yt, Yp, confmat = [
        pred1_batB_seqlQ_cls3_preds_confmat[k]
        for k in ['Yt', 'Yp', 'confmat']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(nu.confusion_matrix_forcategorical(Yt, Yp), confmat)

    yt, yp = [pred1_batB_seqlQ_cls3_preds_confmat[k] for k in ['yt', 'yp']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(nu.confusion_matrix(yt, yp, nclasses), confmat)

    assert True


@pytest.mark.confmat
def test_pred1_batB_seqlQ_generic_confmat(pred1_batB_seqlQ_cls3_preds_confmat):
    Yt, Yp, confmat = [
        pred1_batB_seqlQ_cls3_preds_confmat[k]
        for k in ['Ytg', 'Ypg', 'confmatg']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(
        nu.confusion_matrix_forcategorical(
            Yt, Yp, keepdims=True), confmat)

    yt, yp = [pred1_batB_seqlQ_cls3_preds_confmat[k] for k in ['ytg', 'ypg']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(
        nu.confusion_matrix(
            yt, yp, nclasses, keepdims=True), confmat)

    assert True


@pytest.fixture(
    scope='module',
    params=[[list(range(4)), list(range(2))]],  # P: Predictors, B: batchsize,
    ids=lambda i: "P={}".format(i),  #pylint: disable=unnecessary-lambda
)  #pylint: disable=too-many-locals
def predP_batB_seqlQ_cls3_preds_confmat(request, base_labels_cls3,
                                        batB_seqlQ_cls3_trues_confmat):
    p, b = request.param

    yp = []
    for pp in p:
        ypp = []
        for bb in b:
            ypp.append(base_labels_cls3[bb + pp])
        yp.append(ypp)

    nclasses = batB_seqlQ_cls3_trues_confmat['nclasses']
    Yp = nu.to_categorical(yp, nclasses)

    yt = batB_seqlQ_cls3_trues_confmat['yt']
    Yt = batB_seqlQ_cls3_trues_confmat['Yt']

    confmat, confrecall, confprecision = [], [], []
    for pp, ypp in enumerate(yp):
        __confmat, __confrecall, __confprecision = [], [], []
        for bb, ybb in enumerate(ypp):
            _confmat = ext_confusionmatrix(
                yt[bb, ...], ybb, labels=np.arange(nclasses))

            _confrecall = _confmat / (_confmat.sum(axis=1))[:, np.newaxis]
            _confprecision = (_confmat.T /
                              (_confmat.sum(axis=0))[:, np.newaxis]).T

            __confmat.append(_confmat)
            __confrecall.append(_confrecall)
            __confprecision.append(_confprecision)

        confmat.append(__confmat)
        confrecall.append(__confrecall)
        confprecision.append(__confprecision)

    confmat = np.array(confmat)
    confrecall = np.array(confrecall)
    confprecision = np.array(confprecision)
    yp = np.array(yp)

    # Predictor=P, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
    ytg = batB_seqlQ_cls3_trues_confmat['ytg']
    Ytg = batB_seqlQ_cls3_trues_confmat['Ytg']
    ypg = yp
    Ypg = nu.to_categorical(ypg, nclasses)

    # Predictor=P, Batchsize=B, SequenceLength=1(sumaxis), ClassLabel=(nclasses, nclasses)(implicit)
    confmatg = confmat[..., np.newaxis, :, :]
    confrecallg = confrecall[..., np.newaxis, :, :]
    confprecisiong = confprecision[..., np.newaxis, :, :]

    return {
        'yt': yt,
        'Yt': Yt,
        'yp': yp,
        'Yp': Yp,
        'confmat': confmat,
        'confrecall': confrecall,
        'confprecision': confprecision,
        'ytg': ytg,
        'Ytg': Ytg,
        'ypg': ypg,
        'Ypg': Ypg,
        'confmatg': confmatg,
        'confrecallg': confrecallg,
        'confprecisiong': confprecisiong,
    }


@pytest.mark.confmat
def test_predP_batB_seqlQ_user_confmat(predP_batB_seqlQ_cls3_preds_confmat):
    Yt, Yp, confmat = [
        predP_batB_seqlQ_cls3_preds_confmat[k]
        for k in ['Yt', 'Yp', 'confmat']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(nu.confusion_matrix_forcategorical(Yt, Yp), confmat)

    yt, yp = [predP_batB_seqlQ_cls3_preds_confmat[k] for k in ['yt', 'yp']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(nu.confusion_matrix(yt, yp, nclasses), confmat)

    assert True


@pytest.mark.confmat
def test_predP_batB_seqlQ_generic_confmat(predP_batB_seqlQ_cls3_preds_confmat):
    Yt, Yp, confmat = [
        predP_batB_seqlQ_cls3_preds_confmat[k]
        for k in ['Ytg', 'Ypg', 'confmatg']
    ]
    print("\nTEST", Yt.shape, Yp.shape, confmat.shape)
    print()
    assert_almost_equal(
        nu.confusion_matrix_forcategorical(
            Yt, Yp, keepdims=True), confmat)

    yt, yp = [predP_batB_seqlQ_cls3_preds_confmat[k] for k in ['ytg', 'ypg']]
    nclasses = Yt.shape[-1]
    print(yt.shape, yp.shape)
    assert_almost_equal(
        nu.confusion_matrix(
            yt, yp, nclasses, keepdims=True), confmat)

    assert True


# TODO: test for different confusion matrix reduction axis

## TESTS FOR NORMALIZING CONFUSION MATRICES ################################


@pytest.mark.normconf
def test_pred1_batB_seql1_normconfmat(pred1_batB_seql1_cls3_preds_confmat):
    confmat, confrecall, confprecision = [
        pred1_batB_seql1_cls3_preds_confmat[k]
        for k in ['confmat', 'confrecall', 'confprecision']
    ]

    print(confmat.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmat)
    assert_almost_equal(confprecp, confprecision)
    assert_almost_equal(confrecp, confrecall)

    confmatg, confrecallg, confprecisiong = [
        pred1_batB_seql1_cls3_preds_confmat[k]
        for k in ['confmatg', 'confrecallg', 'confprecisiong']
    ]

    print("G:", confmatg.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmatg)
    assert_almost_equal(confprecp, confprecisiong)
    assert_almost_equal(confrecp, confrecallg)


@pytest.mark.normconf
def test_predP_batB_seql1_normconfmat(predP_batB_seql1_cls3_preds_confmat):
    confmat, confrecall, confprecision = [
        predP_batB_seql1_cls3_preds_confmat[k]
        for k in ['confmat', 'confrecall', 'confprecision']
    ]

    print(confmat.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmat)
    assert_almost_equal(confprecp, confprecision)
    assert_almost_equal(confrecp, confrecall)

    confmatg, confrecallg, confprecisiong = [
        predP_batB_seql1_cls3_preds_confmat[k]
        for k in ['confmatg', 'confrecallg', 'confprecisiong']
    ]

    print("G:", confmatg.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmatg)
    assert_almost_equal(confprecp, confprecisiong)
    assert_almost_equal(confrecp, confrecallg)


@pytest.mark.normconf
def test_pred1_batB_seqlQ_normconfmat(pred1_batB_seqlQ_cls3_preds_confmat):
    provider = pred1_batB_seqlQ_cls3_preds_confmat
    confmat, confrecall, confprecision = [
        provider[k] for k in ['confmat', 'confrecall', 'confprecision']
    ]

    print(confmat.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmat)
    assert_almost_equal(confprecp, confprecision)
    assert_almost_equal(confrecp, confrecall)

    confmatg, confrecallg, confprecisiong = [
        provider[k] for k in ['confmatg', 'confrecallg', 'confprecisiong']
    ]

    print("G:", confmatg.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmatg)
    assert_almost_equal(confprecp, confprecisiong)
    assert_almost_equal(confrecp, confrecallg)


@pytest.mark.normconf
def test_predP_batB_seqlQ_normconfmat(predP_batB_seqlQ_cls3_preds_confmat):
    provider = predP_batB_seqlQ_cls3_preds_confmat
    confmat, confrecall, confprecision = [
        provider[k] for k in ['confmat', 'confrecall', 'confprecision']
    ]

    print(confmat.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmat)
    assert_almost_equal(confprecp, confprecision)
    assert_almost_equal(confrecp, confrecall)

    confmatg, confrecallg, confprecisiong = [
        provider[k] for k in ['confmatg', 'confrecallg', 'confprecisiong']
    ]

    print("G:", confmatg.shape)

    confprecp, confrecp = nu.normalize_confusion_matrix(confmatg)
    assert_almost_equal(confprecp, confprecisiong)
    assert_almost_equal(confrecp, confrecallg)


@pytest.mark.confmat
def test_pred1_batB_seql1_confmat_printing(
        pred1_batB_seql1_cls3_preds_confmat):
    confrecall, confprecision = [
        pred1_batB_seql1_cls3_preds_confmat[k]
        for k in ['confrecall', 'confprecision']
    ]

    nu.print_normalized_confusion(confrecall)
    nu.print_normalized_confusion(confprecision)

    assert True


## TESTS FOR SHARE DATA #######################################################


@pytest.fixture(
    scope='module', )
def base_2d_array(sourcefn=base_labels_cls3):
    """ Base 2D array fixture for numpy tricks tests

    Copying the `base_labels_cls3` cuz lazy
    """
    return sourcefn()


@pytest.fixture(
    scope="module",
    params=list(range(len(base_2d_array()))),
    ids=lambda x: "1D-[{}]".format(x),  #pylint: disable=unnecessary-lambda
)
def derived_1d_array(request, base_2d_array):
    return base_2d_array[request.param, ...]


@pytest.mark.shared_data
def test_1d_does_share_data_with_base_2d(base_2d_array, derived_1d_array):
    assert nu.arrays_do_share_data(base_2d_array, derived_1d_array)
    # striding creates a DummyArray with the same data
    # pure call to the as_strided function does just that, without any striding
    assert nu.arrays_do_share_data(
        np.lib.stride_tricks.as_strided(derived_1d_array), base_2d_array)


@pytest.mark.shared_data
def test_1dcopy_does_not_share_data_with_base_2d(base_2d_array,
                                                 derived_1d_array):
    assert not nu.arrays_do_share_data(base_2d_array, derived_1d_array.copy())


@pytest.fixture(
    scope="module",
    params=list(range(len(base_2d_array()))),
    ids=lambda x: "2D-[{}]".format(x),  #pylint: disable=unnecessary-lambda
)
def derived_2d_array_from_0(request, base_2d_array):
    """ this includes base_2d_array[0:0, ...] """
    return base_2d_array[0:request.param, ...]


@pytest.mark.shared_data
def test_2d_does_share_data_with_base_2d(base_2d_array,
                                         derived_2d_array_from_0):
    """ This should pass even for 0-length derived arrays """
    assert nu.arrays_do_share_data(base_2d_array, derived_2d_array_from_0)
    # striding creates a DummyArray with the same data
    # pure call to the as_strided function does just that, without any striding
    assert nu.arrays_do_share_data(
        np.lib.stride_tricks.as_strided(derived_2d_array_from_0),
        base_2d_array)


@pytest.mark.shared_data
def test_2dcopy_doesnot_share_data_with_base_2d(base_2d_array,
                                                derived_2d_array_from_0):
    """ This should pass even for 0-length derived arrays """
    assert not nu.arrays_do_share_data(base_2d_array,
                                       derived_2d_array_from_0.copy())


# NOTE: Not testing for any other possible sub-arrays. Hope it is fine
