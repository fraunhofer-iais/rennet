"""
@motjuste
Created: 08-02-2017

Test the Numpy Utilities
"""
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


@pytest.fixture(
    scope='module',
    params=[*list(range(len(base_labels_cls3())))],
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
    Y = ext_tocategorical(y, nb_classes=nclasses)

    return {
        'y': y,
        'Y': Y,
        'nclasses': nclasses,
    }


@pytest.mark.user
def test_tocategorical_trues_user(pred1_batB_seqL1_cls3_trues_preds_user_cat):
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
    params=[*list(range(len(base_labels_cls3())))],
    ids=lambda i: "TG-(B, 1, 3)-{}".format(i)  #pylint: disable=unnecessary-lambda
)
def batB_seqL1_cls3_trues_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format expected by the generic function
    """
    i = request.param
    y = base_labels_cls3[i]
    nclasses = max(y) + 1
    Y = ext_tocategorical(y, nb_classes=nclasses)

    return {
        'nclasses': nclasses,

        # Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': y[:, np.newaxis],

        # Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': Y[:, np.newaxis, :],
    }


@pytest.mark.generic
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
    params=[*list(range(len(base_labels_cls3())))],
    ids=lambda i: "PG-(B, 1, 3)-{}".format(i)  #pylint: disable=unnecessary-lambda
)
def pred1_batB_seqL1_cls3_generic_cat(request, base_labels_cls3):
    """ y and Y (categorical) in format expected by the generic function
    """
    i = request.param
    y = base_labels_cls3[i]
    nclasses = max(y) + 1
    Y = ext_tocategorical(y, nb_classes=nclasses)

    return {
        'nclasses': nclasses,

        # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': y[np.newaxis, :, np.newaxis],

        # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': Y[np.newaxis, :, np.newaxis, :],
    }


@pytest.mark.generic
def test_tocategorical_preds1_generic(pred1_batB_seqL1_cls3_generic_cat):
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
    trues and preds look the same from the user's perspective when
    there are P predictions
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, nb_classes=nclasses) for yy in y]

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


@pytest.mark.user
def test_tocategorical_predsP_user(predP_batB_seqL1_cls3_user_cat):
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
    Y = [ext_tocategorical(yy, nb_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Predictor=P, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
        'y': np.array(y)[..., np.newaxis],

        # Predictor=P, Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[..., np.newaxis, :],
    }


@pytest.mark.generic
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
    Y = [ext_tocategorical(yy, nb_classes=nclasses) for yy in y]

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


@pytest.mark.user
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
    Y = [ext_tocategorical(yy, nb_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y),

        # Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y),
    }


@pytest.mark.generic
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
    this is exactly like predP_batB_seqL1_cls3_user_cat
    but these are trues
    non-zero sequence length trues and preds look similar for single predictor
    """
    i = request.param
    y = [base_labels_cls3[ii] for ii in i]
    nclasses = max([max(yy) for yy in y]) + 1
    Y = [ext_tocategorical(yy, nb_classes=nclasses) for yy in y]

    return {
        'nclasses': nclasses,

        # Predictor=1, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y)[np.newaxis, ...],

        # Predictor=1, Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[np.newaxis, ...],
    }


@pytest.mark.generic
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
            YY.append(ext_tocategorical(yb, nb_classes=nclasses))
        Y.append(YY)

    return {
        'y': np.array(y),
        'Y': np.array(Y),
        'nclasses': nclasses,
    }


@pytest.mark.user
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
            YY.append(ext_tocategorical(yb, nb_classes=nclasses))
        Y.append(YY)

    return {
        'nclasses': nclasses,

        # Predictor=P, Batchsize=B, SequenceLength=Q, ClassLabel=1(implicit)
        'y': np.array(y)[...],

        # Predictor=P, Batchsize=B, SequenceLength=Q, ClassLabel=nclasses(categorical)
        'Y': np.array(Y)[...],
    }


@pytest.mark.generic
def test_tocategorical_predP_batB_seqLQ_generic(
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


# @pytest.mark.user
# def test_tocategorical_user_preds(pred1_batszB_seql1_cls3_user):
#     y, Y, nc = [
#         pred1_batszB_seql1_cls3_user[k]
#         for k in ['ypred', 'Ypred', 'nclasses']
#     ]
#
#     if Y is None:
#         pytest.skip('Erroneous result expected')
#
#     assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)
#     if Y.shape[-1] != nc:
#         assert_almost_equal(nu.to_categorical(y), Y)
#
#     assert True
#
#
# @pytest.mark.generic
# def test_tocategorical_generic_trues(pred1_batszB_seql1_cls3_generic):
#     y, Y, nc = [
#         pred1_batszB_seql1_cls3_generic[k]
#         for k in ['ytrue', 'Ytrue', 'nclasses']
#     ]
#
#     if Y is None:
#         pytest.skip('Erroneous result expected')
#
#     assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)
#     if Y.shape[-1] != nc:
#         assert_almost_equal(nu.to_categorical(y), Y)
#
#     assert True
#
#
# @pytest.mark.generic
# def test_tocategorical_generic_preds(pred1_batszB_seql1_cls3_generic):
#     y, Y, nc = [
#         pred1_batszB_seql1_cls3_generic[k]
#         for k in ['ypred', 'Ypred', 'nclasses']
#     ]
#
#     if Y is None:
#         pytest.skip('Erroneous result expected')
#
#     assert_almost_equal(nu.to_categorical(y, nclasses=nc), Y)
#     if Y.shape[-1] != nc:
#         assert_almost_equal(nu.to_categorical(y), Y)
#
#     assert True

# @pytest.fixture(
#     scope='module',
#     params=[(t, p)
#             for t in range(len(base_labels_cls3()))
#             for p in range(len(base_labels_cls3()))],
#     ids=lambda tp: "T={}-P={}".format(*tp))
# def pred1_batszB_seql1_cls3_user(request, base_labels_cls3):
#     """ inputs and outputs in format expected from / by user """
#     t, p = request.param
#     ytrue = base_labels_cls3[t]
#     ypred = base_labels_cls3[p]
#
#     nclasses = max(y) + 1  # Change here accordingly based on base_labels_cls3
#
#     if any(i == len(base_labels_cls3) - 1
#            for i in (t, p)):  # Change here if you know of failures
#         Ytrue = None
#         Ypred = None
#         confusion = None
#         confrecall = None
#         confprecision = None
#     else:
#         Ytrue = ext_tocategorical(ytrue, nb_classes=nclasses)
#         Ypred = ext_tocategorical(ypred, nb_classes=nclasses)
#
#         confusion = ext_confusionmatrix(
#             ytrue, ypred, labels=np.arange(nclasses))
#         confrecall = confusion / (confusion.sum(axis=1))[:, np.newaxis]
#         confprecision = (confusion.T / confusion.sum(axis=0)[:, np.newaxis]).T
#
#     return {
#         "ytrue": ytrue,
#         "ypred": ypred,
#         "nclasses": nclasses,
#         "Ytrue": Ytrue,
#         "Ypred": Ypred,
#         "confusion": confusion,
#         "confrecall": confrecall,
#         "confprecision": confprecision
#     }

# @pytest.fixture(scope='module')
# def pred1_batszB_seql1_cls3_generic(pred1_batszB_seql1_cls3_user):
#     """ i/o expected formatted based on what the generic functions expect
#     Generic Shape:
#     (Predictor, Batchsize, SequenceLength, ClassLabel)
#     """
#     d = pred1_batszB_seql1_cls3_user
#     return {
#         # Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
#         "ytrue": d['ytrue'][:, np.newaxis],  # np.newaxis],
#
#         # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=1(implicit)
#         "ypred": d['ypred'][np.newaxis, :, np.newaxis],  #np.newaxis],
#
#         # nclasses is integer
#         "nclasses": d['nclasses'],
#
#         # Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
#         "Ytrue": None
#         if d['Ytrue'] is None else d['Ytrue'][np.newaxis, :, np.newaxis, :],
#
#         # Predictor=1, Batchsize=B, SequenceLength=1, ClassLabel=nclasses(categorical)
#         "Ypred": None if d['Ypred'] is None else
#         d['Ypred'][np.newaxis, np.newaxis, :, np.newaxis, :],
#
#         # Predictor=1, Batchsize=1(sumaxis), SequenceLength=1, ClassLabel=(nclasses, nclasses)
#         "confusion": None if d['confusion'] is None else
#         d['confusion'][np.newaxis, np.newaxis, np.newaxis, ...],
#
#         # summary axis : Batchsize
#         "sumaxis": 1,
#
#         # Predictor=1, Batchsize=1(sumaxis), SequenceLength=1, ClassLabel=(nclasses, nclasses)
#         "confrecall": None if d['confrecall'] is None else
#         d['confrecall'][np.newaxis, np.newaxis, np.newaxis, ...],
#
#         # Predictor=1, Batchsize=1(sumaxis), SequenceLength=1, ClassLabel=(nclasses, nclasses)
#         "confprecision": None if d['confprecision'] is None else
#         d['confprecision'][np.newaxis, np.newaxis, np.newaxis, ...]
#     }

## TO CATEGORICAL TESTS #######################################################

## CONFUSION MATRIX TESTS #####################################################

# @pytest.fixture(scope='module', params=list(range(len(confusion))))
# def normal_preds_confusion(request):
#     return {
#         "labels": labels,
#         "predictions": predictions[request.param, ...],
#         "confusion": confusion[request.param, ...],
#     }
#
#
# def test_normal_confusion_matrix(normal_preds_confusion):
#     labels = normal_preds_confusion['labels']
#     preds = normal_preds_confusion['predictions']
#     true_confusion = normal_preds_confusion['confusion']
#
#     assert_almost_equal(true_confusion, nu.confusion_matrix(labels, preds))
#
#
# @pytest.fixture(scope='module', params=[np.arange(len(confusion))])
# def normal_multi_preds_confusion(request):
#     return {
#         "labels": labels,
#         "predictions": predictions[request.param, ...],
#         "confusion": confusion[request.param, ...],
#     }
#
#
# def test_normal_multi_confusion_matrix(normal_multi_preds_confusion):
#     labels = normal_multi_preds_confusion['labels']
#     preds = normal_multi_preds_confusion['predictions']
#     true_confusion = normal_multi_preds_confusion['confusion']
#
#     assert_almost_equal(true_confusion, nu.confusion_matrices(labels, preds))
#
#
# @pytest.fixture(scope='module', params=[6, ])
# def extra_class_preds_confusion(request):
#     return {
#         "labels": labels,
#         "predictions": predictions[request.param, ...],
#     }
#
#
# def test_extra_pred_label_raises(extra_class_preds_confusion):
#     labels = extra_class_preds_confusion['labels']
#     preds = extra_class_preds_confusion['predictions']
#
#     # NOTE: The Exception is raised while converting the preds to categorical
#     with pytest.raises(RuntimeError):
#         nu.confusion_matrix(labels, preds)
#
#
# def test_missing_label_warns(extra_class_preds_confusion):
#     preds = extra_class_preds_confusion['labels']  # has extra class
#     labels = extra_class_preds_confusion['predictions']
#
#     # NOTE: The Exception is raised while converting the preds to categorical
#     with pytest.raises(RuntimeWarning):
#         nu.confusion_matrix(labels, preds, warn=True)
#
#     # Should not raise warning
#     nu.confusion_matrix(labels, preds)
#
#
# ## CONF-PRECISION AND CONF-RECALL TESTS #######################################
#
#
# @pytest.fixture(scope='module', params=[0, 1, 2, 5])
# def normal_preds_conf_prec_rec(request):
#     return {
#         "confusion": confusion[request.param, ...],
#         "confprec": confprec[request.param, ...],
#         "confrecall": confrecall[request.param, ...],
#     }
#
#
# def test_individual_normal_preds_conf_prec_rec(normal_preds_conf_prec_rec):
#     confusion = normal_preds_conf_prec_rec['confusion']
#     true_confprec = normal_preds_conf_prec_rec['confprec']
#     true_confrecall = normal_preds_conf_prec_rec['confrecall']
#
#     confprec, confrecall = nu.normalize_confusion_matrix(confusion)
#     assert_almost_equal(true_confprec, confprec, decimal=2)
#     assert_almost_equal(true_confrecall, confrecall, decimal=2)
#
#
# @pytest.fixture(scope='module', params=[3, 4])
# def nan_preds_conf_prec_rec(request):
#     return {
#         "confusion": confusion[request.param, ...],
#         "confprec": confprec[request.param, ...],
#         "confrecall": confrecall[request.param, ...],
#     }
#
#
# def test_individual_nan_preds_conf_prec_rec(nan_preds_conf_prec_rec):
#     confusion = nan_preds_conf_prec_rec['confusion']
#     true_confprec = nan_preds_conf_prec_rec['confprec']
#     true_confrecall = nan_preds_conf_prec_rec['confrecall']
#
#     confprec, confrecall = nu.normalize_confusion_matrix(confusion)
#     assert_almost_equal(true_confprec, confprec, decimal=2)
#     assert_almost_equal(true_confrecall, confrecall, decimal=2)
#
#
# @pytest.fixture(
#     scope='module',
#     params=[np.array([0, 1, 2, 5]), np.arange(len(confusion))])
# def normal_multi_preds_conf_prec_rec(request):
#     return {
#         "confusion": confusion[request.param, ...],
#         "confprec": confprec[request.param, ...],
#         "confrecall": confrecall[request.param, ...],
#     }
#
#
# def test_multi_normal_preds_conf_prec_rec(normal_multi_preds_conf_prec_rec):
#     confusion = normal_multi_preds_conf_prec_rec['confusion']
#     true_confprec = normal_multi_preds_conf_prec_rec['confprec']
#     true_confrecall = normal_multi_preds_conf_prec_rec['confrecall']
#
#     confprec, confrecall = nu.normalize_confusion_matrices(confusion)
#     assert_almost_equal(true_confprec, confprec, decimal=2)
#     assert_almost_equal(true_confrecall, confrecall, decimal=2)
