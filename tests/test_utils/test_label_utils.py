"""
@motjuste
Created: 26-08-2016

Test the label utilities module
"""
from __future__ import print_function, division
import pytest
import numpy as np
import numpy.testing as npt
# from collections import namedtuple

from rennet.utils import label_utils as lu

# pylint: disable=redefined-outer-name


@pytest.fixture(scope='module')
def base_contiguous_small_seqdata():
    starts_secs = np.array([0., 1., 3., 4.5])
    ends_secs = np.array([1., 3., 4.5, 6.])
    labels = [1, 0, 2, 1]
    samplerate = 1.

    labels_at_secs = [
        (0.5, 1),
        (1., 1),
        (1.4, 0),
        (3.8, 2),
        (5.5, 1),
        (5.9999375, 1),
        (6, 1),
    ]

    iscontiguous = True

    return starts_secs, ends_secs, labels, samplerate, labels_at_secs, iscontiguous


@pytest.fixture(scope='module')
def base_noncontiguous_small_seqdata():
    starts_secs = np.array([0., 1., 3., 4.5])
    ends_secs = np.array([1., 4.8, 5., 6.])
    labels = [1, 0, 2, 1]
    samplerate = 1.

    labels_at_secs_nonconti = [
        (0.5, (1, )),
        (1., (1, )),
        (3.4, (
            0,
            2, )),
        (4.8, (
            0,
            2,
            1, )),
        (5.5, (1, )),
        (6, (1, )),
    ]

    iscontiguous = False

    return starts_secs, ends_secs, labels, samplerate, labels_at_secs_nonconti, iscontiguous


@pytest.fixture(
    scope='module',
    params=[
        base_contiguous_small_seqdata(),
        base_noncontiguous_small_seqdata(),
    ],
    ids=['contiguous', 'non-contiguous'])
def base_small_seqdata(request):
    starts_secs, ends_secs, labels, samplerate, la, isconti = request.param
    se_secs = np.vstack([starts_secs, ends_secs]).T
    return {
        'starts_ends': se_secs,
        'samplerate': samplerate,
        'labels': labels,
        'isconti': isconti,
        'labels_at': la,
    }


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 8000, 16000],  # samplerate
    ids=lambda x: "SR={}".format(x))  # pylint: disable=unnecessary-lambda
def init_small_seqdata(request, base_small_seqdata):
    sr = request.param
    se = (base_small_seqdata['starts_ends'] *
          (sr / base_small_seqdata['samplerate']))

    return {
        'starts_ends': se,
        'samplerate': sr,
        'labels': base_small_seqdata['labels'],
        'isconti': base_small_seqdata['isconti'],
        'labels_at': base_small_seqdata['labels_at'],
    }


def test_SequenceLabels_initializes(init_small_seqdata):
    """ Test SequenceLabels class initializes w/o errors """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.SequenceLabels(se, l, samplerate=sr)

    npt.assert_equal(s.starts_ends, se)
    assert s.samplerate == sr
    assert s.orig_samplerate == sr
    assert all([e == r for e, r in zip(l, s.labels)]), list(zip(l, s.labels))


def test_ContiguousSequenceLabels_init_conti_fail_nonconti(init_small_seqdata):
    """ Test ContiguousSequenceLabels class initializes
        - w/o errors if labels are contiguous
        - raises AssertionError when non-contiguous
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']
    if init_small_seqdata['isconti']:
        s = lu.ContiguousSequenceLabels(se, l, samplerate=sr)

        npt.assert_equal(s.starts_ends, se)
        assert s.samplerate == sr
        assert s.orig_samplerate == sr
        assert all(
            [e == r for e, r in zip(l, s.labels)]), list(zip(l, s.labels))
    else:
        with pytest.raises(AssertionError):
            lu.ContiguousSequenceLabels(se, l, samplerate=sr)


@pytest.fixture(
    scope='module',
    params=[lu.SequenceLabels, lu.ContiguousSequenceLabels],
    ids=lambda x: x.__name__)
def seqlabelinst_small_seqdata(request, init_small_seqdata):
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    if (not init_small_seqdata['isconti'] and
            request.param is lu.ContiguousSequenceLabels):
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")
    else:
        s = request.param(se, l, samplerate=sr)

    return {
        'seqlabelinst': s,
        'orig_sr': sr,
        'orig_se': se,
    }


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 100, 101],  # to_samplerate
    ids=lambda x: "toSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def se_to_sr_small_seqdata_SequenceLabels(request, seqlabelinst_small_seqdata):
    """ Fixture to test starts_ends are calculated correctly in contextual sr """
    s = seqlabelinst_small_seqdata['seqlabelinst']
    sr = seqlabelinst_small_seqdata['orig_sr']
    se = seqlabelinst_small_seqdata['orig_se']

    to_sr = request.param
    target_se = se * (to_sr / sr)

    return {
        'seqlabelinst': s,
        'orig_sr': sr,
        'target_sr': to_sr,
        'target_se': target_se
    }


def test_se_to_sr_SeqLabels(se_to_sr_small_seqdata_SequenceLabels):
    s, osr, tsr, tse = [
        se_to_sr_small_seqdata_SequenceLabels[k]
        for k in ['seqlabelinst', 'orig_sr', 'target_sr', 'target_se']
    ]

    with s.samplerate_as(tsr):
        rse = s.starts_ends
        rsr = s.samplerate
        rosr = s.orig_samplerate

    assert rosr == osr
    assert rsr == tsr
    npt.assert_equal(rse, tse)


@pytest.fixture(
    scope='module',
    params=[[2, 3, 101]],  # to_samplerate cycles
    ids=lambda x: "toSRcycle={}".format(x)  #pylint: disable=unnecessary-lambda
)
def threelevel_sr_as_SeqLabels(request, seqlabelinst_small_seqdata):
    to_sr_cycle = request.param
    return {
        'seqlabelinst': seqlabelinst_small_seqdata['seqlabelinst'],
        'orig_sr': seqlabelinst_small_seqdata['orig_sr'],
        'to_sr_cycle': to_sr_cycle
    }


def test_threelevel_to_sr_cycle(threelevel_sr_as_SeqLabels):
    s, orig_sr, to_sr_cycle = [
        threelevel_sr_as_SeqLabels[k]
        for k in ['seqlabelinst', 'orig_sr', 'to_sr_cycle']
    ]

    assert s.orig_samplerate == orig_sr
    assert s.samplerate == orig_sr

    with s.samplerate_as(to_sr_cycle[0]):
        assert s.orig_samplerate == orig_sr
        assert s.samplerate == to_sr_cycle[0]

        with s.samplerate_as(to_sr_cycle[1]):
            assert s.orig_samplerate == orig_sr
            assert s.samplerate == to_sr_cycle[1]

            with s.samplerate_as(to_sr_cycle[2]):
                assert s.orig_samplerate == orig_sr
                assert s.samplerate == to_sr_cycle[2]

            # sr is reset when pulled out of context 2
            assert s.orig_samplerate == orig_sr
            assert s.samplerate == to_sr_cycle[1]

        # sr is reset when pulled out of context 1
        assert s.orig_samplerate == orig_sr
        assert s.samplerate == to_sr_cycle[0]

    # sr is reset when pulled out of context 0
    assert s.orig_samplerate == orig_sr
    assert s.samplerate == orig_sr


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def SequenceLabels_small_seqdata_labels_at_allwithin(request,
                                                     init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.SequenceLabels(se, l, samplerate=sr)

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append(l)
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append((l, ))

    la_sr = request.param
    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
    }


def test_SequenceLabels_labels_at_allwithin(
        SequenceLabels_small_seqdata_labels_at_allwithin):
    s, la_ends, lasr, target_labels = [
        SequenceLabels_small_seqdata_labels_at_allwithin[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, samplerate=lasr)

    assert all([e == r for e, r in zip(target_labels, labels)]), list(
        zip(target_labels, labels))


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def SequenceLabels_small_seqdata_labels_at_outside(request,
                                                   init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.SequenceLabels(se, l, samplerate=sr)

    sminstart = s.starts.min()
    smaxend = s.ends.max()
    la_ends = [sminstart - (1 / sr), sminstart, smaxend + (1 / sr)]
    # Yes, there is no label for sminstart. So the default_label is expected
    # Why? We are looking at the label for the segment between
    # (x - (1/samplerate)) and (x) when finding labels_at
    # and we don't have any info about the label before sminstart

    la_labels = [request.param for _ in range(len(la_ends))]

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'target_labels': la_labels,
        'default_label': request.param
    }


def test_SequenceLabels_labels_at_outside(
        SequenceLabels_small_seqdata_labels_at_outside):
    s, ends, tlabels, deflabel = [
        SequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends', 'target_labels', 'default_label']
    ]

    # not passing default label == passing None
    labels = s.labels_at(ends, default_label=deflabel)

    assert all([e == r for e, r in zip(tlabels, labels)]), list(
        zip(tlabels, labels))


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def SequenceLabels_small_seqdata_labels_at_general(request,
                                                   init_small_seqdata):
    """ fixture with labels_at at different samplerates for general case

    General case where ends can be outside the starts_ends as well

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.SequenceLabels(se, l, samplerate=sr)

    la_sr = request.param
    with s.samplerate_as(la_sr):
        _se = s.starts_ends
        mins = _se[:, 0].min()
        maxe = _se[:, 1].max() + (1 / la_sr)

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append(l)

        la_ends.extend([mins, maxe])
        la_labels.extend([None, None])
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append([l])

        la_ends.extend([mins, maxe])
        la_labels.extend([None, None])

    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
    }


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_allwithin(request,
                                                          init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.SequenceLabels(se, l, samplerate=sr)

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append((l, ))

    la_sr = request.param
    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
    }


def test_ContiSequenceLabels_labels_at_allwithin(
        ContiSequenceLabels_small_seqdata_labels_at_allwithin):
    s, la_ends, lasr, target_labels = [
        ContiSequenceLabels_small_seqdata_labels_at_allwithin[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, samplerate=lasr)

    assert all([e == r for e, r in zip(target_labels, labels)]), list(
        zip(target_labels, labels))


# TODO: Test for multi-dimensional labels
# TODO: Test for non-numerical labels
