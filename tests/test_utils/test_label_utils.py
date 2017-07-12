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

def test_SequenceLabels_doesnt_init():
    # Non-iterables
    with pytest.raises(TypeError):
        lu.SequenceLabels(0, 0, 1)

    # iterables of mismatching length
    with pytest.raises(AssertionError):
        lu.SequenceLabels([0], [0, 1])

    # starts_ends not like starts_ends
    with pytest.raises(AssertionError):
        lu.SequenceLabels([0], [[0, 1]])
    with pytest.raises(AssertionError):
        lu.SequenceLabels([[0, 1, 2]], [[0, 1]])

    # ends before starts
    with pytest.raises(ValueError):
        lu.SequenceLabels([[1, 0]], [[0, 1]])

    # bad samplerate
    with pytest.raises(ValueError):
        lu.SequenceLabels([[0, 1]], [[0, 1]], 0)
    with pytest.raises(ValueError):
        lu.SequenceLabels([[0, 1]], [[0, 1]], -1)

def test_ContiguousSequenceLabels_doesnt_init():
    # Non-iterables
    with pytest.raises(TypeError):
        lu.ContiguousSequenceLabels(0, 0, 1)

    # iterables of mismatching length
    with pytest.raises(AssertionError):
        lu.ContiguousSequenceLabels([0], [0, 1])

    # starts_ends not like starts_ends
    with pytest.raises(AssertionError):
        lu.ContiguousSequenceLabels([0], [[0, 1]])
    with pytest.raises(AssertionError):
        lu.ContiguousSequenceLabels([[0, 1, 2]], [[0, 1]])

    # ends before starts
    with pytest.raises(ValueError):
        lu.ContiguousSequenceLabels([[1, 0]], [[0, 1]])

    # bad samplerate
    with pytest.raises(ValueError):
        lu.ContiguousSequenceLabels([[0, 1]], [[0, 1]], 0)
    with pytest.raises(ValueError):
        lu.ContiguousSequenceLabels([[0, 1]], [[0, 1]], -1)

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
        (3.4, (0, 2, )),
        (4.8, (0, 2, 1, )),
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
        'minstart': min(starts_secs),
        'maxend': max(ends_secs),
    }


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 8000, 16000],  # samplerate
    ids=lambda x: "SR={}".format(x))  # pylint: disable=unnecessary-lambda
def init_small_seqdata(request, base_small_seqdata):
    sr = request.param
    _srmult = (sr / base_small_seqdata['samplerate'])
    se = base_small_seqdata['starts_ends'] * _srmult
    minstart = base_small_seqdata['minstart'] * _srmult
    maxend = base_small_seqdata['maxend'] * _srmult

    return {
        'starts_ends': se,
        'samplerate': sr,
        'labels': base_small_seqdata['labels'],
        'isconti': base_small_seqdata['isconti'],
        'labels_at': base_small_seqdata['labels_at'],
        'minstart': minstart,
        'maxend': maxend,
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
    assert s.min_start == init_small_seqdata['minstart']
    assert s.max_end == init_small_seqdata['maxend']
    assert all([e == r for e, r in zip(l, s.labels)]), list(zip(l, s.labels))
    assert len(s) == len(l)

    for ose, ol, (sse, sl) in zip(se[-2:, ...], l[-2:], s[-2:]):
        assert all(x == y for x, y in zip(ose, sse))
        assert ol == sl

    for sse, sl in s[0]:
        assert all(x == y for x, y in zip(se[0, ...], sse))
        assert sl == l[0]

    print(s)


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
        assert s.min_start == init_small_seqdata['minstart']
        assert s.max_end == init_small_seqdata['maxend']
        assert all([e == r
                    for e, r in zip(l, s.labels)]), list(zip(l, s.labels))
        assert len(s) == len(l)

        for ose, ol, (sse, sl) in zip(se[-2:, ...], l[-2:], s[-2:]):
            assert all(x == y for x, y in zip(ose, sse))
            assert ol == sl

        print(s)
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
        'minstart': init_small_seqdata['minstart'],
        'maxend': init_small_seqdata['maxend'],
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
    minstart = seqlabelinst_small_seqdata['minstart']
    maxend = seqlabelinst_small_seqdata['maxend']

    to_sr = request.param
    _srmult = to_sr / sr
    target_se = se * _srmult
    target_minstart = minstart * _srmult
    target_maxend = maxend * _srmult

    return {
        'seqlabelinst': s,
        'orig_sr': sr,
        'target_sr': to_sr,
        'target_se': target_se,
        't_minstart': target_minstart,
        't_maxend': target_maxend,
    }


def test_se_to_sr_SeqLabels(se_to_sr_small_seqdata_SequenceLabels):
    s, osr, tsr, tse, tms, tme = [
        se_to_sr_small_seqdata_SequenceLabels[k]
        for k in [
            'seqlabelinst',
            'orig_sr',
            'target_sr',
            'target_se',
            't_minstart',
            't_maxend',
        ]
    ]

    with s.samplerate_as(tsr):
        rse = s.starts_ends
        rsr = s.samplerate
        rosr = s.orig_samplerate
        rms = s.min_start
        rme = s.max_end

    assert rosr == osr
    assert rsr == tsr
    assert rms == tms
    assert rme == tme
    npt.assert_equal(rse, tse)

    with pytest.raises(ValueError):
        with s.samplerate_as(-1 * tsr):
            pass

    with pytest.raises(ValueError):
        with s.samplerate_as(0):
            pass


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

    assert all([e == r for e, r in zip(target_labels, labels)
                ]), list(zip(target_labels, labels))


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

    sminstart = s.starts_ends[:, 0].min()
    smaxend = s.starts_ends[:, 1].max()
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

    assert all([e == r
                for e, r in zip(tlabels, labels)]), list(zip(tlabels, labels))


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
        la_labels.extend((None, None))
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append((l, ))

        la_ends.extend([mins, maxe])
        la_labels.extend((None, None))

    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
    }


def test_SequenceLabels_labels_at_general(
        SequenceLabels_small_seqdata_labels_at_general):
    s, la_ends, lasr, target_labels = [
        SequenceLabels_small_seqdata_labels_at_general[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, lasr, None)

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels))

    assert s.labels_at(la_ends[0], lasr, None)[-1] == labels[0]

@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_allwithin(
        request, init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    _l = init_small_seqdata['labels']

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append(l)

    la_sr = request.param
    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    s = lu.ContiguousSequenceLabels(se, _l, samplerate=sr)

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

    assert all([e == r for e, r in zip(target_labels, labels)
                ]), list(zip(target_labels, labels))


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_outside(
        request, init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    _l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, _l, samplerate=sr)

    sminstart = s.starts_ends[:, 0].min()
    smaxend = s.starts_ends[:, 1].max()
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


def test_ContiSequenceLabels_labels_at_outside_with_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_outside):
    s, ends, tlabels, deflabel = [
        ContiSequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends', 'target_labels', 'default_label']
    ]

    # when default_label is passed
    labels = s.labels_at(ends, default_label=deflabel)

    assert all([e == r
                for e, r in zip(tlabels, labels)]), list(zip(tlabels, labels))

    with pytest.raises(KeyError):
        s.labels_at(ends, default_label='raise')


def test_ContiSequenceLabels_labels_at_outside_with_auto_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_outside):
    s, ends = [
        ContiSequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends']
    ]
    zlabels = np.zeros(
        shape=((len(ends), ) + s.labels.shape[1:]), dtype=s.labels.dtype)

    # when default_label is to be determined automatically
    labels = s.labels_at(ends, default_label='zeros')

    assert all([e == r
                for e, r in zip(zlabels, labels)]), list(zip(zlabels, labels))


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_general(
        request, init_small_seqdata):
    """ fixture with labels_at at different samplerates for general case

    General case where ends can be outside the starts_ends as well

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, l, samplerate=sr)

    la_sr = request.param
    with s.samplerate_as(la_sr):
        _se = s.starts_ends
        mins = _se[:, 0].min()
        maxe = _se[:, 1].max() + (1 / la_sr)

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append(l)

        la_ends.extend([mins, maxe])
        la_labels.extend([0, 0])

    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr
    la_labels = np.array(la_labels)

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
    }


def test_ContiSequenceLabels_labels_at_general_with_auto_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_general):
    s, la_ends, lasr, target_labels = [
        ContiSequenceLabels_small_seqdata_labels_at_general[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, lasr, default_label='zeros')

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels))

    assert s.labels_at(la_ends[0], lasr, default_label='zeros')[-1] == target_labels[0]


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel(
        request, init_small_seqdata):
    """ fixture with labels_at at different samplerates for general case

    General case where ends can be outside the starts_ends as well

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, l, samplerate=sr)

    la_sr = 1.0
    with s.samplerate_as(la_sr):
        _se = s.starts_ends
        mins = _se[:, 0].min()
        maxe = _se[:, 1].max() + (1 / la_sr)

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize")
    else:
        for e, l in init_small_seqdata['labels_at']:
            la_ends.append(e)
            la_labels.append(l)

        la_ends.extend([mins, maxe])
        la_labels.extend([request.param, request.param])

    # ends are more than likely to be provided as np.ndarray
    la_ends = np.array(la_ends) * la_sr

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'at_sr': la_sr,
        'target_labels': la_labels,
        'deflabel': request.param,
    }


def test_ContiSequenceLabels_labels_at_general_with_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel):
    s, la_ends, lasr, target_labels, deflabel = [
        ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel[k]
        for k in
        ['seqlabelinst', 'ends', 'at_sr', 'target_labels', 'deflabel']
    ]

    labels = s.labels_at(la_ends, lasr, default_label=deflabel)

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels))


@pytest.fixture(
    scope='module',
    params=[0, 1., 3., 100, 101],  # start_as
    ids=lambda x: "startas={}".format(x)  #pylint: disable=unnecessary-lambda
)
def shifted_start_same_sr_small_seqdata(request, seqlabelinst_small_seqdata):
    """ Fixture to test min_start, max_end and starts_ends with start_as at same sr"""
    s = seqlabelinst_small_seqdata['seqlabelinst']
    ms = seqlabelinst_small_seqdata['minstart']
    me = seqlabelinst_small_seqdata['maxend']
    sr = seqlabelinst_small_seqdata['orig_sr']
    se = seqlabelinst_small_seqdata['orig_se']

    new_start = request.param
    tms = new_start - ms
    tme = me + tms
    tse = se + tms
    tsr = sr

    return {
        'seqlabelinst': s,
        'orig_sr': sr,
        'orig_se': se,
        't_sr': tsr,
        't_se': tse,
        't_ms': tms,
        't_me': tme,
    }


def test_shifted_start_same_sr(shifted_start_same_sr_small_seqdata):
    s, osr, tsr, tse, tms, tme = [
        shifted_start_same_sr_small_seqdata[k]
        for k in [
            'seqlabelinst',
            'orig_sr',
            't_sr',
            't_se',
            't_ms',
            't_me',
        ]
    ]

    # Assume samplerate to be orig when param not set
    with s.min_start_as(tms):
        assert s.orig_samplerate == osr
        assert s.samplerate == osr
        assert s.min_start == tms
        assert s.max_end == tme
        npt.assert_array_equal(s.starts_ends, tse)

    with s.min_start_as(tms, samplerate=tsr):
        assert s.orig_samplerate == osr
        assert s.samplerate == tsr
        assert s.min_start == tms
        assert s.max_end == tme
        npt.assert_array_equal(s.starts_ends, tse)

    # raise error for negative samplerate
    with pytest.raises(ValueError):
        with s.min_start_as(tms, -1 * tsr):
            pass

    # raise error for zero samplerate
    with pytest.raises(ValueError):
        with s.min_start_as(tms, 0):
            pass


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 100, 101],  # to_samplerate
    ids=lambda x: "toSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def shifted_start_param_sr_small_seqdata(request,
                                         shifted_start_same_sr_small_seqdata):
    """ Fixture to test starts_ends are calculated correctly in contextual sr """
    s = shifted_start_same_sr_small_seqdata['seqlabelinst']
    sr = shifted_start_same_sr_small_seqdata['orig_sr']
    minstart = shifted_start_same_sr_small_seqdata['t_ms']
    maxend = shifted_start_same_sr_small_seqdata['t_me']
    se = shifted_start_same_sr_small_seqdata['t_se']
    ose = shifted_start_same_sr_small_seqdata['orig_se']

    to_sr = request.param
    _srmult = to_sr / sr
    target_se = se * _srmult
    target_minstart = minstart * _srmult
    target_maxend = maxend * _srmult

    return {
        'seqlabelinst': s,
        'orig_sr': sr,
        't_sr': to_sr,
        't_se': target_se,
        't_ms': target_minstart,
        't_me': target_maxend,
        'o_se': ose,
        'o_ms': ose[:, 0].min(),
        'o_me': ose[:, 1].max(),
    }


@pytest.mark.shifting
def test_shifted_start_param_sr(  # pylint: disable=too-many-statements
        shifted_start_param_sr_small_seqdata):
    s, osr, tsr, tse, tms, tme, ose, oms, ome = [
        shifted_start_param_sr_small_seqdata[k]
        for k in [
            'seqlabelinst',
            'orig_sr',
            't_sr',
            't_se',
            't_ms',
            't_me',
            'o_se',
            'o_ms',
            'o_me',
        ]
    ]

    assert s.orig_samplerate == osr
    assert s.samplerate == osr
    assert s.min_start == oms
    assert s.max_end == ome
    npt.assert_array_equal(s.starts_ends, ose)

    with s.min_start_as(tms, samplerate=tsr):
        assert s.orig_samplerate == osr
        assert s.samplerate == tsr
        npt.assert_almost_equal(s.min_start, tms)
        npt.assert_almost_equal(s.max_end, tme)
        npt.assert_almost_equal(s.starts_ends, tse)

    with s.samplerate_as(tsr):
        with s.min_start_as(tms):
            assert s.orig_samplerate == osr
            assert s.samplerate == tsr
            npt.assert_almost_equal(s.min_start, tms)
            npt.assert_almost_equal(s.max_end, tme)
            npt.assert_almost_equal(s.starts_ends, tse)

    with s.samplerate_as(tsr):
        assert s.orig_samplerate == osr
        assert s.samplerate == tsr
        with s.min_start_as(tms, tsr):
            assert s.orig_samplerate == osr
            assert s.samplerate == tsr
            npt.assert_almost_equal(s.min_start, tms)
            npt.assert_almost_equal(s.max_end, tme)
            npt.assert_almost_equal(s.starts_ends, tse)

    with s.samplerate_as(None):
        assert s.orig_samplerate == osr
        assert s.samplerate == osr
        assert s.min_start == oms
        assert s.max_end == ome
        npt.assert_almost_equal(s.starts_ends, ose)
        with s.min_start_as(tms, tsr):
            assert s.orig_samplerate == osr
            assert s.samplerate == tsr
            npt.assert_almost_equal(s.min_start, tms)
            npt.assert_almost_equal(s.max_end, tme)
            npt.assert_almost_equal(s.starts_ends, tse)

    with s.samplerate_as(osr):
        assert s.orig_samplerate == osr
        assert s.samplerate == osr
        assert s.min_start == oms
        assert s.max_end == ome
        npt.assert_almost_equal(s.starts_ends, ose)
        with s.min_start_as(tms, tsr):
            assert s.orig_samplerate == osr
            assert s.samplerate == tsr
            npt.assert_almost_equal(s.min_start, tms)
            npt.assert_almost_equal(s.max_end, tme)
            npt.assert_almost_equal(s.starts_ends, tse)


# TODO: Test for multi-dimensional labels
# TODO: Test ContiguousSequenceLabels for differet dtype labels
# TODO: Test for non-numerical labels
