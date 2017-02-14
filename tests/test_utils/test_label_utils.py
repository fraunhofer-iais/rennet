"""
@motjuste
Created: 26-08-2016

Test the label utilities module
"""
import pytest
import numpy as np
import numpy.testing as npt
# from collections import namedtuple

from rennet.utils import label_utils as lu

# pylint: disable=redefined-outer-name


@pytest.fixture(scope='module')
def base_contigious_small_seqdata():
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
def base_noncontigious_small_seqdata():
    starts_secs = np.array([0., 1., 3., 4.5])
    ends_secs = np.array([1., 4.8, 5., 6.])
    labels = [1, 0, 2, 1]
    samplerate = 1.

    labels_at_secs_nonconti = [
        (0.5, [1]),
        (1., [1]),
        (3.4, [0, 2]),
        (4.8, [0, 2, 1]),
        (5.5, [1]),
        (6, [1]),
    ]

    iscontiguous = False

    return starts_secs, ends_secs, labels, samplerate, labels_at_secs_nonconti, iscontiguous


@pytest.fixture(
    scope='module',
    params=[
        base_contigious_small_seqdata(),
        base_noncontigious_small_seqdata(),
    ],
    ids=['contiguous', 'non-contiguous'])
def base_small_seqdata(request):
    starts_secs, ends_secs, labels, samplerate, _, isconti = request.param
    se_secs = np.vstack([starts_secs, ends_secs]).T
    return {
        'starts_ends': se_secs,
        'samplerate': samplerate,
        'labels': labels,
        'isconti': isconti
    }


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101],  # samplerate
    ids=lambda x: "SR={}".format(x))  # pylint: disable=unnecessary-lambda
def init_small_seqdata(request, base_small_seqdata):
    sr = request.param
    se = (base_small_seqdata['starts_ends'] *
          (sr / base_small_seqdata['samplerate']))

    if isinstance(sr, int):
        se = se.astype(np.int)

    return {
        'starts_ends': se,
        'samplerate': sr,
        'labels': base_small_seqdata['labels'],
        'isconti': base_small_seqdata['isconti']
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
    params=[1., 3., 3, 101],  # to_samplerate
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


# SeqLabelData = namedtuple('SeqLabelData', [
#     'starts_secs', 'ends_secs', 'labels', 'samplerate', 'starts_samples',
#     'ends_samples', 'labels_at_secs_conti', 'labels_at_secs_nonconti',
#     'labels_at_samples_conti', 'labels_at_samples_nonconti',
#     'default_label_conti', 'default_label_nonconti',
#     'default_default_label_conti', 'default_default_label_nonconti'
# ])
#
#
# @pytest.fixture(scope='module')
# def sample_contigious_seqlabel():
#     starts_secs = np.array([0., 1., 3., 4.5])
#     ends_secs = np.array([1., 3., 4.5, 6.])
#     labels = [1, 0, 2, 1]
#     samplerate = 16000
#     starts_samples = (starts_secs * samplerate).astype(np.int)
#     ends_samples = (ends_secs * samplerate).astype(np.int)
#
#     labels_at_secs_conti = [
#         (0.5, 1),
#         (1., 1),
#         (1.4, 0),
#         (3.8, 2),
#         (5.5, 1),
#         (5.9999375, 1),
#         (6, 1),
#     ]
#     labels_at_secs_nonconti = [(t, [l]) for t, l in labels_at_secs_conti]
#
#     labels_at_samples_conti = [(t * samplerate, l)
#                                for t, l in labels_at_secs_conti]
#     labels_at_samples_nonconti = [(t * samplerate, l)
#                                   for t, l in labels_at_secs_nonconti]
#
#     default_label_conti = -1
#     default_label_nonconti = [default_label_conti, ]
#
#     # when default label is not passed
#     default_default_label_conti = None
#     default_default_label_nonconti = default_default_label_conti
#
#     return SeqLabelData(starts_secs, ends_secs, labels, samplerate,
#                         starts_samples, ends_samples, labels_at_secs_conti,
#                         labels_at_secs_nonconti, labels_at_samples_conti,
#                         labels_at_samples_nonconti, default_label_conti,
#                         default_label_nonconti, default_default_label_conti,
#                         default_default_label_nonconti)
#
#
# def test_SequenceLabels_from_samples(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_samples,
#         sample_contigious_seqlabel.ends_samples
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_contigious_seqlabel.labels,
#                                   sample_contigious_seqlabel.samplerate)
#
#     print(seqlabels)
#     with seqlabels.samplerate_as(1.0):
#         npt.assert_equal(seqlabels.starts,
#                          sample_contigious_seqlabel.starts_secs)
#         npt.assert_equal(seqlabels.ends, sample_contigious_seqlabel.ends_secs)
#         assert seqlabels.orig_samplerate == sample_contigious_seqlabel.samplerate
#         assert seqlabels.samplerate == 1.0
#         print(seqlabels)
#
#
# def test_SequenceLabels_from_secs(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_secs,
#         sample_contigious_seqlabel.ends_secs
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_contigious_seqlabel.labels, 1)
#
#     with seqlabels.samplerate_as(sample_contigious_seqlabel.samplerate):
#         npt.assert_equal(seqlabels.starts,
#                          sample_contigious_seqlabel.starts_samples)
#         npt.assert_equal(seqlabels.ends,
#                          sample_contigious_seqlabel.ends_samples)
#         assert seqlabels.samplerate == sample_contigious_seqlabel.samplerate
#         assert seqlabels.orig_samplerate == 1.0
#
#
# def test_ContigiousSequenceLabels_from_samples(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_samples,
#         sample_contigious_seqlabel.ends_samples
#     ]).T
#
#     seqlabels = lu.ContigiousSequenceLabels(
#         sample_startsends, sample_contigious_seqlabel.labels,
#         sample_contigious_seqlabel.samplerate)
#
#     print(seqlabels)
#     with seqlabels.samplerate_as(1.0):
#         npt.assert_equal(seqlabels.starts,
#                          sample_contigious_seqlabel.starts_secs)
#         npt.assert_equal(seqlabels.ends, sample_contigious_seqlabel.ends_secs)
#         assert seqlabels.orig_samplerate == sample_contigious_seqlabel.samplerate
#         assert seqlabels.samplerate == 1.0
#         print(seqlabels)
#
#
# def test_ContigiousSequenceLabels_from_secs(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_secs,
#         sample_contigious_seqlabel.ends_secs
#     ]).T
#
#     seqlabels = lu.ContigiousSequenceLabels(
#         sample_startsends, sample_contigious_seqlabel.labels, 1)
#
#     with seqlabels.samplerate_as(sample_contigious_seqlabel.samplerate):
#         npt.assert_equal(seqlabels.starts,
#                          sample_contigious_seqlabel.starts_samples)
#         npt.assert_equal(seqlabels.ends,
#                          sample_contigious_seqlabel.ends_samples)
#         assert seqlabels.orig_samplerate == 1.0
#         assert seqlabels.samplerate == sample_contigious_seqlabel.samplerate
#
#
# @pytest.fixture(scope='module')
# def sample_noncontigious_seqlabel():
#     starts_secs = np.array([0., 1., 3., 4.5])
#     ends_secs = np.array([1., 4.8, 5., 6.])
#     labels = [1, 0, 2, 1]
#     samplerate = 16000
#     starts_samples = (starts_secs * samplerate).astype(np.int)
#     ends_samples = (ends_secs * samplerate).astype(np.int)
#
#     labels_at_secs_nonconti = [
#         (0.5, [1]),
#         (1., [1]),
#         (3.4, [0, 2]),
#         (4.8, [0, 2, 1]),
#         (5.5, [1]),
#         (6, [1]),
#     ]
#     labels_at_secs_conti = None
#
#     labels_at_samples_conti = None
#     labels_at_samples_nonconti = [(t * samplerate, l)
#                                   for t, l in labels_at_secs_nonconti]
#
#     default_label_conti = None
#     default_label_nonconti = [-1, ]
#
#     # when default label is not passed
#     default_default_label_conti = None
#     default_default_label_nonconti = None
#
#     return SeqLabelData(starts_secs, ends_secs, labels, samplerate,
#                         starts_samples, ends_samples, labels_at_secs_conti,
#                         labels_at_secs_nonconti, labels_at_samples_conti,
#                         labels_at_samples_nonconti, default_label_conti,
#                         default_label_nonconti, default_default_label_conti,
#                         default_default_label_nonconti)
#
#
# def test_SequenceLabels_from_samples_nofail_noncontigious(
#         sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_samples,
#         sample_noncontigious_seqlabel.ends_samples
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_noncontigious_seqlabel.labels,
#                                   sample_noncontigious_seqlabel.samplerate)
#
#     print(seqlabels)
#     with seqlabels.samplerate_as(1.0):
#         npt.assert_equal(seqlabels.starts,
#                          sample_noncontigious_seqlabel.starts_secs)
#         npt.assert_equal(seqlabels.ends,
#                          sample_noncontigious_seqlabel.ends_secs)
#         assert seqlabels.orig_samplerate == sample_noncontigious_seqlabel.samplerate
#         assert seqlabels.samplerate == 1.0
#         print(seqlabels)
#
#
# def test_SequenceLabels_from_secs_nofail_noncontigious(
#         sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_secs,
#         sample_noncontigious_seqlabel.ends_secs
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_noncontigious_seqlabel.labels, 1)
#
#     with seqlabels.samplerate_as(sample_noncontigious_seqlabel.samplerate):
#         npt.assert_equal(seqlabels.starts,
#                          sample_noncontigious_seqlabel.starts_samples)
#         npt.assert_equal(seqlabels.ends,
#                          sample_noncontigious_seqlabel.ends_samples)
#         assert seqlabels.orig_samplerate == 1
#         assert seqlabels.samplerate == sample_noncontigious_seqlabel.samplerate
#
#
# def test_ContigiousSequenceLabels_from_samples_fail_noncontigious(
#         sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_samples,
#         sample_noncontigious_seqlabel.ends_samples
#     ]).T
#
#     with pytest.raises(AssertionError):
#         lu.ContigiousSequenceLabels(sample_startsends,
#                                     sample_noncontigious_seqlabel.labels,
#                                     sample_noncontigious_seqlabel.samplerate)
#
#
# def test_ContigiousSequenceLabels_from_secs_fail_noncontigious(
#         sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_secs,
#         sample_noncontigious_seqlabel.ends_secs
#     ]).T
#
#     with pytest.raises(AssertionError):
#         lu.ContigiousSequenceLabels(sample_startsends,
#                                     sample_noncontigious_seqlabel.labels, 1)
#
#
# def test_labels_at_SeqLabels_from_secs_conti(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_secs,
#         sample_contigious_seqlabel.ends_secs
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_contigious_seqlabel.labels, 1)
#
#     # test retrieving labels with seconds
#     la_t, la_expected_labels = [], []
#     for t, l in sample_contigious_seqlabel.labels_at_secs_nonconti:
#         la_t.append(t)
#         la_expected_labels.append(l)
#
#     la_labels = seqlabels.labels_at(la_t)
#
#     assert la_labels is not None
#     assert len(la_labels) == len(la_t)
#     assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#     # single end still returns a list
#     assert seqlabels.labels_at(la_t[-1]) == [la_expected_labels[-1]]
#
#     # test for different samplerate
#     sr = sample_contigious_seqlabel.samplerate
#     la_t_sr, la_expected_labels_sr = [], []
#     for t, l in sample_contigious_seqlabel.labels_at_samples_nonconti:
#         la_t_sr.append(t)
#         la_expected_labels_sr.append(l)
#
#     la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#
#     assert la_labels_sr is not None
#     assert len(la_labels_sr) == len(la_t_sr)
#     assert all([e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#     # test for contextually different samplerate
#     with seqlabels.samplerate_as(sample_contigious_seqlabel.samplerate):
#         la_labels_sr = seqlabels.labels_at(la_t_sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#
# def test_labels_at_SeqLabels_from_samples_conti(sample_contigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_contigious_seqlabel.starts_samples,
#         sample_contigious_seqlabel.ends_samples
#     ]).T
#
#     sr = sample_contigious_seqlabel.samplerate
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_contigious_seqlabel.labels, sr)
#
#     # test retrieving labels with seconds
#     la_t, la_expected_labels = [], []
#     for t, l in sample_contigious_seqlabel.labels_at_secs_nonconti:
#         la_t.append(t)
#         la_expected_labels.append(l)
#
#     la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#
#     # single end still returns a list
#     assert seqlabels.labels_at(la_t[-1]) == [la_expected_labels[-1]]
#
#     assert la_labels is not None
#     assert len(la_labels) == len(la_t)
#     assert all([e == r for e, r in zip(la_expected_labels, la_labels)]), list(
#         zip(la_expected_labels, la_labels))
#
#     # test for different samplerate
#     la_t_sr, la_expected_labels_sr = [], []
#     for t, l in sample_contigious_seqlabel.labels_at_samples_nonconti:
#         la_t_sr.append(t)
#         la_expected_labels_sr.append(l)
#
#     la_labels_sr = seqlabels.labels_at(la_t_sr)
#
#     assert la_labels_sr is not None
#     assert len(la_labels_sr) == len(la_t_sr)
#     assert all([e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#     # test for contextually different samplerate
#     with seqlabels.samplerate_as(1.0):
#         la_labels = seqlabels.labels_at(la_t)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#         la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#
# def test_labels_at_SeqLabels_from_secs_nonconti(sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_secs,
#         sample_noncontigious_seqlabel.ends_secs
#     ]).T
#
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_noncontigious_seqlabel.labels, 1)
#
#     # test retrieving labels with seconds
#     la_t, la_expected_labels = [], []
#     for t, l in sample_noncontigious_seqlabel.labels_at_secs_nonconti:
#         la_t.append(t)
#         la_expected_labels.append(l)
#
#     # single end still returns a list
#     assert seqlabels.labels_at(la_t[-1]) == [la_expected_labels[-1]]
#
#     la_labels = seqlabels.labels_at(la_t)
#
#     assert la_labels is not None
#     assert len(la_labels) == len(la_t)
#     assert all([e == r for e, r in zip(la_expected_labels, la_labels)]), list(
#         zip(la_expected_labels, la_labels))
#
#     # test for different samplerate
#     sr = sample_noncontigious_seqlabel.samplerate
#     la_t_sr, la_expected_labels_sr = [], []
#     for t, l in sample_noncontigious_seqlabel.labels_at_samples_nonconti:
#         la_t_sr.append(t)
#         la_expected_labels_sr.append(l)
#
#     la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#
#     assert la_labels_sr is not None
#     assert len(la_labels_sr) == len(la_t_sr)
#     assert all([e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#     # test for contextually different samplerate
#     with seqlabels.samplerate_as(sample_noncontigious_seqlabel.samplerate):
#         la_labels_sr = seqlabels.labels_at(la_t_sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#
# def test_labels_at_SeqLabels_from_samples_nonconti(
#         sample_noncontigious_seqlabel):
#     sample_startsends = np.vstack([
#         sample_noncontigious_seqlabel.starts_samples,
#         sample_noncontigious_seqlabel.ends_samples
#     ]).T
#
#     sr = sample_noncontigious_seqlabel.samplerate
#     seqlabels = lu.SequenceLabels(sample_startsends,
#                                   sample_noncontigious_seqlabel.labels, sr)
#
#     # test retrieving labels with seconds
#     la_t, la_expected_labels = [], []
#     for t, l in sample_noncontigious_seqlabel.labels_at_secs_nonconti:
#         la_t.append(t)
#         la_expected_labels.append(l)
#
#     la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#
#     # single end still returns a list
#     assert seqlabels.labels_at(la_t[-1]) == [la_expected_labels[-1]]
#
#     assert la_labels is not None
#     assert len(la_labels) == len(la_t)
#     assert all([e == r for e, r in zip(la_expected_labels, la_labels)]), list(
#         zip(la_expected_labels, la_labels))
#
#     # test for different samplerate
#     la_t_sr, la_expected_labels_sr = [], []
#     for t, l in sample_noncontigious_seqlabel.labels_at_samples_nonconti:
#         la_t_sr.append(t)
#         la_expected_labels_sr.append(l)
#
#     la_labels_sr = seqlabels.labels_at(la_t_sr)
#
#     assert la_labels_sr is not None
#     assert len(la_labels_sr) == len(la_t_sr)
#     assert all([e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#     # test for contextually different samplerate
#     with seqlabels.samplerate_as(1.0):
#         la_labels = seqlabels.labels_at(la_t)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#         la_labels_sr = seqlabels.labels_at(la_t_sr, samplerate=sr)
#         assert la_labels_sr is not None
#         assert len(la_labels_sr) == len(la_t_sr)
#         assert all(
#             [e == r for e, r in zip(la_expected_labels_sr, la_labels_sr)])
#
#         la_labels = seqlabels.labels_at(la_t, samplerate=1.0)
#         assert la_labels is not None
#         assert len(la_labels) == len(la_t)
#         assert all([e == r for e, r in zip(la_expected_labels, la_labels)])
#
#
# TODO: benchmark different labels_at approaches

# TODO: test for multi-level samplerate_as

# TODO: test for default, and default_default
