#  Copyright 2018 Fraunhofer IAIS. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Test the label utilities module

@motjuste
Created: 26-08-2016
"""
from __future__ import print_function, division
from six.moves import zip
import pytest
import numpy as np
import numpy.testing as npt
# from collections import namedtuple

from rennet.utils import label_utils as lu

# pylint: disable=redefined-outer-name, invalid-name, missing-docstring


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
        (1., 0),
        (1.4, 0),
        (3.8, 2),
        (5.5, 1),
        (5.9999375, 1),
        # (6, 1),
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
        (1., (0, )),
        (3.4, (
            0,
            2,
        )),
        (4.8, (
            2,
            1,
        )),
        (5.5, (1, )),
        # (6, (1, )),
    ]

    iscontiguous = False

    return starts_secs, ends_secs, labels, samplerate, labels_at_secs_nonconti, iscontiguous


@pytest.fixture(
    scope='module',
    params=[
        base_contiguous_small_seqdata(),
        base_noncontiguous_small_seqdata(),
    ],
    ids=['contiguous', 'non-contiguous']
)
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
    ids=lambda x: "SR={}".format(x)  # pylint: disable=unnecessary-lambda
)  # pylint: disable=unnecessary-lambda
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
        assert all([e == r for e, r in zip(l, s.labels)]), list(zip(l, s.labels))
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
    ids=lambda x: x.__name__
)
def seqlabelinst_small_seqdata(request, init_small_seqdata):
    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    if (
            not init_small_seqdata['isconti']
            and request.param is lu.ContiguousSequenceLabels
    ):  # yapf: disable
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )
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
        se_to_sr_small_seqdata_SequenceLabels[k] for k in [
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
        threelevel_sr_as_SeqLabels[k] for k in ['seqlabelinst', 'orig_sr', 'to_sr_cycle']
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
def SequenceLabels_small_seqdata_labels_at_allwithin(request, init_small_seqdata):
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
        SequenceLabels_small_seqdata_labels_at_allwithin
):  # yapf: disable
    s, la_ends, lasr, target_labels = [
        SequenceLabels_small_seqdata_labels_at_allwithin[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, samplerate=lasr)

    assert all([e == r for e, r in zip(target_labels, labels)]), list(
        zip(target_labels, labels)
    )


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def SequenceLabels_small_seqdata_labels_at_outside(request, init_small_seqdata):
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
    la_ends = [sminstart - (1 / sr), smaxend, smaxend + (1 / sr)]
    # Yes, there is no label for smaxend. So the default_label is expected
    # Why? We are looking at the label for the segment between
    # (x) and (x + (1/samplerate)) when finding labels_at
    # and we don't have any info about the label after smaxend
    # It's like array indexing (there is no element at len(arr)),
    # or 24-hr clocks (there is 24:00:00 for a date)

    la_labels = [request.param for _ in range(len(la_ends))]

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'target_labels': la_labels,
        'default_label': request.param
    }


def test_SequenceLabels_labels_at_outside(SequenceLabels_small_seqdata_labels_at_outside):
    s, ends, tlabels, deflabel = [
        SequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends', 'target_labels', 'default_label']
    ]

    # not passing default label == passing None
    labels = s.labels_at(ends, default_label=deflabel)

    assert all([e == r for e, r in zip(tlabels, labels)]), list(zip(tlabels, labels))


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def SequenceLabels_small_seqdata_labels_at_general(request, init_small_seqdata):
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
        mins = _se[:, 0].min() - (1 / la_sr)
        maxe = _se[:, 1].max()

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


def test_SequenceLabels_labels_at_general(SequenceLabels_small_seqdata_labels_at_general):
    s, la_ends, lasr, target_labels = [
        SequenceLabels_small_seqdata_labels_at_general[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, lasr, None)

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels)
    )

    assert s.labels_at(la_ends[0], lasr, None)[-1] == labels[0]


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_allwithin(request, init_small_seqdata):
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
            "will fail to initialize"
        )
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
        ContiSequenceLabels_small_seqdata_labels_at_allwithin
):  # yapf: disable
    s, la_ends, lasr, target_labels = [
        ContiSequenceLabels_small_seqdata_labels_at_allwithin[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, samplerate=lasr)

    assert all([e == r for e, r in zip(target_labels, labels)]), list(
        zip(target_labels, labels)
    )


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_outside(request, init_small_seqdata):
    """ fixture with labels_at at different samplerates

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    _l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, _l, samplerate=sr)

    sminstart = s.starts_ends[:, 0].min()
    smaxend = s.starts_ends[:, 1].max()
    la_ends = [sminstart - (1 / sr), smaxend, smaxend + (1 / sr)]
    # Yes, there is no label for smaxend. So the default_label is expected
    # Why? We are looking at the label for the segment between
    # (x) and (x + (1/samplerate)) when finding labels_at
    # and we don't have any info about the label after smaxend
    # It's like array indexing (there is no element at len(arr)),
    # or 24-hr clocks (there is 24:00:00 for a date)

    la_labels = [request.param for _ in range(len(la_ends))]

    return {
        'seqlabelinst': s,
        'ends': la_ends,
        'target_labels': la_labels,
        'default_label': request.param
    }


def test_ContiSequenceLabels_labels_at_outside_with_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_outside
):  # yapf: disable
    s, ends, tlabels, deflabel = [
        ContiSequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends', 'target_labels', 'default_label']
    ]

    # when default_label is passed
    labels = s.labels_at(ends, default_label=deflabel)

    assert all([e == r for e, r in zip(tlabels, labels)]), list(zip(tlabels, labels))

    with pytest.raises(KeyError):
        s.labels_at(ends, default_label='raise')


def test_ContiSequenceLabels_labels_at_outside_with_auto_deflabel(
        ContiSequenceLabels_small_seqdata_labels_at_outside
):  # yapf: disable
    s, ends = [
        ContiSequenceLabels_small_seqdata_labels_at_outside[k]
        for k in ['seqlabelinst', 'ends']
    ]
    zlabels = np.zeros(shape=((len(ends), ) + s.labels.shape[1:]), dtype=s.labels.dtype)

    # when default_label is to be determined automatically
    labels = s.labels_at(ends, default_label='zeros')

    assert all([e == r for e, r in zip(zlabels, labels)]), list(zip(zlabels, labels))


@pytest.fixture(
    scope='module',
    params=[1., 3., 3, 101, 1000, 8000, 16000],  # samplerate for labels_at
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_general(request, init_small_seqdata):
    """ fixture with labels_at at different samplerates for general case

    General case where ends can be outside the starts_ends as well

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, l, samplerate=sr)

    la_sr = request.param
    with s.samplerate_as(la_sr):
        _se = s.starts_ends
        mins = _se[:, 0].min() - (1 / la_sr)
        maxe = _se[:, 1].max()

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )
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
        ContiSequenceLabels_small_seqdata_labels_at_general
):  # yapf: disable
    s, la_ends, lasr, target_labels = [
        ContiSequenceLabels_small_seqdata_labels_at_general[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels']
    ]

    labels = s.labels_at(la_ends, lasr, default_label='zeros')

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels)
    )

    assert s.labels_at(la_ends[0], lasr, default_label='zeros')[-1] == target_labels[0]


@pytest.fixture(
    scope='module',
    params=[None, [], -1, [-1]],  # expected default labels
    ids=lambda x: "laSR={}".format(x)  #pylint: disable=unnecessary-lambda
)
def ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel(
        request, init_small_seqdata
):  # yapf: disable
    """ fixture with labels_at at different samplerates for general case

    General case where ends can be outside the starts_ends as well

    And of course instance of SequenceLabels class that handles both
    contiguous and non-contiguous seqdata
    """
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )

    se = init_small_seqdata['starts_ends']
    sr = init_small_seqdata['samplerate']
    l = init_small_seqdata['labels']

    s = lu.ContiguousSequenceLabels(se, l, samplerate=sr)

    la_sr = 1.0
    with s.samplerate_as(la_sr):
        _se = s.starts_ends
        mins = _se[:, 0].min() - (1 / la_sr)
        maxe = _se[:, 1].max()

    la_ends, la_labels = [], []
    if not init_small_seqdata['isconti']:
        pytest.skip(
            "Non-Contiguous Sequence data for ContiguousSequenceLabels "
            "will fail to initialize"
        )
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
        ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel
):  # yapf: disable
    s, la_ends, lasr, target_labels, deflabel = [
        ContiSequenceLabels_small_seqdata_labels_at_general_with_deflabel[k]
        for k in ['seqlabelinst', 'ends', 'at_sr', 'target_labels', 'deflabel']
    ]

    labels = s.labels_at(la_ends, lasr, default_label=deflabel)

    assert all([e == r for e, r in zip(target_labels, labels)]), ", ".join(
        "({} {})".format(e, t) for e, t in zip(target_labels, labels)
    )


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
        shifted_start_same_sr_small_seqdata[k] for k in [
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
def shifted_start_param_sr_small_seqdata(request, shifted_start_same_sr_small_seqdata):
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


def test_shifted_start_param_sr(  # pylint: disable=too-many-statements
        shifted_start_param_sr_small_seqdata):
    s, osr, tsr, tse, tms, tme, ose, oms, ome = [
        shifted_start_param_sr_small_seqdata[k] for k in [
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


@pytest.fixture(
    scope='module',
)
def from_dense_labels_start_and_sr_small_seqdata(shifted_start_param_sr_small_seqdata):
    """ Fixture for testing from_dense_labels with different start and samplerates"""
    s = shifted_start_param_sr_small_seqdata['seqlabelinst']
    tsr = shifted_start_param_sr_small_seqdata['t_sr']
    if tsr < 100:
        tsr = int(tsr * 1000)
    else:
        tsr = int(tsr)

    tms = shifted_start_param_sr_small_seqdata['t_ms']

    with s.min_start_as(tms, tsr):
        if s.__class__ is lu.SequenceLabels:
            se, li = s._flattened_indices()  # pylint: disable=protected-access
            l = []
            for i in li:
                if not i:
                    l.append(i)
                else:
                    l.append(tuple(s.labels[i, ...]))
        else:
            se, l = s.starts_ends, s.labels

        with s.min_start_as(0):
            ends = np.arange(s.max_end)
            labels = s.labels_at(ends)

    return {
        'labels': labels,
        't_sr': tsr,
        't_se': se,
        't_lb': l,
        't_ms': tms,
    }


@pytest.mark.dense
def test_from_dense_SeqLabels(from_dense_labels_start_and_sr_small_seqdata):
    labels, tsr, tse, tlb, tms = [
        from_dense_labels_start_and_sr_small_seqdata[k]
        for k in ['labels', 't_sr', 't_se', 't_lb', 't_ms']
    ]

    s = lu.SequenceLabels.from_dense_labels(
        labels, samplerate=tsr, min_start=tms, keep='keys'
    )

    npt.assert_allclose(s.starts_ends, tse, atol=1e-12, rtol=1)
    npt.assert_array_equal(s.labels, tlb)

    with pytest.raises(ValueError):
        s = lu.SequenceLabels.from_dense_labels(labels, keep=9)


@pytest.mark.dense
def test_from_dense_ContiSeqLabels(from_dense_labels_start_and_sr_small_seqdata):
    labels, tsr, tse, tlb, tms = [
        from_dense_labels_start_and_sr_small_seqdata[k]
        for k in ['labels', 't_sr', 't_se', 't_lb', 't_ms']
    ]

    s = lu.ContiguousSequenceLabels.from_dense_labels(
        labels, samplerate=tsr, min_start=tms, keep='keys'
    )

    npt.assert_allclose(s.starts_ends, tse, atol=1e-12, rtol=1)
    npt.assert_array_equal(s.labels, tlb)

    with pytest.raises(ValueError):
        s = lu.ContiguousSequenceLabels.from_dense_labels(labels, keep=9)


@pytest.fixture
def viterbi_wiki_data():
    # obs = ('normal', 'cold', 'dizzy')
    # states = ('Healthy', 'Fever')
    # start_p = {'Healthy': 0.6, 'Fever': 0.4}
    # trans_p = {
    #    'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
    #    'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
    #    }
    # emit_p = {
    #    'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    #    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
    #    }
    emit = np.array(
        [
            # normal, cold, dizzy
            [0.5, 0.4, 0.1],  # healthy
            [0.1, 0.3, 0.6],  # fever
        ]
    )
    obs = emit[:, [0, 1, 2]].T
    obs /= obs.sum(axis=1)[:, None]
    print(obs)

    init = np.array([0.6, 0.4])
    tran = np.array([
        # healthy, fever
        [0.7, 0.3],  # healthy
        [0.4, 0.6],  # fever
    ])

    preds = np.array([0, 0, 1])  # (healthy, healthy, fever)

    return {
        'obs': obs,
        'init': init,
        'tran': tran,
        'preds': preds,
    }


@pytest.mark.viterbi
def test_viterbi_smoothing_wiki_data(viterbi_wiki_data):
    w = viterbi_wiki_data
    e = w['preds']
    r = lu.viterbi_smoothing(w['obs'], w['init'], w['tran'].T)

    assert np.array_equal(e, r), str(e) + '\n' + str(r)


@pytest.fixture(scope='module')
def viterbi_priors_for_small_contiseqdata(  # pylint: disable=too-many-locals
        ContiSequenceLabels_small_seqdata_labels_at_allwithin):
    """ Fixture with raw and normalized Viterbi priors from ContiguousSequenceLabels """
    fixt = ContiSequenceLabels_small_seqdata_labels_at_allwithin

    S = fixt['seqlabelinst']
    round_to_int = True
    samplerate = fixt['at_sr']
    if samplerate % 10 != 0:  #or S.samplerate % 10 != 0:
        pytest.skip(
            "Testing for ContiguousSequenceLabels with non-zero decimal "
            "ends is not clear and not tested"
        )
        # TODO: Clarify and testing Viterbi Priors for non-integral multiples of samplerates
        # and perhaps chnage round_to_int

    with S.samplerate_as(samplerate):
        ends = lu.samples_for_labelsat(S.max_end - S.min_start, 1, 1)
        labels_at = S.labels_at(ends)

    unique_states = {ul: i for i, ul in enumerate(sorted(set(labels_at)))}

    init = np.zeros(len(unique_states), dtype=np.int)
    init[unique_states[labels_at[0]]] = 1

    transitions = np.zeros(shape=(len(unique_states), len(unique_states)), dtype=np.int)
    priors = np.zeros(len(unique_states), dtype=np.int)
    for l1, l2 in zip(labels_at[:-1], labels_at[1:]):
        transitions[unique_states[l1], unique_states[l2]] += 1
        priors[unique_states[l1]] += 1
    priors[unique_states[labels_at[-1]]] += 1

    # Normalization
    norm_transitions = transitions / transitions.sum(axis=1)[..., None]
    norm_init = init.astype(np.float)
    norm_priors = priors / priors.sum()

    return {
        'seqlabelinst': S,
        'at_sr': samplerate,
        'ustates': unique_states,
        'raw_init': init,
        'raw_tran': transitions,
        'raw_priors': priors,
        'norm_init': norm_init,
        'norm_tran': norm_transitions,
        'norm_priors': norm_priors,
        'state_keyfn': lambda x: x,
        'round_to_int': round_to_int,
        'labels_at': labels_at,
    }


@pytest.mark.viterbi
def test_viterbi_priors_from_contiseqlabels(  # pylint: disable=too-many-locals
        viterbi_priors_for_small_contiseqdata):
    fixt = viterbi_priors_for_small_contiseqdata

    seqlabels = fixt['seqlabelinst']
    samplerate = fixt['at_sr']
    statekeyfn = fixt['state_keyfn']
    rinit_t = fixt['raw_init']
    rpriors_t = fixt['raw_priors']
    rtran_t = fixt['raw_tran']
    round_to_int = fixt['round_to_int']

    ustates_t = np.array(sorted(fixt['ustates'].keys()))

    ustates_p, rinit_p, rtran_p, rpriors_p = seqlabels.calc_raw_viterbi_priors(
        samplerate=samplerate,
        state_keyfn=statekeyfn,
        round_to_int=round_to_int,
    )

    npt.assert_equal(ustates_t, ustates_p)
    npt.assert_equal(rinit_t, rinit_p)
    npt.assert_equal(rpriors_t, rpriors_p)
    npt.assert_equal(rtran_t, rtran_p)

    # within a samplerate_as context
    with seqlabels.samplerate_as(samplerate):
        ustates_p, rinit_p, rtran_p, rpriors_p = seqlabels.calc_raw_viterbi_priors(
            state_keyfn=statekeyfn,
            round_to_int=round_to_int,
        )

    npt.assert_equal(ustates_t, ustates_p)
    npt.assert_equal(rinit_t, rinit_p)
    npt.assert_equal(rpriors_t, rpriors_p)
    npt.assert_equal(rtran_t, rtran_p)

    # normalized priors
    ninit_p, ntran_p = lu.normalize_raw_viterbi_priors(rinit_p, rtran_p)
    npriors_p = rpriors_p / rpriors_p.sum()

    npt.assert_allclose(fixt['norm_init'], ninit_p)
    npt.assert_allclose(fixt['norm_tran'], ntran_p)
    npt.assert_allclose(fixt['norm_priors'], npriors_p)


# TODO: Test importing and exporting eaf
# TODO: Test importing mpeg7
# TODO: Test for multi-dimensional labels
# TODO: Test ContiguousSequenceLabels for differet dtype labels
# TODO: Test for non-numerical labels
