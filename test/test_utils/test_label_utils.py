"""
@motjuste
Created: 26-08-2016

Test the label utilities module
"""
import pytest
import numpy as np
import numpy.testing as npt
from collections import namedtuple

from rennet.utils import label_utils as lu

# pylint: disable=redefined-outer-name

SeqLabelData = namedtuple('SeqLabelData', [
    'starts_secs', 'ends_secs', 'labels', 'samplerate', 'starts_samples',
    'ends_samples'
])


@pytest.fixture(scope='module')
def sample_seqlabel():
    starts_secs = np.array([0., 1., 3., 4.5])
    ends_secs = np.array([1., 3., 4.5, 6.])
    labels = [1, 0, 2, 1]
    samplerate = 16000
    starts_samples = starts_secs * samplerate
    ends_samples = ends_secs * samplerate

    return SeqLabelData(starts_secs, ends_secs, labels, samplerate,
                        starts_samples, ends_samples)


def test_SequenceLabels_from_samples(sample_seqlabel):
    sample_startsends = np.vstack([sample_seqlabel.starts_samples,
                                   sample_seqlabel.ends_samples]).T

    seqlabels = lu.SequenceLabels(sample_startsends, sample_seqlabel.labels,
                                  sample_seqlabel.samplerate)

    print(seqlabels)
    with seqlabels.samplerate_as(1.0):
        npt.assert_equal(seqlabels.starts, sample_seqlabel.starts_secs)
        npt.assert_equal(seqlabels.ends, sample_seqlabel.ends_secs)
        assert seqlabels.samplerate == sample_seqlabel.samplerate
        print(seqlabels)


def test_SequenceLabels_from_secs(sample_seqlabel):
    sample_startsends = np.vstack([sample_seqlabel.starts_secs,
                                   sample_seqlabel.ends_secs]).T

    seqlabels = lu.SequenceLabels(sample_startsends, sample_seqlabel.labels, 1)

    with seqlabels.samplerate_as(sample_seqlabel.samplerate):
        npt.assert_equal(seqlabels.starts, sample_seqlabel.starts_samples)
        npt.assert_equal(seqlabels.ends, sample_seqlabel.ends_samples)
        assert seqlabels.samplerate == 1
