"""
@mojuste
Created: 18-08-2016

Test the audio utilities module
"""
import pytest
from collections import namedtuple
from rennet.utils import audio_utils as au

# pylint: disable=redefined-outer-name
ValidAudioFile = namedtuple(
    'ValidAudioFile',
    [
        'filepath',
        'format',
        'samplerate',
        'channels',
        'seconds',
    ]
)

test_1_wav = ValidAudioFile(
    "./data/test/test1.wav",  # NOTE: Running from the project root
    'wav',
    32000,
    2,
    10.00946875
)

test_1_mp3 = ValidAudioFile(
    "./data/test/test1.mp3",  # NOTE: Running from the project root
    'mp3',
    32000,
    2,
    10.00946875
)

test_1_mp4 = ValidAudioFile(
    "./data/test/creative_common.mp4",  # NOTE: Running from the project root
    'mp4',
    48000,
    2,
    2.26133334
)


@pytest.fixture(scope="module", params=[test_1_wav, test_1_mp3])
def valid_audio_file(request):
    """ A valid wav file for testing

    The test1.wav is assumed to exist
    """
    return request.param

@pytest.mark.sanity_check
def test_valid_audio_samplerate(valid_audio_file):
    """ Test the audio_utils.get_samplerate(...) for valid wav file
    """
    filepath = valid_audio_file.filepath
    correct_sr = valid_audio_file.samplerate

    calculated_sr = au.get_samplerate(filepath)

    assert calculated_sr == correct_sr
