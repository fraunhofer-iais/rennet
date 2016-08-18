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
@pytest.fixture(scope="module")
def valid_wav_file():
    """ A valid wav file for testing

    The test1.wav is assumed to exist
    """
    filepath = "./data/test/test1.wav"  # NOTE: Running from the project root
    samplerate = 32000
    seconds = 10.0
    channels = 2

    return ValidAudioFile(filepath, 'wav', samplerate, channels, seconds)


@pytest.mark.sanity_check
def test_valid_wav_samplerate(valid_wav_file):
    """ Test the audio_utils.get_samplerate_wav(...) for valid wav file
    """
    filepath = valid_wav_file.filepath
    correct_sr = valid_wav_file.samplerate

    calculated_sr = au.get_samplerate_wav(filepath)

    assert calculated_sr == correct_sr
