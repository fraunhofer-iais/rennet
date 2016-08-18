"""
@mojuste
Created: 18-08-2016

Test the audio utilities module
"""
import pytest
from collections import namedtuple
from rennet.utils import audio_utils as au

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
    filepath = "../../data/test/test1.wav"
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
    print(correct_sr)
    assert False
