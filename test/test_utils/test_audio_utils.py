"""
@mojuste
Created: 18-08-2016

Test the audio utilities module
"""
import pytest
from rennet.utils import audio_utils as au
from numpy.testing import assert_almost_equal

# pylint: disable=redefined-outer-name
ValidAudioFile = au.AudioMetadata

test_1_wav = ValidAudioFile(
    "./data/test/test1.wav",  # NOTE: Running from the project root
    'wav',
    32000,
    2,
    10.00946875,
    320303)

test_1_96k_wav = ValidAudioFile(
    "./data/test/test1_96k.wav",  # NOTE: Running from the project root
    'wav',
    96000,
    2,
    10.00946875,
    960909)

test_1_mp3 = ValidAudioFile(
    "./data/test/test1.mp3",  # NOTE: Running from the project root
    'mp3',
    32000,
    2,
    10.04,  # FIXME: not sure of the correct duration
    320303)

test_1_mp4 = ValidAudioFile(
    "./data/test/creative_common.mp4",  # NOTE: Running from the project root
    'mp4',
    48000,
    2,
    2.27,
    None  # FIXME: What are the total number of samples expected
)


@pytest.fixture(scope="module",
                params=[test_1_wav, test_1_mp3, test_1_96k_wav])
def valid_audio_files(request):
    """ A valid wav file for testing

    The test1.wav is assumed to exist
    """
    return request.param


@pytest.fixture(scope="module", params=[test_1_wav, test_1_96k_wav])
def valid_wav_files(request):
    return request.param


@pytest.fixture(scope="module",
                params=[test_1_mp3, test_1_mp4, test_1_wav, test_1_96k_wav])
def valid_media_files(request):
    """
    ultimate one to pass for get_samplerate(...) ... etc
    """
    return request.param


def test_valid_wav_metadata(valid_wav_files):
    filepath = valid_wav_files.filepath
    correct_sr = valid_wav_files.samplerate
    correct_noc = valid_wav_files.nchannels
    correct_nsamples = valid_wav_files.nsamples
    correct_duration = valid_wav_files.seconds

    ns, noc, sr, ds = au.read_wavefile_metadata(filepath)

    assert sr == correct_sr
    assert noc == correct_noc
    assert ns == correct_nsamples
    assert_almost_equal(correct_duration, ds, decimal=3)


def test_valid_media_metadata_ffmpeg(valid_media_files):
    filepath = valid_media_files.filepath
    correct_sr = valid_media_files.samplerate
    correct_noc = valid_media_files.nchannels
    correct_duration = valid_media_files.seconds

    # TODO: Test for raised warnings
    _, noc, sr, ds = au.read_audio_metadata_ffmpeg(filepath)

    assert sr == correct_sr
    assert noc == correct_noc

    assert_almost_equal(correct_duration, ds, decimal=2)


def test_valid_audio_samplerate(valid_media_files):
    """ Test the audio_utils.get_samplerate(...) for valid wav file
    """
    filepath = valid_media_files.filepath
    correct_sr = valid_media_files.samplerate

    calculated_sr = au.get_samplerate(filepath)

    assert calculated_sr == correct_sr
