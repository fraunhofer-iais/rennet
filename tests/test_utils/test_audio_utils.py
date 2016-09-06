"""
@mojuste
Created: 18-08-2016

Test the audio utilities module
"""
import pytest
from numpy.testing import assert_almost_equal
from glob import glob
from tempfile import NamedTemporaryFile

# pylint: disable=import-error
from rennet.utils import audio_utils as au
# pylint: enable=import-error

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
    10.00946875,
    320303)

test_1_mp4 = ValidAudioFile(
    "./data/test/creative_common.mp4",  # NOTE: Running from the project root
    'mp4',
    48000,
    2,
    2.2613333333333334,
    108544)

WORKING_DATA_RAW_MEDIA = glob("./data/working/*/*/media/raw/*.*")


@pytest.fixture(scope="module", params=WORKING_DATA_RAW_MEDIA)
def working_data_raw_media(request):
    """ All raw media files for all projects and datasets """
    return request.param


@pytest.fixture(
    scope="module", params=[test_1_wav, test_1_mp3, test_1_96k_wav])
def valid_audio_files(request):
    """ Valid audio files for testing """
    return request.param


@pytest.fixture(scope="module", params=[test_1_wav, test_1_96k_wav])
def valid_wav_files(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[test_1_mp3, test_1_mp4, test_1_wav, test_1_96k_wav])
def valid_media_files(request):
    """
    ultimate one to pass for get_samplerate(...) ... etc
    """
    return request.param


def test_valid_wav_metadata(valid_wav_files):
    """ Test au.read_wavefile_metadata(...)"""

    filepath = valid_wav_files.filepath

    assert au.read_wavefile_metadata(filepath) == valid_wav_files


@pytest.mark.skipif(
    not au.get_codec(), reason="FFMPEG not available")
def test_valid_media_metadata_codec(valid_media_files):
    """ test au.read_audio_metadata_codec(...)
    The function does not return exact nsamples or seconds
    It is expected that the function will raise a RuntimeWarning for that
    Such files will be converted to wav before reading anyway
    """
    filepath = valid_media_files.filepath
    correct_sr = valid_media_files.samplerate
    correct_noc = valid_media_files.nchannels
    correct_duration = valid_media_files.seconds

    # TODO: [A] Test for raised warnings
    metadata = au.read_audio_metadata_codec(filepath)

    assert metadata.samplerate == correct_sr
    assert metadata.nchannels == correct_noc

    assert_almost_equal(correct_duration, metadata.seconds, decimal=1)


@pytest.mark.skipif(
    not au.get_codec(), reason="FFMPEG not available")
def test_valid_audio_metadata(valid_media_files):
    """ Test the audio_utils.get_audio_metadata(...) for valid wav file"""
    filepath = valid_media_files.filepath
    fmt = valid_media_files.format

    metadata = au.get_audio_metadata(filepath)
    if fmt == 'wav':
        assert valid_media_files == metadata
    else:
        assert metadata.samplerate == valid_media_files.samplerate
        assert metadata.nchannels == valid_media_files.nchannels


@pytest.mark.skipif(
    not au.get_codec(), reason="FFMPEG not available")
def test_AudioIO_from_audiometadata(valid_media_files):
    """Test if the returned updated metadata is accurate"""

    # known unsipported functionality for >48kHz files
    if valid_media_files.samplerate <= 48000:
        _, updated_metadata = au.AudioIO.from_audiometadata(valid_media_files)

        # HACK: avconv and ffmpeg give different results for mp3 format
        if valid_media_files.format == "mp3":
            vm = valid_media_files
            sr = vm.samplerate
            assert abs(vm.nsamples - updated_metadata.nsamples) <= sr * 5e-2
            assert_almost_equal(updated_metadata.seconds, vm.seconds,
                    decimal=1)
        else:
            assert valid_media_files == updated_metadata
    else:
        pytest.skip(">48khz audio not supported by AudioIO")


@pytest.mark.skipif(
    not au.get_codec(), reason="FFMPEG not available")
def test_AudioIO_get_numpy_data(valid_media_files):
    """ Test for correct nsamples and nchannels """

    correct_ns = valid_media_files.nsamples
    correct_noc = valid_media_files.nchannels

    if valid_media_files.samplerate <= 48000:
        data = au.AudioIO.from_audiometadata(valid_media_files)[
            0].get_numpy_data()

        # HACK: avconv and ffmpeg give different nsamples for mp3
        if valid_media_files.format == "mp3":
            assert data.shape[1] == correct_noc
            sr = valid_media_files.samplerate
            assert abs(data.shape[0] - correct_ns) <= sr * 5e-2
        else:
            assert data.shape == (correct_ns, correct_noc)
    else:
        pytest.skip(">48khz audio not supported by AudioIO")


@pytest.mark.check_dataset
def test_able_to_get_metadata_for_all_raw_dataset(working_data_raw_media):
    """ Test if able to use au.get_audio_metadata(...) for all raw datasets """
    fp = working_data_raw_media

    if not fp.endswith("wav") and not au.get_codec():
        pytest.skip("FFMPEG not available")
    else:
        _ = au.get_audio_metadata(fp)
        assert True


@pytest.mark.long_running
@pytest.mark.check_dataset
def test_able_to_create_AudioIO_for_all_raw_dataset(working_data_raw_media):
    """ Test if able to create AudioIO object for all raw datasets """
    fp = working_data_raw_media

    if not fp.endswith("wav") and not au.get_codec():
        pytest.skip("FFMPEG not available")
    else:
        _ = au.AudioIO.from_file(fp)
        assert True


def test_AudioIO_export_standard(valid_media_files):
    vml = valid_media_files
    if vml.samplerate > 48000:
        pytest.skip(">48kHz audio not supported by AudioIO")
    s, um = au.AudioIO.from_audiometadata(vml)

    with NamedTemporaryFile() as tfp:
        s.export_standard(tfp)
        nm = au.get_audio_metadata(tfp.name)
        assert nm.samplerate == 16000
        assert nm.nchannels == 1
        assert_almost_equal(nm.seconds, um.seconds, decimal=3)
