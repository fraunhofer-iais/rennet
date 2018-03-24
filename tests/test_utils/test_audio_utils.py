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
"""Test the audio utilities module

@mojuste
Created: 18-08-2016
"""
from __future__ import print_function, division
from glob import glob
from tempfile import NamedTemporaryFile
from math import ceil
import pytest
from numpy.testing import assert_almost_equal

import rennet.utils.audio_utils as au
import rennet.utils.pydub_utils as pu

# pylint: disable=redefined-outer-name, invalid-name
ValidAudioFile = au.AudioMetadata

test_1_wav = ValidAudioFile(
    "./data/test/test1.wav",  # NOTE: Running from the project root
    'wav',
    32000,
    2,
    10.00946875,
    320303
)

test_1_96k_wav = ValidAudioFile(
    "./data/test/test1_96k.wav",  # NOTE: Running from the project root
    'wav',
    96000,
    2,
    10.00946875,
    960909
)

test_1_mp3 = ValidAudioFile(
    "./data/test/test1.mp3",  # NOTE: Running from the project root
    'mp3',
    32000,
    2,
    10.00946875,
    320303
)

test_1_mp4 = ValidAudioFile(
    "./data/test/creative_common.mp4",  # NOTE: Running from the project root
    'mp4',
    48000,
    2,
    2.2613333333333334,
    108544
)

WORKING_DATA_RAW_MEDIA = glob("./data/working/*/*/media/raw/*.*")


@pytest.fixture(scope="module", params=WORKING_DATA_RAW_MEDIA)
def working_data_raw_media(request):
    """ All raw media files for all projects and datasets """
    return request.param


@pytest.fixture(scope="module", params=[test_1_wav, test_1_mp3, test_1_96k_wav])
def valid_audio_files(request):
    """ Valid audio files for testing """
    return request.param


@pytest.fixture(scope="module", params=[test_1_wav, test_1_96k_wav])
def valid_wav_files(request):
    return request.param


@pytest.fixture(
    scope="module", params=[test_1_mp3, test_1_mp4, test_1_wav, test_1_96k_wav]
)
def valid_media_files(request):
    """ All valid media files
    ultimate one to pass for get_samplerate(...) ... etc
    """
    return request.param


def test_valid_wav_metadata(valid_wav_files):
    """ Test au.read_wavefile_metadata(...)"""

    filepath = valid_wav_files.filepath

    assert au.read_wavefile_metadata(filepath) == valid_wav_files


# AUDIOMETADATA ######################################################## AUDIOMETADATA #
@pytest.mark.skipif(not au.get_codec(), reason="No FFMPEG or AVCONV found")
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

    with pytest.warns(RuntimeWarning):
        metadata = au.read_audio_metadata_codec(filepath)

    assert metadata.samplerate == correct_sr
    assert metadata.nchannels == correct_noc

    assert_almost_equal(correct_duration, metadata.seconds, decimal=1)


@pytest.mark.skipif(not au.get_codec(), reason="No FFMPEG or AVCONV found")
@pytest.mark.filterwarnings('ignore:Metadata')
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


# NOTE: no dataset available in rennet, check rennet-x
# @pytest.mark.check_dataset
# def test_able_to_get_metadata_for_all_raw_dataset(working_data_raw_media):
#     """ Test if able to use au.get_audio_metadata(...) for all raw datasets """
#     fp = working_data_raw_media
#
#     if not fp.endswith("wav") and not au.get_codec():
#         pytest.skip("No FFMPEG or AVCONV found")
#     else:
#         _ = au.get_audio_metadata(fp)
#         assert True


# LOAD_AUDIO ############################################################## LOAD_AUDIO #
def test_load_audio_as_is(valid_media_files):
    correct_sr = valid_media_files.samplerate
    correct_ns = valid_media_files.nsamples
    correct_nc = valid_media_files.nchannels
    filepath = valid_media_files.filepath

    data, sr = au.load_audio(
        filepath, mono=False, samplerate=correct_sr, return_samplerate=True
    )

    assert sr == correct_sr

    # HACK: avconv and ffmpeg give different nsamples for non-wav
    if valid_media_files.format != "wav":
        assert len(data.shape) == correct_nc
        assert abs(data.shape[0] - correct_ns) <= correct_sr * 0.05
    else:
        assert data.shape == (correct_ns, correct_nc)


def test_load_audio_with_defaults(valid_media_files):
    correct_sr = valid_media_files.samplerate
    correct_ns = ceil(valid_media_files.nsamples * (8000 / correct_sr))
    filepath = valid_media_files.filepath

    data, sr = au.load_audio(filepath, return_samplerate=True)

    assert sr == 8000

    # HACK: avconv and ffmpeg give different nsamples for non-wav
    if valid_media_files.format != "wav":
        assert len(data.shape) == 1  # mono
        assert abs(data.shape[0] - correct_ns) <= 8000 * 0.05
    else:
        assert data.shape == (correct_ns, )  # mono

    data_defaults = au.load_audio(filepath)
    assert data_defaults.shape == data.shape
    assert_almost_equal(data_defaults, data)


# PYDUB_UTILS ############################################################ PYDUB_UTILS #
@pytest.mark.skipif(not au.get_codec(), reason="No FFMPEG or AVCONV found")
@pytest.mark.filterwarnings('ignore:Metadata')
def test_AudioIO_from_audiometadata(valid_media_files):
    """Test if the returned updated metadata is accurate"""

    # known unsipported functionality for >48kHz files
    if valid_media_files.samplerate <= 48000:
        _, updated_metadata = pu.AudioIO.from_audiometadata(valid_media_files)

        # HACK: avconv and ffmpeg give different results for mp3 format
        if valid_media_files.format == "mp3":
            vm = valid_media_files
            sr = vm.samplerate
            assert abs(vm.nsamples - updated_metadata.nsamples) <= sr * 5e-2
            assert_almost_equal(updated_metadata.seconds, vm.seconds, decimal=1)
        else:
            assert valid_media_files == updated_metadata
    else:
        pytest.skip(">48khz audio not supported by AudioIO")


@pytest.mark.skipif(not au.get_codec(), reason="No FFMPEG or AVCONV found")
@pytest.mark.filterwarnings('ignore:Metadata')
def test_AudioIO_get_numpy_data(valid_media_files):
    """ Test for correct nsamples and nchannels """

    correct_ns = valid_media_files.nsamples
    correct_noc = valid_media_files.nchannels

    if valid_media_files.samplerate <= 48000:
        data = pu.AudioIO.from_audiometadata(valid_media_files)[0].get_numpy_data()

        # HACK: avconv and ffmpeg give different nsamples for mp3
        if valid_media_files.format == "mp3":
            assert data.shape[1] == correct_noc
            sr = valid_media_files.samplerate
            assert abs(data.shape[0] - correct_ns) <= sr * 5e-2
        else:
            assert data.shape == (correct_ns, correct_noc)
    else:
        pytest.skip(">48khz audio not supported by AudioIO")


# NOTE: no dataset available in rennet, check rennet-x
# @pytest.mark.long_running
# @pytest.mark.check_dataset
# def test_able_to_create_AudioIO_for_all_raw_dataset(working_data_raw_media):
#     """ Test if able to create AudioIO object for all raw datasets """
#     fp = working_data_raw_media
#
#     if not fp.endswith("wav") and not au.get_codec():
#         pytest.skip("No FFMPEG or AVCONV found")
#     else:
#         _ = pu.AudioIO.from_file(fp)
#         assert True


@pytest.mark.filterwarnings('ignore:Metadata')
def test_AudioIO_export_standard(valid_media_files):
    vml = valid_media_files
    if vml.samplerate > 48000:
        pytest.skip(">48kHz audio not supported by AudioIO")
    s, um = pu.AudioIO.from_audiometadata(vml)

    with NamedTemporaryFile() as tfp:
        s.export_standard(tfp)
        nm = au.get_audio_metadata(tfp.name)
        assert nm.samplerate == 16000
        assert nm.nchannels == 1
        assert_almost_equal(nm.seconds, um.seconds, decimal=3)
