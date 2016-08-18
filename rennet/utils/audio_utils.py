"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from __future__ import print_function
import os
import struct
import warnings
from pydub import AudioSegment

FFMPEG_EXEC = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)


def _ffmpeg_available():
    """ Check if FFMPEG is available """

    envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)

    for envdir in envdir_list:
        possible_path = os.path.join(envdir, FFMPEG_EXEC)
        if os.path.isfile(possible_path) and os.access(possible_path, os.X_OK):
            return True

    # FFMPEG_EXEC was not found
    return False


def read_wavefile_metadata(filepath):
    """
    # Reference
        https://github.com/scipy/scipy/blob/v0.14.0/scipy/io/wavfile.py#L116
    """
    fid = open(filepath, 'rb')

    def _read_riff_chunk(fid):
        s = fid.read(4)
        big_endian = False

        if s == b'RIFX':
            big_endian = True
        elif s != b'RIFF':
            raise ValueError("Not a WAVE file")

        if big_endian:
            fmt = '>I'
        else:
            fmt = '<I'

        size = struct.unpack(fmt, fid.read(4))[0] + 8

        s2 = fid.read(4)
        if s2 != b'WAVE':
            raise ValueError("Not a WAVE file")

        return size, big_endian

    def _read_fmt_chunk(fid, big_endian):
        if big_endian:
            fmt = '>'
        else:
            fmt = '<'

        res = struct.unpack(fmt + 'iHHIIHH', fid.read(20))
        size2, compression, channels, samplerate, _, _, bits = res

        if compression not in KNOWN_WAVE_FORMATS or size2 > 16:
            warnings.warn("Unknown wave file format", RuntimeWarning)

        return channels, samplerate, bits


    def _read_n_samples(fid, big_endian, bits):
        if big_endian:
            fmt = '>i'
        else:
            fmt = '<i'

        size = struct.unpack(fmt, fid.read(4))[0]

        return size // (bits // 8)

    try:
        size, big_endian = _read_riff_chunk(fid)

        while fid.tell() < size:
            chunk = fid.read(4)
            if chunk == b'fmt ':
                channels, samplerate, bits = _read_fmt_chunk(fid, big_endian)
            elif chunk == b'data':
                n_samples = _read_n_samples(fid, big_endian, bits)
                break
    finally:
        fid.close()

    return n_samples//channels, channels, samplerate


def get_samplerate(filepath):
    """ Get the sample rate of an audio file without reading all of it

    NOTE: Tested only on formats [wav, mp3, mp4]
    NOTE: for file formats other than wav, requires FFMPEG installed

    # Arguments
        filepath: path to audio file

    # Plan
        - check if ffmeg is available
        - if yes
            + get the information from ffmpeg error stdout
        - if no
            + check if is wav file
            + if yes
                - do the fmt_chunk thing from wave module
            + if no
                - raise exception


    TODO: check if WAV, then read metadata, else use pydub
    TODO: replace mentions of FFMPEG, instead delegate to pydub, because, it is
        okay fast, since it also does not read the complete file straightaway
        but may involve format conversion.
    TODO: reaname samples thing, refactor
    """
    if _ffmpeg_available():
        return AudioSegment.from_file(filepath).frame_rate
    else:
        try:
            _, _, samplerate = read_wavefile_metadata(filepath)
        except ValueError:
            raise RuntimeError(
                "Neither FFMPEG was found, nor is file a valid WAVE file")

    return samplerate

# TODO: Raise warning when samplerate > 48000 as pydub won't be able to support it
