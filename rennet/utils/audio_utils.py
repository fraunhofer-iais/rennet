"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from __future__ import print_function, division
import os
import struct
import warnings

# from pydub import AudioSegment

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

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

    def _skip_unknown_chunk(fid, big_endian):
        if big_endian:
            fmt = '>i'
        else:
            fmt = '<i'

        data = fid.read(4)
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)

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
            elif chunk in (b'fact', b'LIST'):
                _skip_unknown_chunk(fid, big_endian)
    finally:
        fid.close()

    duration_seconds = (n_samples // channels) / samplerate
    return n_samples // channels, channels, samplerate, duration_seconds


def read_audio_metadata_ffmpeg(filepath):
    """
    TODO: manage imports
    """
    import re
    import subprocess as sp

    def is_string(obj):
        """ Returns true if s is string or string-like object,
        compatible with Python 2 and Python 3."""
        try:
            return isinstance(obj, basestring)
        except NameError:
            return isinstance(obj, str)

    def cvsecs(time):
        """ Will convert any time into seconds.
        Here are the accepted formats:
        >>> cvsecs(15.4) -> 15.4 # seconds
        >>> cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
        >>> cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
        >>> cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
        >>> cvsecs('01:01:33.045') -> 3693.045
        >>> cvsecs('01:01:33,5') #coma works too
        """

        if is_string(time):
            if (',' not in time) and ('.' not in time):
                time = time + '.0'
            expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
            finds = re.findall(expr, time)[0]
            nums = [float(f) for f in finds]
            return (3600 * int(finds[0]) + 60 * int(finds[1]) + int(finds[2]) +
                    nums[3] / (10**len(finds[3])))

        elif isinstance(time, tuple):
            if len(time) == 3:
                hr, mn, sec = time
            elif len(time) == 2:
                hr, mn, sec = 0, time[0], time[1]
            return 3600 * hr + 60 * mn + sec

        else:
            return time

    # to throw error for FileNotFound
    fid = open(filepath)
    fid.close()

    command = [FFMPEG_EXEC, "-i", filepath]

    popen_params = {
        "bufsize": 10**5,
        "stdout": sp.PIPE,
        "stderr": sp.PIPE,
        "stdin": DEVNULL
    }

    if os.name == 'nt':
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(command, **popen_params)
    proc.stdout.readline()
    proc.terminate()
    # Ref: http://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
    infos = proc.stderr.read().decode('ISO-8859-1')
    del proc

    lines = infos.splitlines()

    lines_audio = [l for l in lines if ' Audio: ' in l]
    if lines_audio == []:
        raise RuntimeError(
            "ffmpeg did not find audio in the file %s and produced infos\n%s" %
            (filepath, infos))

    # SAMPLE RATE
    try:
        line = lines_audio[0]
        match = re.search(" [0-9]* Hz", line)
        matched = line[match.start():match.end()]
        samplerate = int(matched[1:-3])
    except:
        raise RuntimeError(
            "Failed to load sample rate of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
            % (filepath, infos))

    # N CHANNELS
    try:
        line = lines_audio[0]
        match1 = re.search(" [0-9]* channels", line)

        if match1 is None:
            match2 = re.search(" stereo", line)
            match3 = re.search(" mono", line)
            if match2 is None and match3 is not None:
                channels = 1
            elif match2 is not None and match3 is None:
                channels = 2
            else:
                raise RuntimeError()
        else:
            channels = int(line[match1.start() + 1:match1.end() - 9])
    except:
        raise RuntimeError(
            "Failed to load n channels of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
            % (filepath, infos))

    # DURATION SECONDS
    try:
        keyword = 'Duration: '
        line = [l for l in lines if keyword in l][0]
        match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])",
                           line)[0]
        duration_seconds = cvsecs(match)
    except:
        raise RuntimeError(
            "Failed to load duration of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
            % (filepath, infos))

    n_samples = int(duration_seconds * samplerate) + 1

    return n_samples, channels, samplerate, duration_seconds


def get_samplerate(filepath):
    """ Get the sample rate of an audio file without reading all of it

    NOTE: Tested only on formats [wav, mp3, mp4], only on macOS
    TODO: Test on Windows. The decoding may eff up for the ffmpeg one

    NOTE: for file formats other than wav, requires FFMPEG installed

    The idea is that getting just the sample rate for the audio in a media file
    should not require reading the entire file.

    There is a native implementation for reading metadata for wav files.

    For other formats, the implementation parses ffmpeg (error) output to get the
    required information.

    # Arguments
        filepath: path to audio file

    # Returns
        samplerate: in Hz
    """

    try:  # if it is a WAV file (most likely)
        _, _, samplerate, _ = read_wavefile_metadata(filepath)
        return samplerate
    except ValueError:
        # Was not a wavefile
        if _ffmpeg_available():
            _, _, samplerate, _ = read_audio_metadata_ffmpeg(filepath)
            return samplerate
        else:
            raise RuntimeError(
                "Neither FFMPEG was found, nor is file %s a valid WAVE file" %
                filepath)
