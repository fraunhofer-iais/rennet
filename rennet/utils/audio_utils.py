"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from __future__ import print_function, division
import os
import warnings
from collections import namedtuple

from pydub import AudioSegment

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

FFMPEG_EXEC = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)

AudioMetadata = namedtuple(
    'AudioMetadata',
    [
        'filepath',  # not guaranteed absolute, using user provided
        'format',  # may get converted if not WAV
        'samplerate',
        'nchannels',
        'seconds',  # duration in seconds, may not be exact beyond 1e-2
        'nsamples',  # may not be accurate if not WAV, since being derived from seconds
    ])


def is_ffmpeg_available():
    """ Check if FFMPEG is available """

    envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)

    for envdir in envdir_list:
        possible_path = os.path.join(envdir, FFMPEG_EXEC)
        if os.path.isfile(possible_path) and os.access(possible_path, os.X_OK):
            return True

    # FFMPEG_EXEC was not found
    return False


def is_string(obj):
    """ Returns true if s is string or string-like object,
    compatible with Python 2 and Python 3.

    TODO: [A] move to a general / py3to2 module
    """
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

    TODO: [A] Add tests to test file
    """
    import re
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


def read_wavefile_metadata(filepath):
    """
    # Reference
        https://github.com/scipy/scipy/blob/master/scipy/io/wavfile.py#L116

    TODO: [A] Add documentation
    """
    import struct
    from scipy.io.wavfile import _read_riff_chunk, _read_fmt_chunk, _skip_unknown_chunk

    fid = open(filepath, 'rb')

    def _read_n_samples(fid, big_endian, bits):
        if big_endian:
            fmt = '>i'
        else:
            fmt = '<i'

        size = struct.unpack(fmt, fid.read(4))[0]

        return size // (bits // 8)

    try:
        size, is_big_endian = _read_riff_chunk(fid)

        while fid.tell() < size:
            chunk = fid.read(4)
            if chunk == b'fmt ':
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                channels, samplerate = fmt_chunk[2:4]
                bits = fmt_chunk[6]
            elif chunk == b'fact':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk == b'data':
                n_samples = _read_n_samples(fid, is_big_endian, bits)
                break
            elif chunk == b'LIST':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk in (b'JUNK', b'Fake'):
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                        RuntimeWarning)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        fid.close()

    duration_seconds = (n_samples // channels) / samplerate

    return AudioMetadata(filepath=filepath,
                         format='wav',
                         samplerate=samplerate,
                         nchannels=channels,
                         seconds=duration_seconds,
                         nsamples=n_samples // channels  # per channel nsamples
                         )


def read_audio_metadata_ffmpeg(filepath):
    """
    TODO: [A] Add documentation
    """
    import re
    import subprocess as sp

    def _read_ffmpeg_error_output(filepath):
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

        return infos

    def _read_samplerate(line):
        try:
            match = re.search(" [0-9]* Hz", line)
            matched = line[match.start():match.end()]
            samplerate = int(matched[1:-3])
            return samplerate
        except:
            raise RuntimeError(
                "Failed to load sample rate of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
                % (filepath, infos))

    def _read_n_channels(line):
        try:
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

            return channels
        except:
            raise RuntimeError(
                "Failed to load n channels of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
                % (filepath, infos))

    def _read_duration(line):
        try:
            keyword = 'Duration: '
            line = [l for l in lines if keyword in l][0]
            match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])",
                               line)[0]
            duration_seconds = cvsecs(match)
            return duration_seconds
        except:
            raise RuntimeError(
                "Failed to load duration of file %s from ffmpeg\n the infos from ffmpeg are \n%s"
                % (filepath, infos))

    # to throw error for FileNotFound
    # TODO: [A] test error when FileNotFound
    with open(filepath):
        pass

    infos = _read_ffmpeg_error_output(filepath)
    lines = infos.splitlines()
    lines_audio = [l for l in lines if ' Audio: ' in l]
    if lines_audio == []:
        raise RuntimeError(
            "ffmpeg did not find audio in the file %s and produced infos\n%s" %
            (filepath, infos))

    samplerate = _read_samplerate(lines_audio[0])
    channels = _read_n_channels(lines_audio[0])
    duration_seconds = _read_duration(lines)

    n_samples = int(duration_seconds * samplerate) + 1

    warnings.warn(
        "Metadata was read from FFMPEG, duration and number of samples may not be accurate",
        RuntimeWarning)

    return AudioMetadata(
        filepath=filepath,
        format=os.path.splitext(filepath)[1][1:],  # extension after the dot
        samplerate=samplerate,
        nchannels=channels,
        seconds=duration_seconds,
        nsamples=n_samples
    )


def get_audio_metadata(filepath):
    """ Get the metadat for an audio file without reading all of it

    NOTE: Tested only on formats [wav, mp3, mp4, avi], only on macOS
    TODO: [A] Test on Windows. The decoding may eff up for the ffmpeg one

    NOTE: for file formats other than wav, requires FFMPEG installed

    The idea is that getting just the sample rate for the audio in a media file
    should not require reading the entire file.

    The implementation for reading metadata for wav files REQUIRES scipy

    For other formats, the implementation parses ffmpeg (error) output to get the
    required information.

    # Arguments
        filepath: path to audio file

    # Returns
        samplerate: in Hz
    """

    try:  # if it is a WAV file (most likely)
        return read_wavefile_metadata(filepath)
    except ValueError:
        # Was not a wavefile
        if is_ffmpeg_available():
            return read_audio_metadata_ffmpeg(filepath)
        else:
            raise RuntimeError(
                "Neither FFMPEG was found, nor is file %s a valid WAVE file" %
                filepath)


class AudioIO(AudioSegment):
    """ A extension of the pydub.AudioSegment class with some added methods"""

    @classmethod
    def from_audiometadata(cls, audiometadata):
        """ classmethod to create new AudioIO object from AudioMetadata, PLUS
            update the metadata

        The file will most likely be read for this.
        The file may also be temporarily converted to WAV file, if not originally so.

        Refer to pydub.AudioSegment docuemntation for further reference

        The updated metadata will be from the read file, hence expected to be
        correct.
        # Arguments
            audiometadata: AudioMetadata instance

        # Returns
            obj: instance of AudioIO (pydub.AudioSegment)
            updated_metadata: Updated metadata for the read file
        """
        obj = cls.from_file(audiometadata.filepath)

        nframes = obj.frame_count()
        if nframes != int(nframes):
            warnings.warn(
                "Frame Count is calculated as float = {} by pydub".format(
                    nframes), RuntimeWarning)

        updated_metadata = AudioMetadata(filepath=audiometadata.filepath,
                                         format=audiometadata.format,
                                         samplerate=obj.frame_rate,
                                         nchannels=obj.channels,
                                         seconds=obj.duration_seconds,
                                         nsamples=int(nframes))

        return obj, updated_metadata

    def get_numpy_data(self):
        """ Get the raw data as numpy array

        # Returns
            data: numpy array of shape (nsamples x nchannels)
        """
        from numpy import array as nparr

        data = self.get_array_of_samples()
        nchannels = self.channels

        return nparr([data[i::nchannels] for i in range(nchannels)]).T
