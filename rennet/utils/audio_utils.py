"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from __future__ import print_function, division
import os
import warnings
from collections import namedtuple

from tempfile import NamedTemporaryFile
import subprocess as sp

from pydub import AudioSegment  # external dependency

from rennet.utils.py_utils import cvsecs

try:
    from subprocess import DEVNULL
except ImportError:
    DEVNULL = open(os.devnull, 'wb')  # FIXME: Okay to never close this?

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


def which(executable):
    """ Check if executable is available on the system

    Works with Unix type systems and Windows
    NOTE: no changes are made to the executable's name, only path is added

    # Arguments
        executable: str name of the executable (with `.exe` for Windows)

    # Returns:
        False or executable: depending on if the executable is accessible
    """

    envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)

    for envdir in envdir_list:
        possible_path = os.path.join(envdir, executable)
        if os.path.isfile(possible_path) and os.access(possible_path, os.X_OK):
            return executable

    # executable was not found
    return False  # Sorry for type-stability


def get_codec():
    """ Get codec to use for the audio file

    Searches for existence of `FFMPEG` first, then `AVCONV`
    NOTE: `.exe` is appended to name if on Windows

    # Returns
        False or executable: bool or str : executable of the codec if available
    """
    if os.name == "nt":
        return which("ffmpeg.exe")
    else:
        ffmpeg = which("ffmpeg")
        if not ffmpeg:
            return which("avconv")
        else:
            return ffmpeg


CODEC_EXEC = get_codec()  # NOTE: Available codec; False when none available


def read_wavefile_metadata(filepath):
    """ Read AudioMetadata of a WAV file without reading all of it

    Heavily depends on `scipy.io.wavfile`.

    # Arguments
        filepath: str: full path to the WAV file

    # Returns
        meta: AudioMetadata object (namedtuple): with information as:

    # Reference
        https://github.com/scipy/scipy/blob/master/scipy/io/wavfile.py#L116

    """
    import struct
    from scipy.io.wavfile import _read_riff_chunk, _read_fmt_chunk
    from scipy.io.wavfile import _skip_unknown_chunk

    fid = open(filepath, 'rb')

    def _read_n_samples(fid, big_endian, bits):
        if big_endian:
            fmt = '>i'
        else:
            fmt = '<i'

        size = struct.unpack(fmt, fid.read(4))[0]

        return size // (bits // 8)  # indicates total number of samples

    try:
        size, is_big_endian = _read_riff_chunk(fid)

        while fid.tell() < size:
            chunk = fid.read(4)
            if chunk == b'fmt ':
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                channels, samplerate = fmt_chunk[2:4]  # info relevant to us
                bits = fmt_chunk[6]
            elif chunk == b'data':
                n_samples = _read_n_samples(fid, is_big_endian, bits)
                break  # NOTE: break as now we have all info we need
            elif chunk in (b'JUNK', b'Fake', b'LIST', b'fact'):
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn("Chunk (non-data) not understood, skipping it.",
                              RuntimeWarning)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:  # always close
        fid.close()

    return AudioMetadata(
        filepath=filepath,
        format='wav',
        samplerate=samplerate,
        nchannels=channels,
        seconds=(n_samples // channels) / samplerate,
        nsamples=n_samples // channels  # for one channel
    )


def read_sph_metadata(filepath):
    """
    TODO: [ ] Add documentation
    NOTE: Tested and developed specifically for the Fisher Dataset
    """
    filepath = os.path.abspath(filepath)
    fid = open(filepath, 'rb')

    try:
        # HACK: Going to read the header that is supposed to stop at
        # 'end_header'. If it is not found, I stop at 100 readlines anyway

        # First line gives the header type
        fid.seek(0)
        assert fid.readline().startswith(
            b'NIST'), "Unrecognized Sphere Header type"

        # The second line tells the header size
        _header_size = int(fid.readline().strip())

        # read the header lines based on the _header_size
        fid.seek(0)
        # Each info is on different lines (per dox)
        readlines = fid.read(_header_size).split(b'\n')

        # Start reading relevant metadata
        nsamples = None
        nchannels = None
        samplerate = None

        for line in readlines:
            splitline = line.split(b' ')
            info, data = splitline[0], splitline[-1]

            if info == b'sample_count':
                nsamples = int(data)
            elif info == b'channel_count':
                nchannels = int(data)
            elif info == b'sample_rate':
                samplerate = int(data)
            else:
                continue
    finally:
        fid.close()

    if any(x is None for x in [nsamples, nchannels, samplerate]):
        raise RuntimeError(
            "The Sphere header was read, but some information was missing")
    else:
        return AudioMetadata(
            filepath=filepath,
            format='sph',
            samplerate=samplerate,
            nchannels=nchannels,
            seconds=nsamples / samplerate,
            nsamples=nsamples)


def read_audio_metadata_codec(filepath):
    """
    TODO: [A] Add documentation
    """
    import re
    import subprocess as sp

    def _read_codec_error_output(filepath):
        command = [CODEC_EXEC, "-i", filepath]

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
                "Failed to load sample rate of file %s from %s\n the infos from %s are \n%s"
                % (filepath, CODEC_EXEC, CODEC_EXEC, infos))

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
                "Failed to load n channels of file %s from %s\n the infos from %s are \n%s"
                % (filepath, CODEC_EXEC, CODEC_EXEC, infos))

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
                "Failed to load duration of file %s from %s\n the infos from %s are \n%s"
                % (filepath, CODEC_EXEC, CODEC_EXEC, infos))

    # to throw error for FileNotFound
    # TODO: [A] test error when FileNotFound
    with open(filepath):
        pass

    if not get_codec():
        raise RuntimeError("No codec available")

    infos = _read_codec_error_output(filepath)
    lines = infos.splitlines()
    lines_audio = [l for l in lines if ' Audio: ' in l]
    if lines_audio == []:
        raise RuntimeError(
            "%s did not find audio in the file %s and produced infos\n%s" %
            (CODEC_EXEC, filepath, infos))

    samplerate = _read_samplerate(lines_audio[0])
    channels = _read_n_channels(lines_audio[0])
    duration_seconds = _read_duration(lines)

    n_samples = int(duration_seconds * samplerate) + 1

    warnings.warn(
        "Metadata was read from %s, duration and number of samples may not be accurate"
        % CODEC_EXEC, RuntimeWarning)

    return AudioMetadata(
        filepath=filepath,
        format=os.path.splitext(filepath)[1][1:],  # extension after the dot
        samplerate=samplerate,
        nchannels=channels,
        seconds=duration_seconds,
        nsamples=n_samples)


def get_audio_metadata(filepath):
    """ Get the metadata for an audio file without reading all of it

    NOTE: Tested only on formats [wav, mp3, mp4, avi], only on macOS

    NOTE: for file formats other than wav, requires FFMPEG or AVCONV installed

    The idea is that getting just the sample rate for the audio in a media file
    should not require reading the entire file.

    The implementation for reading metadata for wav files REQUIRES scipy

    For other formats, the implementation parses ffmpeg or avconv (error) output to get the
    required information.

    # Arguments
        filepath: path to audio file

    # Returns
        samplerate: in Hz
    """

    # TODO: [ ] Do better reading of audiometadata
    try:
        # possibly a sphere file
        if filepath.endswith('sph'):
            return read_sph_metadata(filepath)
        else:  # if it is a WAV file (most likely)
            return read_wavefile_metadata(filepath)
    except ValueError:
        # Was not a wavefile
        if get_codec():
            return read_audio_metadata_codec(filepath)
        else:
            raise RuntimeError(
                "Neither FFMPEG or AVCONV was found, nor is file %s a valid WAVE file"
                % filepath)


class AudioIO(AudioSegment):
    """ A extension of the pydub.AudioSegment class with some added methods"""

    @classmethod
    def from_file(cls, file, format=None, **kwargs):
        meta = get_audio_metadata(file)

        if meta.format == 'sph' or format == 'sph':
            output = NamedTemporaryFile(mode='rb', delete=False)

            # TODO: move the sph2pipe files to the utils folder
            # TODO: automatically compile the sph2pipe
            # TODO: does the executable have to be marked as so using chmod?
            conversion_command = [
                # "$RENNET_ROOT/rennet/utils/sph2pipe/sph2pipe",
                "sph2pipe",
                "-p",  # force into PCM linear
                "-f", "riff",  # export with header for WAV
                meta.filepath,  # input abspath
                output.name  # output filepath
            ]

            p = sp.Popen(conversion_command, stdout=sp.PIPE, stderr=sp.PIPE)
            p_out, p_err = p.communicate()

            if p.returncode != 0:
                raise RuntimeError("Converting sph to wav failed with:\n{}\n{}".format(p.returncode, p_err))

            obj = cls._from_safe_wav(output)

            output.close()
            os.unlink(output.name)

            return obj
        else:
            super().from_file(file, format, **kwargs)

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

        updated_metadata = AudioMetadata(
            filepath=audiometadata.filepath,
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

    def export_standard(self,
                        outfilepath,
                        samplerate=16000,
                        channels=1,
                        fmt="wav"):

        channeled = self.set_channels(channels)
        framed = channeled.set_frame_rate(samplerate)
        return framed.export(outfilepath, format=fmt)


def convert_to_standard(filepath,
                        todir,
                        tofmt="wav",
                        samplerate=16000,
                        channels=1):
    """ Convert a single media file to the standard format """
    tofilename = os.path.splitext(os.path.basename(filepath))[0] + "." + tofmt
    tofilepath = os.path.join(todir, tofilename)
    s = AudioIO.from_file(filepath)
    f = s.export_standard(
        tofilepath, samplerate=samplerate, channels=channels, fmt=tofmt)
    f.close()
    return [tofilename, ]


def convert_to_standard_split(filepath, todir, tofmt="wav", samplerate=16000):
    s = AudioIO.from_file(filepath)
    s = s.set_frame_rate(samplerate)

    splits = s.split_to_mono()

    tofilename = os.path.splitext(os.path.basename(filepath))[0]
    tofilenames = []
    for i, split in enumerate(splits):
        _tofilename = tofilename + ".c{}.".format(i) + tofmt
        tofilepath = os.path.join(todir, _tofilename)
        tofilenames.append(_tofilename)

        f = split.export(tofilepath, format=tofmt)
        f.close()

    return tofilenames
