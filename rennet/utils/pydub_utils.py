"""
@motjuste
Created: 11-10-2017

Utilities for audio-io and conversions using Pydub.
Separated from `audio_utils` to remove dependency, I guess.
"""
from __future__ import print_function, division, absolute_import
from six.moves import range
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import subprocess as sp
import os
import warnings

from .audio_utils import (
    AudioMetadata,
    get_audio_metadata,
    get_sph2pipe,
)


class AudioIO(AudioSegment):
    """ A extension of the pydub.AudioSegment class with some added methods"""

    @classmethod
    def from_file(cls, file, format=None, **kwargs):  # pylint: disable=redefined-builtin
        meta = get_audio_metadata(file)

        if meta.format == 'sph' or format == 'sph':
            output = NamedTemporaryFile(mode='rb', delete=False)

            # check if sph2pipe is provided or else, is it available on path
            converterpath = kwargs.get("sph2pipe_path", get_sph2pipe())

            if not converterpath:
                _e = (
                    "sph2pipe was not found on PATH."
                    "\n\nPlease install the `sph2pipe_v2.5` tool for converting sph files, it can be found at:\n"
                    "https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools"
                    "\n\nPlease make sure it is available on PATH after installation"
                    "\nOR, provide fullpath as `AudioIO.from_file(file, format='sph', sph2pipe_path=SPH2PIPE)``"
                )
                raise RuntimeError(_e)

            conversion_command = [
                converterpath,
                "-p",  # force into PCM linear
                "-f",
                "riff",  # export with header for WAV
                meta.filepath,  # input abspath
                output.name  # output filepath
            ]

            p = sp.Popen(conversion_command, stdout=sp.PIPE, stderr=sp.PIPE)
            _, p_err = p.communicate()

            if p.returncode != 0:
                raise RuntimeError(
                    "Converting sph to wav failed with:\n{}\n{}".format(
                        p.returncode, p_err
                    )
                )

            obj = cls._from_safe_wav(output)

            output.close()
            os.unlink(output.name)

            return obj
        else:
            return super(AudioIO, cls).from_file(file, format, **kwargs)

    @classmethod
    def from_sph(cls, file_path):
        return cls.from_file(file_path, format='sph')

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
                "Frame Count is calculated as float = {} by pydub".format(nframes),
                RuntimeWarning
            )

        updated_metadata = AudioMetadata(
            filepath=audiometadata.filepath,
            format=audiometadata.format,
            samplerate=obj.frame_rate,
            nchannels=obj.channels,
            seconds=obj.duration_seconds,
            nsamples=int(nframes)
        )

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

    def export_standard(self, outfilepath, samplerate=16000, channels=1, fmt="wav"):

        channeled = self.set_channels(channels)
        framed = channeled.set_frame_rate(samplerate)
        return framed.export(outfilepath, format=fmt)


def convert_to_standard(
    filepath, todir, tofmt="wav", samplerate=16000, channels=1, **kwargs
):
    """ Convert a single media file to the standard format """
    tofilename = os.path.splitext(os.path.basename(filepath))[0] + "." + tofmt
    tofilepath = os.path.join(todir, tofilename)
    s = AudioIO.from_file(filepath, **kwargs)
    f = s.export_standard(  # pylint: disable=no-member
        tofilepath,
        samplerate=samplerate,
        channels=channels,
        fmt=tofmt)
    f.close()
    return [
        tofilename,
    ]


def convert_to_standard_split(filepath, todir, tofmt="wav", samplerate=16000, **kwargs):
    s = AudioIO.from_file(filepath, **kwargs)
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
