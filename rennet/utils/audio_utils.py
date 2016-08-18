"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from pydub import AudioSegment


def get_samplerate(filepath):
    """ Get the sample rate of an audio file

    # Arguments
        filepath: path to audio file
    """
    return AudioSegment.from_file(filepath).frame_rate
