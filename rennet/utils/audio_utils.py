"""
@mojuste
Created: 18-08-2016

Utilities for working with audio
"""
from pydub import AudioSegment

def get_samplerate_wav(wav_filepath):
    return AudioSegment.from_wav(wav_filepath).frame_rate
