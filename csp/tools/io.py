#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Optional
from pathlib import Path

from numpy import ndarray
from librosa import load as librosa_load

__docformat__ = 'reStructuredText'
__all__ = ['load_audio_file']


def load_audio_file(file_path: Path,
                    sr: int,
                    mono: Optional[bool] = True,
                    offset: Optional[float] = 0,
                    duration: Optional[Union[float, None]] = None) \
        -> ndarray:
    """Wrapper for loading audio file.

    TODO: Add handling of cases where librosa fails.

    :param file_path: File path of the audio file.
    :type file_path: pathlib.Path
    :param sr: Sampling frequency to be used. If different\
               from actual, data are resampled.
    :type sr: int
    :param mono: Load file as mono? Defaults to True
    :type mono: bool, optional
    :param offset: Offset for reading the file. Default to 0.0
    :type offset: float, optional
    :param duration: Duration of the returned data. Defaults to None.
    :type duration: float|None, optional
    :return: Audio data as numpy array of shape (channels x samples), \
             if channels >= 2, else (samples, )
    :rtype: numpy.ndarray
    """
    return librosa_load(
        path=str(file_path),
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration)[0]

# EOF
