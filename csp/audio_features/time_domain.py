#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from numpy import ndarray
from librosa.feature import zero_crossing_rate as librosa_zcr

__docformat__ = 'reStructuredText'
__all__ = ['zero_crossing_rate']


def zero_crossing_rate(audio_data: ndarray,
                       frame_length: Optional[int] = 2048,
                       hop_length: Optional[int] = 512,
                       center: Optional[bool] = True,
                       thr: Optional[float] = 1e-10,
                       pad: Optional[bool] = True,
                       zero_pos: Optional[bool] = True,
                       axis: Optional[int] = -1) \
        -> ndarray:
    """Calculates zero crossing rate for each frame.

    :param audio_data: Audio data to be used.
    :type audio_data: numpy.ndarray
    :param frame_length: Frame length to use (in samples), \
                         defaults to 2058.
    :type frame_length: int, optional
    :param hop_length: Hop length between frames (in samples), \
                       default to 512.
    :type hop_length: int, optional
    :param center: Center the frames? Defaults to True
    :type center: bool, optional
    :param thr: Threshold to use for considering value as zero, \
                defaults to 1e-10.
    :type thr: float, optional
    :param pad: Is 0th frame as valid zero-crossing? Defaults to True.
    :type pad: bool, optional
    :param zero_pos: Is 0 a positive number? Defaults to True.
    :type zero_pos: bool, optional
    :param axis: Which axis to use? Defaults to -1.
    :type axis: int, optional
    :return: Zero-crossing rate per frame, shape (1, nb_frames).
    :rtype: numpy.ndarrau
    """
    return librosa_zcr(
        y=audio_data,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
        threshold=thr,
        pad=pad,
        zero_pos=zero_pos,
        axis=axis)

# EOF
