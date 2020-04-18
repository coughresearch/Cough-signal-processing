#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union
from functools import partial

from numpy import ndarray, expand_dims, \
    concatenate as np_cat, abs as np_abs, \
    dot as np_dot
from librosa.core import stft as librosa_stft
from librosa.filters import mel

__docformat__ = 'reStructuredText'
__all__ = ['fourier_transform',
           'magnitude_spectrogram',
           'power_spectrogram',
           'mel_scaled_spectrogram']


def fourier_transform(audio_data: ndarray,
                      n_fft: Optional[Union[int, None]] = None,
                      hop_length: Union[Optional[int, None]] = None,
                      win_length: Union[Optional[float, None]] = None,
                      window: Optional[str] = 'hann',
                      center: Optional[bool] = True) \
        -> ndarray:
    """Calculates and returns the complex Fourier transform (FT) of \
    the audio signal.

    If short-time Fourier transform is needed, then the amount \
    of FT points have to be set accordingly.

    :param audio_data: Audio data to be used in shape (channels, data).
    :type audio_data: numpy.ndarray
    :param n_fft: Amount of FT points. If `None`, then FT is \
                  calculated on the whole signal. Defaults to \
                  `None`.
    :type n_fft: int|None, optional
    :param hop_length: Amount of samples to be skipped for \
                       consequent windows. If `None`, then \
                       hop length is equal to `n_fft/4`. \
                       Defaults to `None`.
    :type hop_length: int|None, optional
    :param win_length: Window length to be used for each frame. \
                       If `win_length` > `n_fft`, frames are padded \
                       with zeros. If `win_length` < `n_fft`, frames \
                       are trimmed. If `win_length` is `None`, then
                       it is set to `n_fft`. Defaults to `None`.
    :type win_length: int|None, optional
    :param window: Windowing function to be used. Supported types \
                   are according to librosa. Defaults to `hann`.
    :type window: str, optional
    :param center: Center frames, defaults to True.
    :type center: bool, optional
    :return: The complex frequency transform (FT or STFT) of the signal, \
             shape = (channels, 1 + n_fft/2, n_frames), where \
             `n_frames` i the resulting number of frames for channels >=2,
             else shape = (1 + n_fft/2, n_frames).
    :rtype: numpy.ndarray
    """
    stft_partial = partial(
        librosa_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center)

    if audio_data.ndim == 1:
        return stft_partial(audio_data)
    else:
        return np_cat([expand_dims(stft_partial(i), 0)
                       for i in audio_data], axis=0)


def magnitude_spectrogram(audio_data: ndarray,
                          n_fft: Optional[Union[int, None]] = None,
                          hop_length: Union[Optional[int, None]] = None,
                          win_length: Union[Optional[float, None]] = None,
                          window: Optional[str] = 'hann',
                          center: Optional[bool] = True) \
        -> ndarray:
    """Calculates and returns the magnitude spectrogram of \
    the audio signal, based on the FT.

    :param audio_data: Audio data to be used in shape (channels, data).
    :type audio_data: numpy.ndarray
    :param n_fft: Amount of FT points. If `None`, then FT is \
                  calculated on the whole signal. Defaults to \
                  `None`.
    :type n_fft: int|None, optional
    :param hop_length: Amount of samples to be skipped for \
                       consequent windows. If `None`, then \
                       hop length is equal to `n_fft/4`. \
                       Defaults to `None`.
    :type hop_length: int|None, optional
    :param win_length: Window length to be used for each frame. \
                       If `win_length` > `n_fft`, frames are padded \
                       with zeros. If `win_length` < `n_fft`, frames \
                       are trimmed. If `win_length` is `None`, then
                       it is set to `n_fft`. Defaults to `None`.
    :type win_length: int|None, optional
    :param window: Windowing function to be used. Supported types \
                   are according to librosa. Defaults to `hann`.
    :type window: str, optional
    :param center: Center frames, defaults to True.
    :type center: bool, optional
    :return: The magnitude spectrogram of the signal, \
             shape = (channels, 1 + n_fft/2, n_frames), where \
             `n_frames` i the resulting number of frames for channels >=2,
             else shape = (1 + n_fft/2, n_frames).
    :rtype: numpy.ndarray
    """
    return np_abs(fourier_transform(
        audio_data=audio_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center))


def power_spectrogram(audio_data: ndarray,
                      n_fft: Optional[Union[int, None]] = None,
                      hop_length: Union[Optional[int, None]] = None,
                      win_length: Union[Optional[float, None]] = None,
                      window: Optional[str] = 'hann',
                      center: Optional[bool] = True) \
        -> ndarray:
    """Calculates and returns the power spectrogram of \
    the audio signal, based on the FT.

    :param audio_data: Audio data to be used in shape (channels, data).
    :type audio_data: numpy.ndarray
    :param n_fft: Amount of FT points. If `None`, then FT is \
                  calculated on the whole signal. Defaults to \
                  `None`.
    :type n_fft: int|None, optional
    :param hop_length: Amount of samples to be skipped for \
                       consequent windows. If `None`, then \
                       hop length is equal to `n_fft/4`. \
                       Defaults to `None`.
    :type hop_length: int|None, optional
    :param win_length: Window length to be used for each frame. \
                       If `win_length` > `n_fft`, frames are padded \
                       with zeros. If `win_length` < `n_fft`, frames \
                       are trimmed. If `win_length` is `None`, then
                       it is set to `n_fft`. Defaults to `None`.
    :type win_length: int|None, optional
    :param window: Windowing function to be used. Supported types \
                   are according to librosa. Defaults to `hann`.
    :type window: str, optional
    :param center: Center frames, defaults to True.
    :type center: bool, optional
    :return: The power spectrogram of the signal, \
             shape = (channels, 1 + n_fft/2, n_frames), where \
             `n_frames` i the resulting number of frames for channels >=2,
             else shape = (1 + n_fft/2, n_frames).
    :rtype: numpy.ndarray
    """
    return magnitude_spectrogram(
        audio_data=audio_data,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center) ** 2


def mel_scaled_spectrogram(spectrogram: ndarray,
                           sr: int,
                           n_mels: Optional[int] = 128,
                           fmin: Optional[float] = 0.0,
                           fmax: Optional[Union[float, None]] = None,
                           htk: Optional[bool] = False):
    """Calculates the mel scaled version of the spectrogram.

    :param spectrogram: Spectrogram to be used.
    :type spectrogram: numpy.ndarray
    :param sr: Sampling frequency of the original signal.
    :type sr: int
    :param n_mels: Amount of mel filters to use, defaults to 128.
    :type n_mels: int, optional
    :param fmin: Minimum frequency for mel filters, defaults to 0.0.
    :type fmin: float, optional
    :param fmax: Maximum frequency for mel filters. If `None`, \
                 sr/2.0 is used. Defaults to None
    :type fmax: float|None, optional
    :param htk: Use HTK formula, instead of Slaney, defaults to False.
    :type htk: bool, optional
    :return: Mel scaled version of the input spectrogram, with shape \
             (channels, nb_mels, values) for channels >= 2, else \
             (nb_mels, values).
    :rtype: numpy.ndarray
    """
    ndim = spectrogram.ndim

    if ndim not in [2, 3]:
        raise AttributeError('Input spectrogram must be of shape '
                             '(channels, nb_frames, frames). '
                             f'Current input has {ndim} dimensions. '
                             f'Allowed are either 2 or 3.')

    n_fft = 2 * (spectrogram[ndim - 2] - 1)

    mel_filters = mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        htk=htk)

    if ndim == 2:
        mel_spectrogram = np_dot(mel_filters, spectrogram)
    else:
        mel_spectrogram = np_cat([expand_dims(np_dot(mel_filters, i), 0)
                                  for i in spectrogram], axis=0)

    return mel_spectrogram


# EOF
