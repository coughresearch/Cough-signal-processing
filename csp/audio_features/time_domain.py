#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional


from numpy import ndarray
from librosa.feature import zero_crossing_rate as librosa_zcr

from pydub import AudioSegment,silence
from pydub.utils import make_chunks
import scipy as sp

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



# https://stackoverflow.com/questions/33720395/can-pydub-set-the-maximum-minimum-volume
# set a value for maximum/minimum volume, that is, 
# there won't be too loud or too quiet in output audio file
def set_loudness(sound, target_dBFS):
    loudness_difference = target_dBFS - sound.dBFS
    return sound.apply_gain(loudness_difference)


# detect silence 
def detect_silence(cough_wav, min_silence_len, silence_thresh):
    cough_sound = AudioSegment.from_wav(cough_wav)
    silence     = silence.detect_silence(cough_sound, min_silence_len = min_silence_len, 
                                         silence_thresh = silence_thresh)
    silence = [((start/1000),(stop/1000)) for start,stop in silence] #convert to sec
    return silence


# sound slice normalization 
def sound_slice_normalize(sound, sample_rate, target_dBFS):
    def max_min_volume(min, max):
        for chunk in make_chunks(sound, sample_rate):
            if chunk.dBFS < min:
                yield match_target_amplitude(chunk, min)
            elif chunk.dBFS > max:
                yield match_target_amplitude(chunk, max)
            else:
                yield chunk

    return reduce(lambda x, y: x + y, max_min_volume(target_dBFS[0], target_dBFS[1]))



# get loudness of cough sounds
def get_loudness(sound, slice_size=60*1000):
    return max(chunk.dBFS for chunk in make_chunks(sound, slice_size))



# Segmentation
def cough_signal_segmentation(signal, min_silence_len, silence_thresh):
    
    # decide min_silence_len from all cough_audio
    cough_s = split_on_silence (signal, 
                                min_silence_len = min_silence_len, 
                                silence_thresh = signal.dBFS-silence_thresh
                               )
    signal_array = [np.array(coughs_s.get_array_of_samples()) for coughs_s in cough_s]
    return signal_array


# Filter data along one-dimension with an IIR or FIR filter
def pre_Emph_filter(signal, value, filter_type = 'fil'):
    
    # value = [1., -0.96]
    # a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
    #                         - a[1]*y[n-1] - ... - a[na]*y[n-na]
    if filter_type == 'fil':
        filter_s = sp.signal.filtfilt
    else:
        filter_s = sp.signal.lfilter
    
    # apply filter on each segments of the cough sounds.
    filtered_singal = [filter_s(value, 1, segment) for segment in signal]
    return filtered_singal


def cough_singal_preprocessing(cough_signal, 
                              sample_frequency, 
                              target_dBFS,
                              min_silence_len, 
                              silence_thresh,
                              value, 
                              filter_type , func_log = False):
    
    # All continuous analog signal is sampled at a frequency F
    sample_freq  = cough_signal.set_frame_rate(sample_frequency)
    
    # we can take 1 minute slices
    loudnes_value = get_loudness(sample_freq, slice_size=60*1000)
    if func_log:
        print("loudness", loudnes_value)
    loudness     = set_loudness(sample_freq, target_dBFS)
    #  get_silence  = detect_silence(cough_audio, min_silence_len, silence_thresh)
    if func_log:
        print("silence", get_silence)
    segmentation = cough_signal_segmentation(loudness,min_silence_len, silence_thresh)
    if func_log:
        print("segmentation", get_silence)
    pre_Emph_fi  = pre_Emph_filter(segmentation, value, filter_type = filter_type)
    return pre_Emph_fi
