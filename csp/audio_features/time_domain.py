#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import librosa
import scipy as sp
import numpy as np
import pandas as pd
from time import time
from numpy.fft import rfft
from functools import reduce
from scipy.io import wavfile

from pydub import AudioSegment
from scipy.signal import lfilter
from pydub.utils import mediainfo
from pydub.utils import make_chunks
import python_speech_features as spf
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from pydub import AudioSegment,silence
from scipy.signal import correlate, fftconvolve
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate
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
    cough_s = silence.split_on_silence (signal, 
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

def get_auto_corr_matrix(signal, order):
    ac = librosa.core.autocorrelate(signal)
    R = np.zeros((order, order))
    
    for i in range(order):
        for j in range(order):
            R[i,j] = ac[np.abs(i-j)]
    
    return R, ac[1:(order+1)]


def get_lpc_error(signal, lpc_order):
    ac = librosa.core.autocorrelate(signal)
    lpc = scipy.linalg.solve_toeplitz((ac[:lpc_order], ac[:lpc_order]), ac[1:(lpc_order+1)])
    flipped_lpc = np.flip(lpc)
    
    e = np.zeros_like(signal)
    for j in range(1, signal.shape[0]):
        buff_min = max(0, j-lpc_order)
        e[j] = signal[j] - np.sum(flipped_lpc[(buff_min-j):]*signal[buff_min:j])
    
    return lpc, e


# https://stackoverflow.com/questions/25107806/estimate-formants-using-lpc-in-python/27352810
# scikits.talkbox import lpc is not working anymore, using librosa.core.lpc to calculate lpc
# thank you to Lukasz for help : https://stackoverflow.com/questions/61519826/
# how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu/61528322#61528322
def formant_frequencies(signal, total_formats, order_type, sample_freq, formant_value = None):
    

    """
    In human voice analysis formants are referred as the resonance of the human vocal
    tract. In cough analysis, it is reasonable to expect that the
    resonances of the overall airway that contribute to the
    generation of a cough sound will be represented in the
    formant structure; mucus can change acoustic properties of
    airways. Calculating four formant frequencies (F1,
    F2, F3, F4) in feature set for each frame.

    """
    # https://www.sciencedirect.com/science/article/abs/pii/S0167639301000498
    
    all_frequencies = []
    signal_order_ = {'gautam_method' : formant_value + 2, 'other_method' : int(2 + sample_freq/1000)}

    for sub_frame in signal:
        
        A = librosa.core.lpc(sub_frame, signal_order_[order_type])
        rts = np.roots(A)
        rts = rts[np.imag(rts) >= 0]
        angz = np.arctan2(np.imag(rts), np.real(rts))
        frqs = sorted(angz * sample_freq / (2 *  math.pi))
        all_frequencies.append(frqs[:total_formats])
     
    return all_frequencies
        

                        
# noise generation
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real
                        

# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
def melfrequency_cepstral_coefficients(mfcc_type, 
                                       signal, 
                                       sample_freq, 
                                       frame_size , 
                                       numcep, 
                                       winfunc, 
                                       sound_file = None):
    
    
    """ The mel frequency cepstral coefficients (MFCCs) of a signal are a small 
        set of features (usually about 10–20) 
        which concisely describe the overall shape of a spectral envelope. . 
        mfcc is used to calculate mfccs of a signal.
        
        Resource : http://www.practicalcryptography.com/miscellaneous/
        machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
        
        frame energy is appended to each feature vector. Delta and Delta-Delta features are also appended.
        
    """
    
    if mfcc_type == 'lib':
        y, sr = librosa.load(sound_file)
        mfccs = librosa.feature.mfcc(y, sr = sr)
        #Displaying  the MFCCs
        librosa.display.specshow(mfccs, sr = sr, x_axis='time')
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        return {
                'mfcc' : mfccs, 
                'mfcc_delta' : mfcc_delta, 
                'mfcc_delta2': mfcc_delta2
                } 
    
    else:
        mfcc = spf.mfcc(signal , 
                              sample_freq , 
                              winlen  = frame_size, 
                              winstep = frame_size, 
                              numcep  = numcep,
                              winfunc = winfunc
                             )
        
        mfcc_delta  = spf.delta(mfcc,1)
        mfcc_delta2 = spf.delta(mfcc_delta,1)
        
        return {
                'mfcc' : mfcc, 
                'mfcc_delta' : mfcc_delta, 
                'mfcc_delta2': mfcc_delta2
                }

# https://books.google.co.in/books?id=86RBDwAAQBAJ&pg=PA56&lpg=PA56&dq=zero+crossing+rate+divided+by+length+minus+1&source=bl&ots=SsCSzXx72-&
# sig=ACfU3U1MUEJ-J9fdBizOC3HXky-IeYcH1A&hl=en&sa=X&ved=2ahUKEwiA5Obg95DpAhVcCTQIHVywDy0Q6AEwC3oECAsQAQ#v=onepage&
# q=zero%20crossing%20rate%20divided%20by%20length%20minus%201&f=false
def compute_zcr(signal):

    def change_sign(v1, v2):
        return v1 * v2 < 0
    
    all_frame_values = []
    for frame in signal:
        s = 0
        for ind, _ in enumerate(frame):
            if ind + 1 < len(frame):
                if change_sign(frame[ind], frame[ind+1]):
                    s += 1
        all_frame_values.append(s/(len(frame)-1))
                    
    return all_frame_values  # return zcr for each frame


 
# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0162128&type=printable
# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python

def count_unique(signal):
    # np.bincount not working out for float matrix
    # modifying the bincount for other than int values
    
    if signal.ndim ==1:
        signal = signal
    else:
        signal = signal.flatten()

    uniq_keys = np.unique(signal)
    bins = uniq_keys.searchsorted(signal)
    return uniq_keys, np.bincount(bins)


# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
# todo :  try different methods : shannon
def signal_entropy(signal, entropy_type = math.exp(1)):
    
    """ Computes entropy of signal distribution. """

    signal_size = len(signal)

    if signal_size <= 1:
        return 0

    a,b = count_unique(signal)
    probs = b / signal_size
    n_classes = np.count_nonzero(probs)

    if signal_size <= 1:
        return 
    ent = 0.
    # Compute standard entropy.
    for i in probs:
        ent -= i * math.log(i + 2e-22 , entropy_type)

    return ent


def apply_se(signal):
    
    total_se = []
    for subframe in signal:
        total_se.append(signal_entropy(subframe, entropy_type = math.exp(1)))
        
    return total_se
        

def waveletTransform(lagu):
    cA, cD = pywt.dwt(lagu, 'db1')
    return cD

def signalCorrelate(fullLagu, potonganLagu):
    signalCorrelate = signal.correlate(fullLagu, potonganLagu, mode='full')
    return signalCorrelate

def crossCorrelation(fullLagu, potonganLagu):
    hasilCross = np.correlate(fullLagu, potonganLagu)
    return hasilCross

def magnitude(crossCorrelation):
    magnitude = np.sqrt(crossCorrelation.dot(crossCorrelation))
    return magnitude

# wavelenght denoising of time signals
def denoise_wvlt( x, wavelet, level):
    
    
    """
    1. Adapted from waveletSmooth function found here:
    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/
    2. Threshold equation and using hard mode in threshold as mentioned
    by Tomas Vantuch:
    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    """
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # Calculate sigma for threshold as defined in             http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf
    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation

    sigma = (1/0.6745) * maddest( coeff[-level] )
    # Calculte the univeral threshold
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode='hard') for i in coeff[1:] )  
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet, mode='per' )

  
  
def skew_and_kurtosis(signal_frame):
    
    all_values = {'skew' :  [], 'kurtosis' : [] }
    
    for subframe in signal_frame:
        
        all_values['skew'].append(skew(subframe))
        all_values['kurtosis'].append(kurtosis(subframe))
        
    return {'all_features' : all_values }
 


def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

#Obtain autocorrelation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int((result.size+1)/2):] #Note: other people use re.size/2:, but this does not work for me 
                                   # TODO: check consistency in other computers

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px


def indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    """Peak detection routine borrowed from 
    https://bitbucket.org/lucashnegri/peakutils/src/master/peakutils/peak.py
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks




def freq_from_autocorr_improved(signal, fs):
    signal -= np.mean(signal)  # Remove DC offset
    corr = fftconvolve(signal, signal[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # Find the first peak on the left
    i_peak = indexes(corr, thres=0.8, min_dist=5)[0]
    i_interp = parabolic(corr, i_peak)[0]

    return fs / i_interp, corr, i_interp


def logenergy_computation(signal):
    
    #https://dsp.stackexchange.com/questions/17829/the-logarithm-energy-measure-in-speech
    
    total_energy = []
    for sub_frame in signal:
        constant     = 2e-22
        log_base     = np.log10
        signal_power = sub_frame**2
#         flatten      = np.array(sub_frame).flatten()
        sum_all      = reduce(lambda x, y: (x + y), signal_power)/len(sub_frame) + constant
        total_energy.append(log_base(sum_all))
    return total_energy



def apply_f0(x,fs):
    
    all_values = []
    for subframe in x:
        all_values.append(freq_from_autocorr(subframe,fs))
    return all_values
