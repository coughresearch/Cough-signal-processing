import modin.pandas as modin
from csp import compute_zcr,formant_frequencies,logenergy_computation, \
get_f0, freq_from_autocorr_improved, freq_from_autocorr,skew_and_kurtosis, \
apply_se,apply_f0,melfrequency_cepstral_coefficients
from csp import cough_singal_preprocessing
import python_speech_features as spe_feats


import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import mediainfo
import os

def col_generator(feature_name, total_col):
    col_names = [str(feature_name) + '_feature_' + str(col_na) for col_na in range(total_col)]
    return col_names

def arrange_features(features_list, total_col, feature_type):
    
    features = []
    
    if feature_type == 'mfcc':
        
        mfcc_fetures               = modin.DataFrame(features_list['mfcc'])
        mfcc_fetures.columns       = col_generator('mfcc_features_', total_col)
                
        mfcc_delta_fetures         = modin.DataFrame(features_list['mfcc_delta'])
        mfcc_delta_fetures.columns = col_generator('mfcc_delta', total_col)
        
        mfcc_dd_fetures            = modin.DataFrame(features_list['mfcc_delta2'])
        mfcc_dd_fetures.columns    = col_generator('mfcc_dd_fetures', total_col)
        
        features.extend([mfcc_fetures,mfcc_delta_fetures,mfcc_dd_fetures])
        features = modin.concat(features, axis=1)
        
    else:
        features         = modin.DataFrame(features_list)
        features.columns = col_generator('formant_features_', total_col)
        
        
    
    return features

def cough_features_extraction( subframe, 
                              frame_length, 
                              sliding_window, 
                              mfcc_type, 
                              sample_frequency, 
                              frame_size_ms, 
                              numcep, 
                              total_formants, 
                              filter_order, 
                              formant_value, 
                              features_list = ['default']):
    
    
    if features_list[0]!= 'default':
        features_list = features_list
    else:
        features_list = ['mfcc_features', 
                         'mfcc_delta',
                         'mfcc_delta_delta', 
                         'zero_crossing_rate', 
                         'formant_feq', 
                         'log_energy', 
                         'entropy', 
                         'bispectrum Score', 
                         'skew', 
                         'kurtosis', 
                         'continuous_wavelet_transform',
                         'pitch']
    
    
    
#     features_ =   {'mfcc_features'                : melfrequency_cepstral_coefficients, 
#                     'zero_crossing_rate'           : compute_zcr, 
#                     'formant_feq'                  : formant_frequencies, 
#                     'log_energy'                   : logenergy_computation, 
#                     'entropy'                      : apply_se, 
#                     'bispectrum Score'             : bispectrum Score, 
#                     'skew'                         : skew_and_kurtosis, 
#                     'continuous_wavelet_transform' : continuous_wavelet_transform, 
#                     'pitch'                        : freq_from_autocorr
#                    }
    
    
    all_features = {}
    
    
    # mfcc
    segmentation_frames          = spe_feats.sigproc.framesig(subframe, frame_length, frame_length, sliding_window)
    mfcc_features                = melfrequency_cepstral_coefficients('pyspeech',subframe,sample_frequency ,frame_size_ms , numcep, sliding_window)
    
    arrange_feat                 = arrange_features(mfcc_features,13,'mfcc')
    
    # formant feq
    formant_features             = formant_frequencies(signal = segmentation_frames, 
                                                       total_formats = total_formants , 
                                                       order_type = filter_order , 
                                                       sample_freq = sample_frequency, 
                                                       formant_value = formant_value)
    formant_arrange_feat         = arrange_features(formant_features,4 ,'f0')
    # log energy
    all_features['log_energy']   = logenergy_computation(segmentation_frames)
    all_features.update(skew_and_kurtosis(segmentation_frames)['all_features'])
    
    
    all_features['zcr']         = compute_zcr(segmentation_frames)
    all_features['entropy']     = apply_se(segmentation_frames)
    all_features['pitch']       = apply_f0(segmentation_frames,sample_frequency)
    all_feat                    = modin.DataFrame(all_features)
    concat_all                  = modin.concat([arrange_feat,formant_arrange_feat,all_feat],axis=1)
    
    return concat_all


def singal_preprocessing(config_dict, audio_file_frame):
    
    # config file 
    
    
    audio_id     = list(audio_file_frame['cough_id'])
    audio_name   = list(audio_file_frame['file_name'])
    labels       = list(audio_file_frame['labels'])
    folder_path  = config_dict['folder_path']
    frame_length = int(config_dict['sample_frequency'] * config_dict['frame_size_ms'])

    
    features = []
    
    # opening the file
    

    for file_id, name, label in tqdm(zip(audio_id,audio_name,labels)):
        audio_file_path = folder_path + '/' + name
        open_           = AudioSegment.from_wav(audio_file_path)
        sample_rate     = float(mediainfo(audio_file_path)['sample_rate'])
        
        # ----------------------------------------------------------------
        
        
        # preprocessing
        set_s = cough_singal_preprocessing(cough_signal      = open_, 
                                           sample_frequency  = config_dict['sample_frequency'], 
                                           target_fre        = config_dict['target_fre'], 
                                           silence_length    = config_dict['min_silence_len'],
                                           silence_threshold = config_dict['silence_thresh'], 
                                           value             = config_dict['value'],
                                           filter_type       = 'fa')
        
        # -------------------------------------------------------------------
        # feature_extraction
        
        
        all_values = []

        for audio_frame in set_s:
            new_values = cough_features_extraction(subframe = audio_frame, 
                                      frame_length = frame_length, 
                                      sliding_window = config_dict['sliding_window'], 
                                      mfcc_type = config_dict['sliding_window'], 
                                      sample_frequency = config_dict['sample_frequency'], 
                                      frame_size_ms = config_dict['frame_size_ms'], 
                                      numcep = config_dict['numcep'], 
                                      total_formants = config_dict['total_formants'], 
                                      filter_order = config_dict['filter_order'], 
                                      formant_value = config_dict['formant_value'], 
                                      features_list= ['default'])
            all_values.append(new_values)

        feat = modin.concat(all_values)
        
        a_ids    = []
        a_labels = []
        
        for fill_ in range(len(feat)):
            a_ids.append(file_id)
            a_labels.append(label)
        
        feat['id']    = a_ids
        feat['label'] = a_labels
        
        features.append(feat)
        
    all_f = pd.concat(features)
    all_f.reset_index(drop=True, inplace=True)

    return all_f
