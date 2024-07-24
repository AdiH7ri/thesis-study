import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf
from pathlib import Path
from audio_utils import DIR_PATH, AUD_FILE_PATH

def z_standardization(Arr : np.ndarray):
    return (Arr - np.mean(Arr)/np.std(Arr))


audio_files=AUD_FILE_PATH.glob('*.mp3')
audio_file_names=[i.stem for i in audio_files] 
audio_file_names.sort()

# Cropping particular number of seconds from the beginning of each audio file to remove the initiating signals containing ' 3,2,1 *clap* ' sound 
# As it is observed in almost all of the samples

seconds_to_crop_from_the_beginning = 3.5

# n_std_thresh_stationary - Number of standard deviations above mean to place the threshold between signal and noise., by default 1.5

thresh = 1.75 

# for i in range(0,len(audio_file_names),2):
#     x1, sr = librosa.load(Path.joinpath(AUD_FILE_PATH, audio_file_names[i] + '.mp3'), sr=None)
#     x2, _ = librosa.load(Path.joinpath(AUD_FILE_PATH, audio_file_names[i+1] + '.mp3'), sr=None) # As the sample rate is same for all recordings

#     x1_standardized = z_standardization(x1[int( seconds_to_crop_from_the_beginning * sr ):])
#     x2_standardized = z_standardization(x2[int( seconds_to_crop_from_the_beginning * sr ):])

#     x1_processed = nr.reduce_noise(y = x1_standardized, sr=sr, y_noise = x2_standardized, n_std_thresh_stationary= thresh,stationary=True)
#     x2_processed = nr.reduce_noise(y = x2_standardized, sr=sr, y_noise = x1_standardized, n_std_thresh_stationary= thresh,stationary=True)

#     savepath_1 = Path.joinpath(DIR_PATH, 'processed-data', audio_file_names[i] + '-processed.mp3')
#     savepath_2 = Path.joinpath(DIR_PATH, 'processed-data', audio_file_names[i+1] + '-processed.mp3')

#     sf.write(savepath_1, x1_processed, sr)
#     sf.write(savepath_2, x2_processed, sr)

#     print(f'Processed and saved {audio_file_names[i]},mp3, { audio_file_names[i+1]}.mp3 !')

print('works')