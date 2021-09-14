import os
from tqdm import tqdm
import json
import librosa
import spafe.utils.preprocessing as preprocess
import spafe.features.mfcc as mfcc
from vad_utils import read_label_from_file
import numpy as np


# 预处理音频数据

def generate_feat_data(data_path, csv_file, feat_path, label_path, fs=16000, win_len=0.032, win_hop=0.008):
    data_lbl_dict = read_label_from_file(label_path)
    for wav in tqdm(os.listdir(data_path)):
        wav_id = wav.split('.')[0]
        wav_file = os.path.join(data_path, wav)
        wav_array, fs = librosa.load(wav_file, sr=16000)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=fs, win_len=win_len,
                                                   win_hop=win_hop)
        frame_num = wav_framed.shape[0]
        assert frame_num >= len(data_lbl_dict[wav_id])

        data_lbl_dict[wav_id] += [0] * (frame_num - len(data_lbl_dict[wav_id]))
        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        assert frame_num == len(data_lbl_dict[wav_id])

