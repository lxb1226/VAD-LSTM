import os
from tqdm import tqdm
import json
import librosa

from vad_utils import  read_label_from_file
import configure as C_

# 需要将其写成一个类的形式进行处理更好
# TODO：之后还需要进一步的优化，目前先重构代码吧

def preprocess_data():
    # 数据预处理
    # TODO：考虑将数据预处理以及加载封装成一个类
    print('stage 0: data preparation and feature extraction')
    dev_lbl_dict = read_label_from_file(path=os.path.join(data_path, r"new_dev_label.txt"))
    # 加载验证集
    for wav in tqdm(os.listdir(dev_path)):
        wav_id = wav.split('.')[0]
        wav_file = os.path.join(dev_path, wav)
        wav_array, fs = librosa.load(wav_file, sr=args.fs)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                   win_hop=args.win_hop)
        frame_num = wav_framed.shape[0]
        assert frame_num >= len(dev_lbl_dict[wav_id])
        dev_lbl_dict[wav_id] += [0] * (frame_num - len(dev_lbl_dict[wav_id]))

        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        assert frame_num == len(dev_lbl_dict[wav_id])
        # 提取mfcc特征
        frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
        # 联结帧能量以及mfcc特征
        frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
        # 将特征 + 能量 保存到文件中
        np.save(os.path.join(dev_feat_path, wav_id + '.npy'), frame_feats)
    # if args.make_dev_lbl_dict:
    json_str = json.dumps(dev_lbl_dict)
    with open(os.path.join(data_path, r"dev_lbl_dict.json"), 'w') as json_file:
        json_file.write(json_str)

    # 加载训练集
    train_lbl_dict = read_label_from_file(path=os.path.join(data_path, r"new_train_label.txt"))
    for wav in tqdm(os.listdir(train_path)):
        wav_id = wav.split('.')[0]
        wav_file = os.path.join(train_path, wav)
        wav_array, fs = librosa.load(wav_file, sr=args.fs)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                   win_hop=args.win_hop)
        frame_num = wav_framed.shape[0]
        # if args.make_train_lbl_dict:
        assert frame_num >= len(train_lbl_dict[wav_id])
        train_lbl_dict[wav_id] += [0] * (frame_num - len(train_lbl_dict[wav_id]))  # 补0

        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        assert frame_num == len(train_lbl_dict[wav_id])
        frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
        frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
        np.save(os.path.join(train_feat_path, wav_id + '.npy'), frame_feats)
    json_str = json.dumps(train_lbl_dict)
    with open(os.path.join(data_path, r"train_lbl_dict.json"), 'w') as json_file:
        json_file.write(json_str)

    # 加载测试数据集
    for wav in tqdm(os.listdir(clean_test_path)):
        wav_file = os.path.join(clean_test_path, wav)
        wav_id = wav.split('.')[0]

        wav_array, fs = librosa.load(wav_file, sr=args.fs)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                   win_hop=args.win_hop)
        frame_num = wav_framed.shape[0]

        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
        frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
        np.save(os.path.join(clean_feat_path, wav_id + '.npy'), frame_feats)

    for wav in tqdm(os.listdir(seen_noise_test_path)):
        wav_file = os.path.join(seen_noise_test_path, wav)
        wav_id = wav.split('.')[0]

        wav_array, fs = librosa.load(wav_file, sr=args.fs)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                   win_hop=args.win_hop)
        frame_num = wav_framed.shape[0]

        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
        frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
        np.save(os.path.join(seen_noise_feat_path, wav_id + '.npy'), frame_feats)

    for wav in tqdm(os.listdir(unseen_noise_test_path)):
        wav_file = os.path.join(unseen_noise_test_path, wav)
        wav_id = wav.split('.')[0]

        wav_array, fs = librosa.load(wav_file, sr=args.fs)
        wav_framed, frame_len = preprocess.framing(wav_array, fs=args.fs, win_len=args.win_len,
                                                   win_hop=args.win_hop)
        frame_num = wav_framed.shape[0]

        frame_energy = (wav_framed ** 2).sum(1)[:, np.newaxis]
        frame_mfcc = mfcc.mfcc(wav_array, fs=args.fs, win_len=args.win_len, win_hop=args.win_hop)
        frame_feats = np.concatenate((frame_energy, frame_mfcc), axis=1)
        np.save(os.path.join(unseen_noise_feat_path, wav_id + '.npy'), frame_feats)


    with open(os.path.join(data_path, r"dev_lbl_dict.json"), 'r') as f:
        dev_lbl_dict = json.load(f)
    with open(os.path.join(data_path, r"train_lbl_dict.json"), 'r') as f:
        train_lbl_dict = json.load(f)