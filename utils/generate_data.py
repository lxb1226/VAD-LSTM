import os
import random
import librosa
import soundfile
import scipy.io as scio
import numpy as np
from pathlib import Path

noise_types = ['babble', 'buccaneer1', 'buccaneer2', 'destroyerengine', 'destroyerops',
               'f16', 'factory1', 'factory2', 'hfchannel', 'leopard', 'm109',
               'machinegun', 'pink', 'volvo', 'white']

SEEN_NOISES = ['babble', 'buccaneer1', 'destroyerengine',
               'f16', 'factory1', 'hfchannel', 'leopard', 'm109',
               'pink', 'white']

UNSEEN_NOISES = ['buccaneer2', 'destroyerops', 'factory2', 'machinegun', 'volvo']

SEEN_NOISE_DATA = {}
UNSEEN_NOISE_DATA = {}

NOISE_DIR = "../noisex-92wav/"
DEV_LABEL = "../data/dev_label.txt"
TRAIN_LABEL = "../data/train_label.txt"


# dev_label_dict = read_label_from_file(DEV_LABEL)
# train_label_dict = read_label_from_file(TRAIN_LABEL)

def get_noises(sr=16000):
    files = os.listdir(NOISE_DIR)
    for file in files:
        noise_name = file.split('.')[0]
        noise_file = os.path.join(NOISE_DIR, file)
        # print(noise_file)
        noise_data, sr = librosa.load(noise_file, sr=sr)
        if noise_name in SEEN_NOISES:
            SEEN_NOISE_DATA[noise_name] = noise_data
        else:
            UNSEEN_NOISE_DATA[noise_name] = noise_data
    # print(SEEN_NOISE_DATA.values())


# 给干净的声音添加噪声
# 给音频添加噪声的同时还需要更新标签
# 同时当两段音频不一样长的时候，当clean_data > nosie_data时，需要将nosie_data复制到和clean_data一样长。
# 当clean_data < nosie_data时，需要将noise_data裁减到和clean_data一样长
# 最终生成的数据的文件名=源文件名+噪声名

# 构建数据集
'''
在这里我们将噪声划分为两类，分为见过的和未见过的，比例为10/5.
在10种见过的噪声中随机选取5中噪声加入到干净的训练集以及验证集中；
然后在测试集中加入15种噪声。
并且将测试集分为看见过噪声的以及未看见过噪声的测试集。
验证集中也加入随机的5种噪声类型。
最终形成的测试集分别有干净的测试集、加入10种见过噪声的测试集、加入5种未出现的噪声的测试集。
以这三种作为对比实验，比较模型的效果。

干净的数据集中，训练集有3600条数据；验证集有500条数据；测试集有1000条数据。
最终构建的数据集为：训练集有18000条数据；验证集有7500条数据；测试集有1000 + 10000 + 5000条数据
'''


def merge_wav(clean_data, noise_data):
    # 对齐操作
    if len(clean_data) > len(noise_data):
        noise_data = np.hstack((noise_data, noise_data[len(clean_data) - len(noise_data)]))
    elif len(clean_data) < len(noise_data):
        noise_data = noise_data[: len(clean_data)]
    dst_data = clean_data + noise_data
    # 归一化操作
    data = dst_data * 1.0 / (max(abs(dst_data)))
    return data


# 添加噪声，同时更新label
def add_noise(clean_file, t):
    for file_name in os.listdir(NOISE_DIR):
        noise_file = os.path.join(NOISE_DIR, file_name)
        if not clean_file.endswith('.wav'):
            continue
        noise_name = file_name.split('.')[0]
        clean_name = clean_file.split('/')[-1].split('.')[0]
        save_name = clean_name + "_" + noise_name
        print(noise_name, clean_name, save_name)
        if noise_file.endswith('.wav'):
            data = merge_wav(clean_file, noise_file)
            soundfile.write(save_name + ".wav", data, 16000)
            # 更新标签
            # if t == "dev":
            #     dev_label_dict[save_name] = dev_label_dict[clean_name]
            # elif t == "train":
            #     train_label_dict[save_name] = train_label_dict[clean_name]


# 生成数据
def generate_data(in_path, t):
    if t == 0 or t == 1:
        # 生成训练集以及验证集
        build_train_data(in_path, t)
    else:
        # 生成测试集
        build_test_data(in_path)


def build_test_data(test_path):
    files = os.listdir(test_path)
    for i in range(0, len(files)):
        clean_file = os.path.join(test_path, files[i])
        if os.path.isdir(clean_file):
            continue
        clean_name = files[i].split('.')[0]
        clean_data, sr = librosa.load(clean_file, sr=16000)
        # 生成两种类型的测试集，一种是加入了已知噪声；另一种是加入了未知噪声
        # 1.第一种，加入10种已知噪声，新建一个文件夹，进行存放
        for j in range(0, 10):
            noise_name = SEEN_NOISES[j]
            noise_data = SEEN_NOISE_DATA[noise_name]
            noise_file = test_path + 'seen_noise_test/' + clean_name + '_' + noise_name
            print(noise_file)
            syn_data = merge_wav(clean_data, noise_data)
            soundfile.write(noise_file + '.wav', data=syn_data, samplerate=16000)
        # 2.第二种，加入5种未知噪声，新建一个文件夹，进行存放
        for j in range(0, 5):
            noise_name = UNSEEN_NOISES[j]
            noise_data = UNSEEN_NOISE_DATA[noise_name]
            noise_file = test_path + 'unseen_noise_test/' + clean_name + '_' + noise_name
            print(noise_file)
            syn_data = merge_wav(clean_data, noise_data)
            soundfile.write(noise_file + '.wav', data=syn_data, samplerate=16000)


# 生成训练集
def build_train_data(train_path, t):
    files = os.listdir(train_path)
    for i in range(0, len(files)):
        clean_name = files[i].split('.')[0]
        print(clean_name)
        clean_file = os.path.join(train_path, files[i])
        clean_data, sr = librosa.load(clean_file, sr=16000)
        tmp = []
        j = 0
        while j < 5:
            idx = random.randint(0, 9)
            if idx in tmp:
                continue
            tmp.append(idx)
            noise_name = SEEN_NOISES[idx]
            noise_data = SEEN_NOISE_DATA[noise_name]
            save_name = train_path + clean_name + '_' + noise_name
            print(save_name)
            syn_data = merge_wav(clean_data, noise_data)
            soundfile.write(save_name + '.wav', data=syn_data, samplerate=16000)
            # 更新标签
            # if t == 0:
            #     # 更新训练集标签
            #     train_label_dict[save_name] = train_label_dict[clean_name]
            # else:
            #     # 更新验证集标签
            #     dev_label_dict[save_name] = dev_label_dict[clean_name]
            j += 1


def convertMatToWav(in_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for file_name in os.listdir(in_path):
        mat_data = scio.loadmat(in_path + '/' + file_name)
        sr = 19980
        key = file_name.split('/')[-1].split('.')[0]
        data = mat_data[key]
        # 重采样到16000
        data = 1.0 * data / (max(abs(data)))
        data = data.reshape(-1)
        data = librosa.resample(data, sr, 16000)
        soundfile.write(out_path + key + '.wav', data, sr)


def delete(in_path):
    for filename in os.listdir(in_path):
        if len(filename.split('_')) >= 2:
            # 删除
            path_name = os.path.join(in_path, filename)
            print(path_name)
            os.remove(path_name)


def read_labels(in_path):
    data = {}
    with Path(in_path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{in_path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({in_path}:{linenum})")
            data[k] = v
    return data


def write_labels(data, label_path):
    with Path(label_path).open("w", encoding="utf-8") as f:
        for k in data.keys():
            f.write(k)
            f.write(' ')
            f.write(data[k])
            f.write("\n")


def generate_label(data_path, label_path, new_label_path):
    data = read_labels(label_path)
    for filename in os.listdir(data_path):
        lists = filename.split('_')
        if len(lists) != 1:
            # 说明是添加噪声的数据
            clean_name = lists[0]
            print(clean_name)
            data[filename.split('.')[0]] = data[clean_name]
    # 写入文件
    write_labels(data, new_label_path)





if __name__ == "__main__":
    dev_path = "../wavs/dev/"
    train_path = "../wavs/train/"
    test_path = "../wavs/test/"

    # data = read_labels(TRAIN_LABEL)
    # print(data)
    # get_noises()
    # print(SEEN_NOISE_DATA.keys())
    # generate_data(train_path, 0)
    # 保存标签
    # generate_data(dev_path, 1)
    # generate_data(test_path, 2)
    # delete(dev_path)
    # new_train_label = "../data/new_train_label.txt"
    new_dev_label = "../data/new_dev_label.txt"
    # # generate_label(train_path, TRAIN_LABEL, new_train_label)
    #
    generate_label(dev_path, DEV_LABEL, new_dev_label)
