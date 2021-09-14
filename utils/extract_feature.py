import argparse
import librosa
from tqdm import tqdm
import io
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from pypeln import process as pr
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
parser.add_argument('-o', '--output', type=str, required=True)
parser.add_argument('-c', type=int, default=4)
parser.add_argument('-sr', type=int, default=16000)
parser.add_argument('-col',
                    default='filename',
                    type=str,
                    help='Column to search for audio files')
parser.add_argument('-cmn', default=False, action='store_true')
parser.add_argument('-cvn', default=False, action='store_true')
parser.add_argument('-winlen',
                    default=32,
                    type=float,
                    help='FFT duration in ms')
parser.add_argument('-hoplen',
                    default=8,
                    type=float,
                    help='hop duration in ms')

parser.add_argument('-n_mels', default=14, type=int)
ARGS = parser.parse_args()

DF = pd.read_csv(ARGS.input_csv, sep='\t', usecols=[0])
DF.columns = ['filename']

MEL_ARGS = {
    'n_mels': ARGS.n_mels,
    'n_fft': 2048,
    'hop_length': int(ARGS.sr * ARGS.hoplen / 1000),
    'win_length': int(ARGS.sr * ARGS.winlen / 1000)
}

EPS = np.spacing(1)


def extract_feature(fname):
    ext = Path(fname).suffix
    try:
        if ext == '.wav':
            y, sr = sf.read(fname, dtype='float32')
            if y.ndim > 1:
                y = y.mean(1)
            y = librosa.resample(y, sr, ARGS.sr)
    except Exception as e:
        logging.error(e)
        logging.error(fname)
        raise
    lms_feature = np.log(librosa.feature.melspectrogram(y, **MEL_ARGS) + EPS).T
    return fname, lms_feature


with h5py.File(ARGS.output, 'w') as store:
    for fname, feat in tqdm(pr.map(extract_feature, DF[ARGS.col].unique(), workers=ARGS.c, maxsize=4),
                            total=len(DF[ARGS.col].unique())):
        basename = Path(fname).name
        store[basename] = feat
