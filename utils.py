import io

import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import pandas as pd
import elephant.statistics as stat
from elephant import kernels
from neo import SpikeTrain
from quantities import s
import pickle as pkl
import copy
from scipy.signal import stft
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from pywt import cwt
import pywt

tap_dir = '...'
les_dir = '...'

# reproducing the scales of https://www.nature.com/articles/s41467-017-02577-y
eps = 1e-5
scales1 = np.arange(2.4, 31.2 + eps, 1.2)
scales2 = np.arange(33.6, 48 + eps, 2.4)
scales3 = np.arange(2.2 * 24, 4 * 24 + eps, 4.8)
scales4 = np.arange(4.5 * 24, 10 * 24 + eps, 12)
scales5 = np.arange(11 * 24, 45 * 24 + eps, 24)

scales = np.concatenate([scales1, scales2, scales3, scales4, scales5])


def remove_24h_cwt(x):
    x = (x - np.mean(x)) / (np.std(x) + 1e-15)
    cwtmatr, freqs = cwt(x, scales, wavelet='cmor1.0-1.0', method='fft')

    cwtmatr_no_24 = copy.deepcopy(cwtmatr)
    cwtmatr_no_24[12:28] = 0.0 # np.mean(cwtmatr_no_24)

    mwf = pywt.ContinuousWavelet('cmor1.0-1.0').wavefun()
    y_0 = mwf[0][np.argmin(np.abs(mwf[1]))]

    r_sum_no_24 = np.sum(cwtmatr_no_24.real.T / scales, axis=-1).T
    reconstructed_no_24 = r_sum_no_24 * (y_0.real)
    x1 = (reconstructed_no_24 - np.mean(reconstructed_no_24)) / (np.std(reconstructed_no_24) + 1e-15)
    return x1


def my_corrcoef(x, y, eps=1e-8):

    n = len(x)
    xsum = x.sum()
    ysum = y.sum()
    xmean = xsum / n
    ymean = ysum / n

    xsqsum = ((x - xmean) ** 2).sum()
    ysqsum = ((y - ymean) ** 2).sum()
    cov = ((x - xmean) * (y - ymean)).sum()
    corr = cov / (np.sqrt(xsqsum * ysqsum) + eps)
    return corr


def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class DataGeneratorMixup(Sequence):
    def __init__(self, x, y, batch_size=8, n_samples=2000, shuffle=True, mixup='mixup', det=None):
        self.batch_size = batch_size
        self.indices = list(range(n_samples)) if shuffle else list(range(len(x)))
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.mixup = mixup
        self.det = det

        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):

        X = []
        Y = []
        for i in range(self.batch_size):
            if self.shuffle:
                if self.mixup == 'mixup':
                    idx1 = np.random.choice(len(self.x))
                    idx2 = np.random.choice(len(self.x))
                    alpha = np.random.rand()
                    _x = alpha * self.x[idx1] + (1 - alpha) * self.x[idx2]
                    _y = alpha * self.y[idx1] + (1 - alpha) * self.y[idx2] + 1e-8
                elif self.mixup == 'patch':
                    side = np.random.choice(range(5, 10))
                    cx = np.random.choice(self.x.shape[2] - side - 1)
                    cy = np.random.choice(self.x.shape[3] - side - 1)

                    idx = np.random.choice(len(self.x))
                    _x = self.x[idx]  # (50, 50, 2)
                    _y = self.y[idx]  # (4)

                    for j in range(cx, cx + side):
                        for ll in range(cy, cy + side):
                            _x[:, j, ll] = 0.
                else:
                    idx = np.random.choice(len(self.x))
                    _x = self.x[idx]
                    _y = self.y[idx]
            else:
                idx = index * self.batch_size
                _x = self.x[idx + i]
                _y = self.y[idx + i]

            X.append(_x)
            Y.append(_y)

        if self.det is None:
            return np.array(X).astype('float32'), [np.array(Y)[:, i].astype('float32') for i in range(np.array(Y).shape[1])]
        else:
            return np.array(X).astype('float32'), [np.array(Y)[:, self.det - 1].astype('float32')]

    def on_epoch_end(self):
        pass


def prepare_dataset(sub_id, ctx, split, feat, exp_y, absres, rem='poly'):
    
    ...
    # prepare here the dataset: 

    XX['train'][f"{nameX}"] = nX_train # shape [n_samples, len_ctx, 50, 50, 2]
    XX['test'][f"{nameX}"] = nX_test # shape [n_samples, num_detectors_output]
    YY['train'][f"{nameY}"] = nY_train # shape [n_samples, len_ctx, 50, 50, 2]
    YY['test'][f"{nameY}"] = nY_test # shape [n_samples, num_detectors_output]
    LE['train'] = ll_train
    LE['test'] = ll_test

    for subset in ['train', 'test']:
        for (_name, _x) in XX[subset].items():
            for (_name, _y) in YY[subset].items():
                assert (len(_x) == len(_y) == len(LE[subset]))

    return XX, YY, LE


def extract_ctx(y, x, ctx=(6, 0), is_2D=False, getz=False, norm=False):
    new_x = []
    new_y = []
    new_z = []
    end = y.shape[0] - ctx[1] - 1 if ctx[1] > 0 else y.shape[0]
    for i in range(ctx[0], end):
        if ctx[1] > 0:
            _k = x[i - ctx[0]: i + ctx[1] + 1]
            _z = y[i - ctx[0]: i + ctx[1] + 1]
        else:
            _k = x[i - ctx[0]: i]
            _z = y[i - ctx[0]: i]
        new_x.append(_k if is_2D else _k.reshape(-1))
        new_y.append(y[i])
        new_z.append(_z.reshape(-1))
    
    if getz:
        return np.array(new_x), np.array(new_y), np.array(new_z)
    else:
        return np.array(new_x), np.array(new_y)


def split_season(y, season=24, degree=5):
    x = np.arange(len(y)) % season
    coef = np.polyfit(x, y, degree)

    curve = []
    for i in range(len(x)):
        value = coef[-1]
        for d in range(degree):
            value += x[i]**(degree-d) * coef[d]
        curve.append(value)

    model_24h = np.array(curve)
    residual = y - model_24h
    return model_24h, residual


def dt_to_JID(taps_times, BINS = 50, MIN_H = 1.5, MAX_H = 5):
    some_dt = np.diff(taps_times)  # in ms
    
    xx, yy = np.mgrid[MIN_H : MAX_H : BINS * 1j, MIN_H : MAX_H : BINS * 1j]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

    log_dt1, log_dt2 = np.log10(np.array(some_dt[:-1]) + eps), np.log10(np.array(some_dt[1:]) + eps)
    xy_train  = np.vstack([log_dt1, log_dt2]).T
    kde_skl = KernelDensity(bandwidth=0.1)
    kde_skl.fit(xy_train)
    z = np.exp(kde_skl.score_samples(xy_sample))
    dt_dt= np.reshape(z, xx.shape)
    
    return dt_dt


def JID_entropy(jid, BINS = 50, MIN_H = 1.5, MAX_H = 5):
    # jid == BINS-by-BINS image
    step = (MAX_H - MIN_H) / BINS
    fc = jid * step * step
    if fc.sum() > eps:
        return = - np.sum(fc * np.log2(fc + eps))
    else:
        return 0.


def periodgram(x):
    # reproducing https://www.nature.com/articles/s41467-017-02577-y
    x = (x - np.mean(x)) / np.std(x)
    cwtmatr, freqs = cwt(x, scales, 'cmor1.0-1.0', method='fft')
    my_xaxis = [find_nearest(1. / freqs / 24, i) for i in [0.5, 1, 3, 7, 15, 30]]  # axis in days

    X = np.sqrt(np.mean(np.abs(cwtmatr), 1))
    return X, my_xaxis


def bootstrap_periodgram(x):
    surrogate = []
    for _ in tqdm(range(1000)):
        m = 72
        y = copy.deepcopy(x)
        y = (y - np.mean(y)) / (np.std(y) + 1e-15)
        _a = np.zeros((len(y) // m) * m)
        for k in range(len(_a) // m):
            s = np.random.choice(range(1, m))
            _idx = np.random.choice(len(x) - s)
            _a[k * s:(k + 1) * s] = y[_idx:_idx + s]
        _c, _ = cwt(_a, scales, wavelet='cmor1.0-1.0', method='fft')
        _c = np.sqrt(np.abs(_c).mean(-1))
        surrogate.append(_c)
        np.random.shuffle(y)
        _c, _ = cwt(y, scales, wavelet='cmor1.0-1.0', method='fft')
        _c = np.sqrt(np.abs(_c).mean(-1))
        surrogate.append(_c)
    return surrogate