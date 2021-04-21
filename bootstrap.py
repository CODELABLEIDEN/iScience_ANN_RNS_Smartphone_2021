import numpy as np

import json
from pywt import cwt
import utils
import time
from tqdm import tqdm
import matplotlib.font_manager
import copy
from utils import scales
import pickle as pkl
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import ttest_1samp


def main():
    for j, sub_id in enumerate(['1', '2', '3', '4', '5', '6', '7', '8']):
        res = np.load(f'v5_p{sub_id}_out_recon.npy', allow_pickle=True)
        img_feat = res.item().get('image_feat')[..., 0]  # (B, 50, 50)
        SURROGATE_CWT = []
        BINS = 50
        eps = 1e-8
        for i in range(100):
            sta = time.time()
            _c = copy.deepcopy(img_feat)
            np.random.shuffle(_c)
            res_dt = _c.reshape(-1, BINS * BINS)
            x = copy.deepcopy(res_dt)
            x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + eps)
            cwt_dt_dt, _ = cwt(x.T, scales, 'cmor1.0-1.0', method='fft')
            CWT_DT = np.abs(cwt_dt_dt).mean(-1)
            SURROGATE_CWT.append(CWT_DT)
            print(f"{sub_id} - {i} - {time.time() - sta}")

        pkl.dump(np.array(SURROGATE_CWT), open(f'surrogate_cwt_abs_sub_{sub_id}.pkl', 'wb'))


if __name__ == '__main__':
    main()