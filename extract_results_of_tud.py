import os
import os.path as osp
from collections import defaultdict
from typing import Mapping

import json
import numpy as np


def parse_label(label):
    items = label.split('/')
    dname = items[2]
    mname = items[3].split('_')[0]
    key = items[4]
    fold = items[5]
    return dname, mname, key, fold


def extract_results(src_dir, dst_path):

    flag = True
    dst_file = open(dst_path, 'w')
    results_fpath_list = [
        osp.join(folder, file)
        for folder, _, files in os.walk(src_dir)
        for file in files if file == 'results.json'
    ]

    for result_fpath in results_fpath_list:
        with open(result_fpath, 'r') as rfile:
            result_dict: Mapping = json.load(rfile)
            results_dict = defaultdict(list)
            for label, result in result_dict.items():
                # if result['best_epoch'] == 0:
                #     continue
                dname, mname, key, fold = parse_label(label)
                results_dict['dname'].append(dname)
                results_dict['mname'].append(mname)
                results_dict['key'].append(key)
                results_dict['fold'].append(fold)
                for rkey, rval in result.items():
                    results_dict[rkey].append(rval)

            results_dict2 = dict()
            for rkey, rval_list in results_dict.items():
                if rkey == 'fold':
                    results_dict2['num_fold'] = len(rval_list)
                elif isinstance(rval_list[0], float) or rkey == 'best_epoch':
                    rval_mean, rval_std = np.mean(rval_list), np.std(rval_list)
                    results_dict2[f"{rkey}_mean"] = rval_mean
                    results_dict2[f"{rkey}_std"] = rval_std
                else:
                    rval_set_str = "-".join(
                        [f"{val}" for val in sorted(set(rval_list))])
                    results_dict2[f"{rkey}_set"] = rval_set_str

            if len(results_dict2) > 0:
                key_list = list(results_dict2.keys())
                val_list = [results_dict2[key] for key in key_list]
                if flag:
                    print(*key_list, sep=',', file=dst_file)
                    flag = False
                print(*val_list, sep=',', file=dst_file)

    dst_file.close()


if __name__ == '__main__':
    src_dir = 'logs/TU'
    dst_path = 'logs/TU.csv'
    extract_results(src_dir, dst_path)
