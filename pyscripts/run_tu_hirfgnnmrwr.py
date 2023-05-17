import os
import time
from itertools import product
from multiprocessing import Process

from datautils.tu.data_module import TU_DataModule


def runner(gpu_idx, job):
    cublas = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cublas} && {cuda} && {job}")
    print(f"Done job of `{job}`.")


def main():
    dataset_list = ['BZR', 'COX2', 'DHFR', 'FRANKENSTEIN',
                    'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FR']
    # big_dataset_list = ['ENZYMES',  'PROTEINS_full']

    gpu_capacity = 6
    gpu_idxs = [0, 1, 2, 3]
    free_gpus = sum([[idx] * gpu_capacity for idx in gpu_idxs], list())
    running_jobs = []
    loss_module_dict = {
        'BZR': "torch.nn.BCEWithLogitsLoss",
        'COX2': "torch.nn.BCEWithLogitsLoss",
        'DHFR': "torch.nn.BCEWithLogitsLoss",
        'ENZYMES': "torch.nn.CrossEntropyLoss",
        'FRANKENSTEIN': "torch.nn.BCEWithLogitsLoss",
        'Mutagenicity': "torch.nn.BCEWithLogitsLoss",
        'NCI1': "torch.nn.BCEWithLogitsLoss",
        'NCI109': "torch.nn.BCEWithLogitsLoss",
        'PTC_FR': "torch.nn.BCEWithLogitsLoss",
        'PROTEINS_full': "torch.nn.BCEWithLogitsLoss",
    }
    for dname in dataset_list:
        TU_DataModule(dname, 1, 114514, 32, 0)
        # download the dataset and generate graphs
        for hi, height in product(range(2, 3), [1, 6, 2, 5, 3, 4]):
            job = "python -m CaRFGNNMRwR.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                " --max_ring_size 6" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            while True:
                if len(free_gpus) > 0:
                    gpu_idx = free_gpus.pop(0)
                    p = Process(target=runner, args=(gpu_idx, job,),
                                name=f"process for {job}")
                    running_jobs.append((p, gpu_idx))
                    print(f"launch `{job}`")
                    p.start()
                    time.sleep(1)
                    break
                else:
                    while len(free_gpus) == 0:
                        running_jobs_copy = running_jobs
                        running_jobs = []
                        for p, gpu_idx in running_jobs_copy:
                            if p.is_alive():
                                running_jobs.append((p, gpu_idx))
                            else:
                                free_gpus.append(gpu_idx)
                        if len(free_gpus) == 0:
                            time.sleep(5)

    for (p, gpu_idx) in running_jobs:
        p.join()


if __name__ == "__main__":
    main()
