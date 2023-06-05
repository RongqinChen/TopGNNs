import os
import time
from multiprocessing import Process
from datautils.tu.data_module import TU_DataModule
from itertools import product


def runner(gpu_idx, job):
    cublas = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cublas} && {cuda} && {job}")
    print(f"Done job of `{job}`.")


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


def run_tasks(dataset_list, gpu_idxs, heights, gpu_capacity=2):

    GDPs = [0, 0.5]
    DPs = [0.0]
    free_gpus = sum([[idx] * gpu_capacity for idx in gpu_idxs], list())
    running_jobs = []

    for dname in dataset_list:
        TU_DataModule(dname, 1, 114514, 32, 0)
        # download the dataset and generate graphs
        # for height in range(2, 8):
        for height, GDP, DP in product(
                heights, GDPs, DPs):
            job = "python -m GlobalTopGNN.runner4tud " + \
                f"--dataset_name {dname} " + \
                f"--tree_height {height} " + \
                f"--loss_module_name {loss_module_dict[dname]} " + \
                "--readout sum " + \
                f"--graph_dropout_p {GDP} " + \
                f"--dropout_p {DP} " + \
                "--disable_tqdm " + \
                "--hidden_dim 128 " + \
                "--hilayers 2"

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
    # gpu_idxs = [0, 1, 2, 3]
    gpu_idxs = [0, 0, 0]
    dataset_list = ['BZR', 'COX2', 'DHFR', 'FRANKENSTEIN',
                    'Mutagenicity', 'NCI1', 'NCI109', 'PTC_FR']
    heights = [3, 6, 4, 5, 2, 7, ]
    gpu_capacity = 2
    run_tasks(dataset_list, gpu_idxs, heights, gpu_capacity)

    dataset_list = ['ENZYMES',  'PROTEINS_full']
    # gpu_idxs = [0, 1, 2, 3]
    gpu_idxs = [0, 0, 0]
    gpu_capacity = 1
    run_tasks(dataset_list, gpu_idxs, heights, gpu_capacity)
    heights = [3, 6, 4, 5, 2, ]
