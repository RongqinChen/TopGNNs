import os
import time
from multiprocessing import Process
from datautils.ogbg.data_module import OGBG_DataModule
from itertools import product


def runner(gpu_idx, job):
    cublas = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cublas} && {cuda} && {job}")
    print(f"Done job of `{job}`.")


def main():
    dataset_list = ['ogbg-molpcba', 'ogbg-molhiv', ]

    gpu_capacity = 1
    gpu_idxs = [0, 1, 2, 3]
    free_gpus = sum([[idx] * gpu_capacity for idx in gpu_idxs], list())
    running_jobs = []
    loss_module_dict = {
        'ogbg-molhiv': "torch.nn.BCEWithLogitsLoss",
        'ogbg-molpcba': "torch.nn.BCEWithLogitsLoss",
    }
    for dname in dataset_list:
        OGBG_DataModule(dname, 1, 0)
        # download the dataset and generate graphs

    for dname in dataset_list:
        for hi, height in product(range(2, 4), range(2, 4)):
            job = "python -m CaRFGNN.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 5" + \
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

    for dname in dataset_list:
        for hi, height in product([2, 3], range(2, 8)):
            job = "python -m CaRFGNNMRwR.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                " --max_ring_size 6" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 5" + \
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

    for dname in dataset_list:
        for height in range(2, 8):
            job = "python -m RFGNN.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 5" + \
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

    for dname in dataset_list:
        for max_ring_size, height in product([6, 8], range(2, 8)):
            job = "python -m RFGNNMR.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --max_ring_size {max_ring_size}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 5" + \
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

    for dname in dataset_list:
        for max_ring_size, height in product([6, 8], range(2, 8)):
            job = "python -m RFGNNMRwR.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --max_ring_size {max_ring_size}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 5" + \
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
