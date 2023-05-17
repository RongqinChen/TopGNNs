import os
import time
from multiprocessing import Process
from datautils.ogbg.data_module import OGBG_DataModule


def runner(gpu_idx, job):
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cuda} && {job}")
    print(f"Done job of `{job}`.")


def run():
    dataset_name_list = ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']
    dataset_name_list = ['ogbg-mollipo']
    # num_cudas = 2
    # max_H = 7
    # free_gpus = sum([[idx] * max_H for idx in range(0, num_cudas)], list())
    free_gpus = [0, 0, 1, 1, 0, 1]
    running_jobs = []

    for dname in dataset_name_list:
        OGBG_DataModule(dname, 32, 0, 'to_TPF', {'height': 1})
        for height in range(4, 9, 4):
            job = "python -m RFGNN.runner4ogbgreg" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
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
                            time.sleep(60)

    for (p, gpu_idx) in running_jobs:
        p.join()


if __name__ == "__main__":
    run()
