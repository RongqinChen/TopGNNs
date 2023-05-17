import os
import time
from multiprocessing import Process
from datautils.peptides.data_module import PeptidesDataModule
from itertools import product


def runner(gpu_idx, job):
    cublas = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cublas} && {cuda} && {job}")
    print(f"Done job of `{job}`.")


def main():
    PeptidesDataModule('structural', 1, 0)

    free_gpus = list(range(4))
    running_jobs = []

    for hi, height in product(range(2, 3), [2, 3, 4, 5]):
        job = "python -m CaRFGNNMRwRG.runner" + \
            " --config_name pepstruct" + \
            f" --hilayers {hi}" + \
            f" --tree_height {height}" + \
            " --max_ring_size 6" + \
            " --hidden_dim 80" + \
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
