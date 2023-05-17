import os
import time
from multiprocessing import Process


job_list = [

    """python -m RFGNNMR.runner \
    --config_name "pepstruct" \
    --max_ring_size 6 \
    --tree_height 4 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepstruct" \
    --max_ring_size 6 \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepstruct" \
    --max_ring_size 6 \
    --tree_height 8 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepstruct" \
    --max_ring_size 6 \
    --tree_height 10 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepfunc" \
    --max_ring_size 6 \
    --tree_height 4 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepfunc" \
    --max_ring_size 6 \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepfunc" \
    --max_ring_size 6 \
    --tree_height 8 \
    --readout sum \
    --disable_tqdm""",

    """python -m RFGNNMR.runner \
    --config_name "pepfunc" \
    --max_ring_size 6 \
    --tree_height 10 \
    --readout sum \
    --disable_tqdm""",

]


for job in job_list:
    print(job)


def runner(gpu_idx, job):
    cublas = "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cublas} && {cuda} && {job}")
    print(f"Done job of `{job}`.")


free_gpus = [0, 1, 2, 3]
running_jobs = []
for job in job_list:

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
