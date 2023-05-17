import os
import time
from multiprocessing import Process
from datautils.ged.data_module import GED_DataModule
from itertools import product


def runner(gpu_idx, job):
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cuda} && {job}")
    print(f"Done job of `{job}`.")


def get_gpu_idx(need, gpu_resource, num_gpus):
    for gpu_idx in range(num_gpus):
        if gpu_resource[gpu_idx] >= need:
            gpu_resource[gpu_idx] -= need
            return gpu_idx

    return None


def run_tasks():
    dataset_name_list = ['AIDS700nef', 'LINUX', 'IMDBMulti']
    models = ['RFGNN']

    num_gpus = 2
    heights = [6, 8, 10, 12]
    gpu_resource = {idx: sum(heights) for idx in range(num_gpus)}

    running_jobs = []
    for dname in dataset_name_list:
        GED_DataModule(dname, 32, 0, 'to_TPF', {'height': 1})
        for height, model_name in product(heights, models):

            job = f"python -m {model_name}.runner4ged" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                " --readout sum" + \
                " --num_runs 5" + \
                " --disable_tqdm"

            while True:
                gpu_idx = get_gpu_idx(height, gpu_resource, num_gpus)
                if gpu_idx is not None:
                    jobp = Process(target=runner, args=(gpu_idx, job,),
                                   name=f"process for {job}")
                    running_jobs.append((jobp, height, gpu_idx))
                    print(f"launch `{job}`")
                    jobp.start()
                    time.sleep(2)
                    break
                else:
                    while gpu_idx is None:
                        running_jobs_copy = running_jobs
                        running_jobs = []
                        for jobp, resource_need, gpu_idx in running_jobs_copy:
                            if jobp.is_alive():
                                running_jobs.append(
                                    (jobp, resource_need, gpu_idx))
                            else:
                                gpu_resource[gpu_idx] += resource_need

                        gpu_idx = get_gpu_idx(height, gpu_resource, num_gpus)
                        if gpu_idx is None:
                            time.sleep(60)

    for (jobp, gpu_idx) in running_jobs:
        jobp.join()


if __name__ == "__main__":
    run_tasks()
