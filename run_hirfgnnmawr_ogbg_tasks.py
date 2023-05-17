import os
import time
from multiprocessing import Process
# from datautils.ogbg.data_module import OGBG_DataModule
from itertools import product


def runner(gpu_idx, job):
    cuda = f"export CUDA_VISIBLE_DEVICES={gpu_idx}"
    os.system(f"{cuda} && {job}")
    print(f"Done job of `{job}`.")


def request_gpu(own_dict, need):
    for idx, own in own_dict.items():
        if own >= need:
            own_dict[idx] -= own
            return idx
    return None


def query_gpu(own_dict, need):
    for _, own in own_dict.items():
        if own >= need:
            return True
    return False


def release_gpu(own_dict, gpu_idx, used):
    own_dict[gpu_idx] += used


def binary_classification_nonan():
    dname_list = ['ogbg-molmuv', 'ogbg-molpcba', ]
    # for dname in dname_list:
    #     OGBG_DataModule(dname, 32, 0)
    minutes = 0
    own_dict = {gpu_idx: 7 for gpu_idx in range(1)}
    height_list = [2, 3, 4]
    hilayers_list = [2, 3]

    running_jobs = []
    for hilayers, dname in product(hilayers_list, dname_list):
        for height in height_list:
            job = "python -m CaRFGNNMRwR.runner4ogbgcls" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --hilayers {hilayers}" + \
                f" --max_ring_size {6}" + \
                " --readout sum" + \
                " --disable_tqdm"

            while True:
                gpu_geted = request_gpu(own_dict, height)
                if gpu_geted is not None:
                    p = Process(target=runner, args=(gpu_geted, job,),
                                name=f"process for {job}")
                    running_jobs.append((p, gpu_geted, height))
                    print(f"launch `{job}`")
                    p.start()
                    time.sleep(1)
                    break
                else:
                    while True:
                        running_jobs_copy = running_jobs
                        running_jobs = []
                        for p, gpu_idx, used in running_jobs_copy:
                            if p.is_alive():
                                running_jobs.append((p, gpu_idx, used))
                            else:
                                release_gpu(own_dict, gpu_idx, used)
                        if query_gpu(own_dict, height):
                            break
                        else:
                            time.sleep(60)
                            if minutes % 10 == 0:
                                os.system('nvidia-smi')
                            minutes += 1

    for (p, gpu_idx) in running_jobs:
        p.join()


if __name__ == "__main__":
    binary_classification_nonan()
