import os
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


def main():
    dataset_list = ['ENZYMES',  'PROTEINS_full']
    for dname in dataset_list:
        TU_DataModule(dname, 1, 114514, 32, 0)

    job_list = []
    for dname in dataset_list:
        for hi, height in product(range(2, 3), [2, 7, 3, 6, 4, 5]):
            job = "python -m CaRFGNN.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for dname in dataset_list:
        for hi, height in product(range(2, 3), [2, 7, 3, 6, 4, 5]):
            job = "python -m CaRFGNNG.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for dname in dataset_list:
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

            job_list.append(job)

    for dname in dataset_list:
        for hi, height in product(range(2, 3), [1, 6, 2, 5, 3, 4]):
            job = "python -m CaRFGNNMRwRG.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --hilayers {hi}" + \
                f" --tree_height {height}" + \
                " --max_ring_size 6" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for dname in dataset_list:
        for height in [3, 6, 4, 5, 2, 7, ]:
            job = "python -m RFGNN.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for dname in dataset_list:
        for max_ring_size, height in product([6], [1, 6, 2, 5, 3, 4]):
            job = "python -m RFGNNMR.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --max_ring_size {max_ring_size}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for dname in dataset_list:
        for max_ring_size, height in product([6], [1, 6, 2, 5, 3, 4]):
            job = "python -m RFGNNMRwR.runner4tud" + \
                f" --dataset_name {dname}" + \
                f" --tree_height {height}" + \
                f" --max_ring_size {max_ring_size}" + \
                f" --loss_module_name {loss_module_dict[dname]}" + \
                " --readout sum" + \
                " --num_runs 10" + \
                " --disable_tqdm"

            job_list.append(job)

    for job in job_list:
        runner(0, job)


if __name__ == "__main__":
    main()
