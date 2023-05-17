export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m RFGNN.runner4ogbgcls \
    --dataset_name "ogbg-molbace" \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm
