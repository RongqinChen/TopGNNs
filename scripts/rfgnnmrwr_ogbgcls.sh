export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m RFGNNMRwR.runner4ogbgcls \
    --dataset_name "ogbg-molpcba" \
    --max_ring_size 6 \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm
