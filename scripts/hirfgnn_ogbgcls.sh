export CUDA_VISIBLE_DEVICES=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m HiRFGNNG.runner4ogbgcls \
    --dataset_name "ogbg-molpcba" \
    --hilayers 3 \
    --tree_height 3 \
    --readout sum \
    --disable_tqdm \
    --hidden_dim 64
