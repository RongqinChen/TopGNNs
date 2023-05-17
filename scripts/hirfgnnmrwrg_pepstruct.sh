export CUBLAS_WORKSPACE_CONFIG=:4096:8

export CUDA_VISIBLE_DEVICES=0

python -m CaRFGNNMRwRG.runner \
    --config_name "pepstruct" \
    --hidden_dim 64 \
    --hilayers 3 \
    --max_ring_size 6 \
    --tree_height 2 \
    --readout sum \
    --disable_tqdm


python -m CaRFGNNMRwRG.runner \
    --config_name "pepstruct" \
    --hilayers 4 \
    --max_ring_size 6 \
    --tree_height 2 \
    --readout sum \
    --disable_tqdm
