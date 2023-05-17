export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

python -m RFGNN.runner \
    --config_name "pepstruct" \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm

python -m RFGNN.runner \
    --config_name "pepstruct" \
    --tree_height 8 \
    --readout sum \
    --disable_tqdm
