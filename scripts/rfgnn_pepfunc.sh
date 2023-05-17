export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m RFGNN.runner \
    --config_name "pepfunc" \
    --tree_height 6 \
    --readout sum \
    # --disable_tqdm
