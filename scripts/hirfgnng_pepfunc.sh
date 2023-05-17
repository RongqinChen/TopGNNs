export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m CaRFGNNG.runner \
    --config_name "pepfunc" \
    --hilayers 2 \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm
