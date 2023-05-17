export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m CaRFGNN.runner \
    --config_name "pepstruct" \
    --hilayers 2 \
    --tree_height 6 \
    --readout sum \
    --disable_tqdm
