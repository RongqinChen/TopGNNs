export CUBLAS_WORKSPACE_CONFIG=:4096:8


# python -m CaRFGNNG.runner \
#     --config_name "pepstruct" \
#     --hilayers 2 \
#     --tree_height 6 \
#     --readout sum \
#     --disable_tqdm

python -m CaRFGNNG.runner \
    --config_name "pepstruct" \
    --hilayers 2 \
    --tree_height 3 \
    --readout sum \
    --disable_tqdm

python -m CaRFGNNG.runner \
    --config_name "pepstruct" \
    --hilayers 2 \
    --tree_height 4 \
    --readout sum \
    --disable_tqdm
