export CUBLAS_WORKSPACE_CONFIG=:4096:8

dataset_list=('FRANKENSTEIN')
lossfn_list=("torch.nn.BCEWithLogitsLoss")


for((di=0; di<${#dataset_list[@]}; di++)) do
    echo ${dataset_list[di]};
    echo ${lossfn_list[di]};
    for((hi=2; hi<4; hi++)) do
        for((height=2; height<8; height++)) do 
            python -m GCaRFGNNMRwR.runner4tud \
            --dataset_name ${dataset_list[di]} \
            --hilayers ${hi} \
            --tree_height ${height} \
            --max_ring_size 6 \
            --loss_module_name ${lossfn_list[di]} \
            --readout sum \
            --hidden_dim 64 \
            --disable_tqdm;
        done;
    done;
done;
