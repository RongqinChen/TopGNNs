export CUBLAS_WORKSPACE_CONFIG=:4096:8

dataset_list=('BZR' 'COX2' 'DHFR' 'FRANKENSTEIN' 'Mutagenicity' 
              'NCI1' 'NCI109' 'PTC_FR' 'ENZYMES' 'PROTEINS_full')
lossfn_list=("torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.BCEWithLogitsLoss"
             "torch.nn.CrossEntropyLoss"
             "torch.nn.BCEWithLogitsLoss")


ringsize_list=(6 8)
for((di=0; di<${#dataset_list[@]}; di++)) do
    echo ${dataset_list[di]};
    echo ${lossfn_list[di]};
    for((height=2; height<8; height++)) do
        for max_ring_size in ${ringsize_list[@]}; do
            python -m RFGNNMR.runner4tud \
            --dataset_name ${dataset_list[di]} \
            --max_ring_size ${max_ring_size} \
            --tree_height ${height} \
            --loss_module_name ${lossfn_list[di]} \
            --readout sum \
            --disable_tqdm;
        done;
    done;
done;