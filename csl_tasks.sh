export CUBLAS_WORKSPACE_CONFIG=:4096:8

for((height=1000; height<20000; height=height+1000)) do 
    python -m LocalTopGNN.runner4csl --max_ring_size 8 --tree_height 2 --disable_tqdm --seed ${height}
done;
