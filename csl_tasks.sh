export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m LocalTopGNN.runner4csl --max_ring_size 8 --tree_height 2 --disable_tqdm
python -m LocalTopGNN.runner4csl --max_ring_size 8 --tree_height 3 --disable_tqdm
python -m LocalTopGNN.runner4csl --max_ring_size 8 --tree_height 4 --disable_tqdm
python -m LocalTopGNN.runner4csl --max_ring_size 8 --tree_height 5 --disable_tqdm

python -m RFGNN.runner4csl --tree_height 9 --disable_tqdm
python -m RFGNNMR.runner4csl --tree_height 9 --max_ring_size 8 --disable_tqdm
python -m RFGNNMRwR.runner4csl --tree_height 9  --max_ring_size 8 --disable_tqdm
python -m CaRFGNN.runner4csl --hilayers 2 --tree_height 5 --disable_tqdm
python -m GCaRFGNN.runner4csl --hilayers 2 --tree_height 5 --disable_tqdm
python -m CaRFGNNMRwR.runner4csl --hilayers 2 --tree_height 5  --max_ring_size 8 --disable_tqdm
python -m GCaRFGNNMRwR.runner4csl --hilayers 2 --tree_height 5  --max_ring_size 8 --disable_tqdm

python -m CaRFGNN.runner4csl --hilayers 3 --tree_height 5 --disable_tqdm
python -m GCaRFGNN.runner4csl --hilayers 3 --tree_height 5 --disable_tqdm
python -m CaRFGNNMRwR.runner4csl --hilayers 3 --tree_height 5  --max_ring_size 8 --disable_tqdm
python -m GCaRFGNNMRwR.runner4csl --hilayers 3 --tree_height 5  --max_ring_size 8 --disable_tqdm

python -m CaRFGNN.runner4csl --hilayers 2 --tree_height 9 --disable_tqdm
python -m GCaRFGNN.runner4csl --hilayers 2 --tree_height 9 --disable_tqdm
python -m CaRFGNNMRwR.runner4csl --hilayers 2 --tree_height 9  --max_ring_size 8 --disable_tqdm
python -m GCaRFGNNMRwR.runner4csl --hilayers 2 --tree_height 9  --max_ring_size 8 --disable_tqdm
