# Instructions for creating the experiment environment

## Create virtual environment with CUDA 11.3

```bash
conda create --name gnn113 python==3.9 \
pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 pyg dgl-cuda11.3 \
rdkit graph-tool pip pygraphviz \
-c pyg -c pytorch -c dglteam -c nvidia -c conda-forge -c rdkit
```


```bash
source ~/anaconda3/bin/activate gnn113

pip install Cython tensorboard networkx ogb pydot matplotlib lmdb

bash transform_fns/tpf/build.sh
```

Make results Reproducibile
```bash
# set the environment variable
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
