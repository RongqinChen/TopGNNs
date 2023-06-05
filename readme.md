
# Powerful Graph Topology Neural Networks: Capturing Local, Modelling Global

## Create Experiment Environment

[](create_environment.md)


## Reproducibility

For reproducibility, we should execute the following command:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Circular Skip Link (CSL) Experiments

```bash
bash csl_tasks.sh
```

## TUDatasets

### RFGNN
```bash
python -m pyscripts.run_tu_RFGNN
```

### LocalTopGNN
```bash
python -m pyscripts.run_tu_LocalTopGNN
```

### LongTopGNN
```bash
python -m pyscripts.run_tu_LongTopGNN
```


### GlobalTopGNN

```bash
python -m pyscripts.run_tu_GlobalTopGNN
```

## Peptides

### RFGNN

```bash
python -m pyscripts.run_pep_RFGNN
```

### LongTopGNN

```bash
python -m pyscripts.run_pep_LongTopGNN
```

### GlobalTopGNN

```bash
python -m pyscripts.run_pep_GlobalTopGNN
```
