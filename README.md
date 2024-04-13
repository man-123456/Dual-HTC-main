# Dual Contrastive Learning for Hierarchical Text Classification

## Requirements

* Python = 3.7
* torch = 1.9.0
* transformers == 4.2.1
* fairseq = 0.10.0
* torch-geometric == 1.7.2
* torch-scatter == 2.0.8
* torch-sparse ==  0.6.12

## Preprocess

Please download the original dataset and then use these scripts.

### WebOfScience

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocess code could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy of preprocess code here.
Please save the Excel data file `Data.xlsx` in `WebOfScience/Meta-data` as `Data.txt`.

```shell
cd ./data/WebOfScience
python preprocess_wos.py
python data_wos.py
```

## Train

```
usage: train.py [-h] [--lr LR] [--data {WebOfScience,nyt,rcv1}] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] --name NAME [--update UPDATE] [--warmup WARMUP] [--contrast CONTRAST] [--graph GRAPH] [--layer LAYER]
                [--multi] [--lamb LAMB] [--lamb2 LAMB2] [--thre THRE] [--tau TAU] [--seed SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Learning rate.
  --data {WebOfScience,nyt,rcv1}
                        Dataset.
  --batch BATCH         Batch size.
  --early-stop EARLY_STOP
                        Epoch before early stop.
  --device DEVICE		cuda or cpu. Default: cuda
  --name NAME           A name for different runs.
  --update UPDATE       Gradient accumulate steps
  --warmup WARMUP       Warmup steps.
  --contrast CONTRAST   Whether use contrastive model. Default: True
  --graph GRAPH         Whether use graph encoder. Default: True
  --layer LAYER         Layer of Graphormer.
  --multi               Whether the task is multi-label classification. Should keep default since all 
  						datasets are multi-label classifications. Default: True
  --lamb LAMB           lambda
  --lamb2 LAMB           lambda2
  --thre THRE           Threshold for keeping tokens. Denote as gamma in the paper.
  --tau TAU             Temperature for contrastive model.
  --seed SEED           Random seed.
  --wandb               Use wandb for logging.
```

Checkpoints are in `./checkpoints/DATA-NAME`. Two checkpoints are kept based on macro-F1 and micro-F1 respectively 
(`checkpoint_best_macro.pt`, `checkpoint_best_micro.pt`).

e.g. Train on `WebOfScience` with `batch=12, lambda=0.05, gamma=0.02`. Checkpoints will be in `checkpoints/WebOfScience-test/`.

```shell
python train.py --name test --batch 12 --data WebOfScience --lambda 0.05 --thre 0.02
```

### Reproducibility

Contrastive learning is sensitive to hyper-parameters. We report results with fixed random seed but we observe higher results with unfixed seed.

* The results reported in the main table can be observed with following settings under `seed=3`.

```
WOS: lambda 0.05 thre 0.02
NYT: lambda 0.3 thre 0.002
RCV1: lambda 0.3 thre 0.001
```

We experiment on GeForce RTX 3090 (24G) with CUDA version $11.1$.

* The following settings can achieve higher results with unfixed seed (which we reported in the paper) .

```
WOS: lambda 0.1 thre 0.02
```

* We also find that a higher `tau` (e.g. `tau=2`) is beneficial but we keep it to $1$ for simplicity.

## Test

```
usage: test.py [-h] [--device DEVICE] [--batch BATCH] --name NAME [--extra {_macro,_micro}]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --batch BATCH         Batch size.
  --name NAME           Name of checkpoint. Commonly as DATA-NAME.
  --extra {_macro,_micro}
                        An extra string in the name of checkpoint. Default: _macro
```

Use `--extra _macro` or `--extra _micro`  to choose from using `checkpoint_best_macro.pt` or`checkpoint_best_micro.pt` respectively.

e.g. Test on previous example.

```shell
python test.py --name WebOfScience-test
```
