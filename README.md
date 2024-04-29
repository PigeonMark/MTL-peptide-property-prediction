# Multi-task learning Peptide Property Prediction

## Setup environment
All required packages can be installed in a conda environment by running

    conda env create -f environment.yml

This will create an environment with the name MTL-pep-prop, activate by running

    conda activate MTL-pep-prop

If you want to use our data or our trained models, you should get them from [Zenodo](https://doi.org/10.5281/zenodo.11084463). 
Put the `data` and `lightning_logs` folders in the root directory.

In addition, recreating the plots requires the `dccs_results` and `prosit_results` folders.

## Making predictions
Making predictions for your data with one of our models can be done by running

    python predict.py

An example can be found in the file.
Make sure your data and PTMs are in the same format as e.g. `data/mtl_5fold_cv/all_data.csv`.
`prepare_data.py` can help you with this.


## Training a model
The models used in the study were trained by using the commands below. Get overview of all command line arguments by running

    python train.py -h

Note that there are 2 ways to give your data. You can use 1 datafile with all data and separate files to specify the 
indices for the train/val/test splits, this saves a lot of space for five-fold cross-validation or creating learning curves.
Alternatively, separate datafiles containing all columns can be given for the train/val/test splits.


### Train from scratch
    python train.py --config "mtl_5foldcv_supervised_none_0" -c --data-file data/mtl_5fold_cv/all_data.csv --train-i data/mtl_5fold_cv/train_0.csv --val-i data/mtl_5fold_cv/val_0.csv --test-i data/mtl_5fold_cv/test_0.csv --hpt-config hpt/mtl_hpt_supervised_none.csv --hpt-id 21 --vocab-file data/mtl_5fold_cv/all_data_vocab.p

### Pretrain
    python train.py --config "mtl_5foldcv_pretrain_0" -m pretrain -c --data-file data/mtl_5fold_cv/all_data.csv --train-i data/mtl_5fold_cv/train_0.csv --val-i data/mtl_5fold_cv/val_0.csv --hpt-config hpt/mtl_hpt_pretrain.csv --hpt-id 7 --hidden-size 180 --num-layers 9 --vocab-file data/mtl_5fold_cv/all_data_vocab.p

### Fine-tune own
    python train.py --config "mtl_5foldcv_finetune_own_0" -p own --checkpoint-id 0 -c --data-file data/mtl_5fold_cv/all_data.csv --train-i data/mtl_5fold_cv/train_0.csv --val-i data/mtl_5fold_cv/val_0.csv --test-i data/mtl_5fold_cv/test_0.csv --hpt-config hpt/mtl_hpt_finetune_own.csv --hpt-id 27 --vocab-file data/mtl_5fold_cv/all_data_vocab.p

### Fine-tune TAPE
    python train.py --config "mtl_5foldcv_supervised_none_0" -p tape -c --data-file data/mtl_5fold_cv/all_data.csv --train-i data/mtl_5fold_cv/train_0.csv --val-i data/mtl_5fold_cv/val_0.csv --test-i data/mtl_5fold_cv/test_0.csv --hpt-config hpt/mtl_hpt_finetune_tape.csv --hpt-id 25 --vocab-file data/mtl_5fold_cv/all_data_vocab.p


## Making the plots and statistical tests
The plots were made by running

    python plot.py

The statistical tests for difference between the performance results were performed by running

    python calculate_metrics.py