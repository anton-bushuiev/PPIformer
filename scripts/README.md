# Re-training PPIformer from scratch

The `run.py` script serves as a main entry point for training and validating PPIformer.
The script uses Hydra to manage configurations. Please see the [Hydra documentation](https://hydra.cc/) for details.

## 1. Installation (optional)

If you want to reproduce the experiments from the PPIformer [paper](https://arxiv.org/abs/2310.18515), we recommend installing the same environment as the one used to develop the models. The environment
is different from the deployment environment (specified in `PPIformer/requirements.txt`) since the deployment environment
relies on PyTorch 2.0 required for Hugging Face Spaces while the model was deveoped with PyTorch 1.13. To install the development environment please run:

```bash
# Create environment
conda create -n ppiformer_dev python=3.9.17
conda activate ppiformer_dev

# Install dependencies used for development
pip install -e git+https://github.com/anton-bushuiev/PPIRef.git#egg=ppiref
pip install --no-deps -e git+https://github.com/anton-bushuiev/mutils.git#egg=mutils
pip install git+https://github.com/anton-bushuiev/equiformer-pytorch.git@512dd15350d541804540514a713ac690649ea6a0
conda env update --file conda-environment.yaml

# Install PyTorch (please adapt the version suffix to your system, rocm5.2 is for AMD GPUs)
pip install -U torch==1.13.1+rocm5.2 --index-url https://download.pytorch.org/whl/rocm5.2

# Install PPIformer code
pip install --no-deps -e ../
```

## 2. Downloading preprocessed data (optional)

PPIformer uses the PPIRef package to get .pdb files containing protein-protein interactions for training. The .pdb files are then converted into PyTorch Geometric format using the `ppiformer.data` module as a part of the `run.py` script. To avoid re-building the datasets from .pdb files from scratch, you can download the pre-processed dataset from Zenodo using the following command:

```python
from ppiformer.utils.api import download_from_zenodo
download_from_zenodo('.pyg_dataset_cache.zip')
```

## 3. Pre-training

Pre-training of PPIformer is based on structural masked modeling performed on the PPIRef dataset.
You can run the pre-training experiment using the following command:

```bash
HYDRA_FULL_ERROR=1 python3 run.py \
    trainer.accelerator=gpu \
    trainer.devices=8 \
    job_key="${job_key}" \
    project_name=PRETRAINING \
    run_name='pre_training_iclr24' \
    experiment=pretraining \
    model.optimizer.lr=0.0005 \
    train_dataloader.batch_size=2 \
    val_dataloader.batch_size=2 \
    trainer.accumulate_grad_batches=16 \
    train_dataloader.dataset=\'ppiref_10A_filtered_clustered_03,whole\' \
    trainer.val_check_interval=0.1 \
    train_dataloader._transform.masked_modeling.mask_ratio=0.5 \
    train_dataloader._transform.masked_modeling.mask_sum=null \
    train_dataloader._transform.masked_modeling.bert=true \
    train_dataloader._transform.masked_modeling.same_chain=true \
    val_dataloader._transform.masked_modeling.mask_ratio=0.5 \
    val_dataloader._transform.masked_modeling.mask_sum=null \
    val_dataloader._transform.masked_modeling.bert=true \
    val_dataloader._transform.masked_modeling.same_chain=true \
    val_dataloader.dataset=\'skempi2_iclr24_split,0+1+2\' \
    model.label_smoothing=0.05 \
    model.class_weights=true \
    model.encoder.depth=8 \
    model.encoder.heads=2 \
    model.encoder.num_neighbors=10 \
    tags=null
```

As a result, the script will produce checkpoints selected based on validation performance in the `./<project_name>/<run_name>` directory and log the training process to the WandB project specified in the `project_name` parameter.

Notes on the arguments:

- You may need to adjust the first two parameters (`trainer.accelerator` and `trainer.devices`) to match your system configuration.
- The `job_key` is a unique identifier for the job. It is generate by the `jobs/submit_run_slurm.sh` script. You can ignore the argument if you are not using the Slurm submission script. The `job_key` value is used to identify logs located in the `jobs/submissions` directory.
- The `project_name` and `run_name` are the WandB project and run names, respectively.
- The `experiment` parameter specifies the experiment to run. In this case, it is `pretraining`. This parameter is used to load the stadanrd values for configuration from the `configs/experiments` directory.
- `batch_size` is set to 2 because of large memory requirements of Equiformer. The `accumulate_grad_batches` parameter is set to 16 to simulate a batch size of 32.
- The `train_dataloader.dataset` parameter specifies the dataset to use for training. The `ppiref_10A_filtered_clustered_03,whole` (a.k.a whole PPIRef50K) dataset is used for pre-training. If you download the `.pyg_dataset_cache.zip` from Zenodo (see step 2 above), you can use the pre-processed dataset without a need to re-build it from .pdb files. The same holds for the validation dataset `skempi2_iclr24_split,0+1+2`.

## 4. Fine-tuning

After pre-training, you can run the fine-tuning experiment on the ddG regression task using the following command:

```bash
HYDRA_FULL_ERROR=1 python3 run.py \
    trainer.accelerator=gpu \
    trainer.devices=8 \
    job_key="${job_key}" \
    run_name='fine_tuning_iclr24' \
    experiment=ddg_regression \
    project_name=DDG_REGRESSION \
    train_dataloader.dataset=\'skempi2_iclr24_split,0+1\' \
    train_dataloader.fresh=false \
    train_dataloader.dataset_max_workers=16 \
    train_dataloader._pretransform.ddg_label.df=null \
    val_dataloader.dataset=\'skempi2_iclr24_split,2\' \
    val_dataloader.fresh=false \
    val_dataloader.dataset_max_workers=16 \
    val_dataloader._pretransform.ddg_label.df=null \
    model.optimizer.lr=0.0003 \
    model.checkpoint_path=\'../weights/masked_modeling.ckpt\' \
    model.kind=masked_marginals \
    model.encoder.heads=2 \
    model.encoder.depth=8 \
    train_dataloader.batch_size=1 \
    val_dataloader.batch_size=1 \
    trainer.accumulate_grad_batches=32 \
    trainer.val_check_interval=1.0 \
    trainer.check_val_every_n_epoch=5 \
    train_dataloader._pretransform.ddg_label.strict_wt=True \
    cv=true \
    tags=null
```

Similar to pre-training, the script will produce checkpoints selected based on validation performance in the `./<project_name>/<run_name>` directory and log the training process to the WandB project specified in the `project_name` parameter.

Notes on the arguments:

- The `model.checkpoint_path` parameter specifies the path to the pre-trained model checkpoint. You can use the checkpoint produced during the pre-training experiment.
To use the prepared `../weights/masked_modeling.ckpt` checkpoint, you can download it from Zenodo using the following command:

    ```python
    from ppiformer.utils.api import download_from_zenodo
    download_from_zenodo('weights.zip')
    ```

- In the example, we use the `skempi2_iclr24_split` dataset for training and validation. The dataset is split into for parts: `0`, `1`, `2` and `test`. The first two parts are used for training, and the third part is used for validation. To perform the full cross-validation on the three training folds, you can set the `cv` parameter to `true` and run the script two more times with the `train_dataloader.dataset` and `val_dataloader.dataset` having the data fold part (string after comma) set to `0+2`, `1` and `1+2`, `0`. Then, in WandB, you can group the runs by the `run_name` and monitor the average, minimum and maximum perforamnce at the same time (similar to Figure 5B in the [paper](https://arxiv.org/abs/2310.18515)).
- The `_pretransform.ddg_label.df` arguments specify the path to the CSV file containing the ddG labels. In the example above, we use the `null` value to use the default labels from SKEMPI v2.0.

Next, you can use the three checkpoints produced by the cross validation as input to `PPIformer/notebooks/test.ipynb` to test the models (in the paper we always use three checkpoints selected by
per-PPI Spearman correlation produced by the three cross-validation runs).

## Using Slurm

The `./jobs` directory contains scripts for submitting jobs to a Slurm job scheduler. The `submit_run_slurm.sh` script is used to submit the `run.py` script to Slurm. For example, if you want to run the pre-training experiment on a Slurm cluster for 24 hours, you can copy the Python command from above to the `jobs/run_slurm.sh` script and run

```bash
cd jobs
./submit_run_slurm.sh 24:00
```

## Debugging and other examples

Debug pre-training locally with EGNN on CPU

```bash
python3 run.py experiment=debug_pretraining
```

Debug ratio-based masked modeling

```bash
python3 run.py experiment=debug_pretraining \
    train_dataloader._transform.masked_modeling.mask_ratio=0.2 \
    train_dataloader._transform.masked_modeling.mask_sum=null \
    val_dataloader._transform.masked_modeling.mask_ratio=0.2 \
    val_dataloader._transform.masked_modeling.mask_sum=null
```

Remove edge features for Equiformer

```bash
HYDRA_FULL_ERROR=1 python3 run.py \
    experiment=debug_ddg_regression \
    model/encoder=equiformer \
    model.pre_encoder_transform_kwargs.intra_inter_edge_features=false \
    model.encoder.num_edge_tokens=null \
    model.encoder.edge_dim=null
```

Apply BERT masking

```bash
HYDRA_FULL_ERROR=1 python3 run.py \
    experiment=debug_pretraining \
    train_dataloader._transform.masked_modeling.bert=true \
    val_dataloader._transform.masked_modeling.bert=true
```

Change training ddG labels

```bash
HYDRA_FULL_ERROR=1 python3 run.py \
    experiment=debug_ddg_regression \
    train_dataloader.fresh=true \
    train_dataloader._pretransform.ddg_label.strict_wt=False \
    train_dataloader._pretransform.ddg_label.df=\'/Users/anton/dev/mutils/data/SKEMPI2/augmentations/skempi_v2_permute_reverse.csv\'
```
