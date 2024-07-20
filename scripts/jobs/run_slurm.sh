#!/bin/bash

# Example run:
# ./submit_run_slurm.sh 24:00

echo "job_key \"${job_key}\""

# Prepare project environment
. "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate ppiformer_dev
export SLURM_GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

# Pre-train
srun --export=ALL --preserve-env python3 ../run.py \
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
    val_dataloader.fresh=false \
    val_dataloader.dataset_max_workers=8 \
    model.label_smoothing=0.05 \
    model.class_weights=true \
    model.encoder.depth=8 \
    model.encoder.heads=2 \
    model.encoder.num_neighbors=10 \
    tags=null

# ddG fine-tuning
# srun --export=ALL --preserve-env python3 ../run.py \
#     trainer.accelerator=gpu \
#     trainer.devices=8 \
#     job_key="${job_key}" \
#     run_name='fine_tuning_iclr24' \
#     experiment=ddg_regression \
#     project_name=DDG_REGRESSION \
#     train_dataloader.dataset=\'skempi2_iclr24_split,0+1\' \
#     train_dataloader.fresh=false \
#     train_dataloader.dataset_max_workers=16 \
#     train_dataloader._pretransform.ddg_label.df=null \
#     val_dataloader.dataset=\'skempi2_iclr24_split,2\' \
#     val_dataloader.fresh=false \
#     val_dataloader.dataset_max_workers=16 \
#     val_dataloader._pretransform.ddg_label.df=null \
#     model.optimizer.lr=0.0003 \
#     model.checkpoint_path=\'../../weights/masked_modeling.ckpt\' \
#     model.kind=masked_marginals \
#     model.encoder.heads=2 \
#     model.encoder.depth=8 \
#     train_dataloader.batch_size=1 \
#     val_dataloader.batch_size=1 \
#     trainer.accumulate_grad_batches=32 \
#     trainer.val_check_interval=1.0 \
#     trainer.check_val_every_n_epoch=5 \
#     train_dataloader._pretransform.ddg_label.strict_wt=True \
#     cv=true \
#     tags=null
