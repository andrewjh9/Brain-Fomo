##  Improving Large Brain Models with Pre-training and PEFT

A code base for a project submitted as part of the **UvA Foundation Models Course 2025**.

This work builds upon the paper:  
 _[Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP)_  
Original codebase: [LaBraM GitHub](https://github.com/935963004/LaBraM)

---

##  Abstract

Inspired by recent breakthroughs of Large Language Models (LLMs), Jiang et al. (2024) introduced a Large Brain Model (LaBraM) to address the scaling limitations of dataset-specific deep learning (DL) models for EEG. LaBraM learns a universal EEG representation by self-supervised pre-training on more than 2,500 hours of EEG data. 

In this study, we replicate and extend LaBraM in several ways:

- We validate LaBraM’s reported performance on three benchmark datasets.
- We test generalizability to the small-scale BCI-IV2a dataset.
- We apply **Parameter-Efficient Fine-Tuning (PEFT)** techniques.
- We explore **Residual Vector Quantization (RVQ)** to improve tokenizer performance.

---

##  PEFT Experiments

This section covers **replication and extension** of LaBraM using PEFT techniques such as LoRA, AdaLoRA, and BitFit across TUEV, TUAB, and SEED-V.

###  To replicate PEFT results, run:

```bash
for DATASET in TUEV TUAB SEED-V; do
  for FLAG in --use_lora --use_adalora --use_bitfit; do
    echo "[$(date)] Starting Dataset ${DATASET}, Flag ${FLAG}"

    OUTPUT_DIR="/path/to/checkpoints/finetune_${DATASET}_${FLAG#--}/"
    LOG_DIR="/path/to/logs/finetune_${DATASET}_${FLAG#--}"

      srun torchrun "$FILEPATH" \
      --output_dir "$OUTPUT_DIR" \
      --log_dir "$LOG_DIR" \
      --model labram_base_patch200_200 \
      --finetune /path/to/pretrained/labram-base.pth \
      --weight_decay 0.05 \
      --batch_size 64 \
      --lr 5e-4 \
      --update_freq 1 \
      --warmup_epochs 5 \
      --epochs 50 \
      --layer_decay 0.65 \
      --drop_path 0.1 \
      --save_ckpt_freq 5 \
      --disable_rel_pos_bias \
      --abs_pos_emb \
      --dataset "$DATASET" \
      --disable_qkv_bias \
      --num_workers 8 \
      --seed 0 \
      "$FLAG"
  done
done
```

---

##  Residual Vector Quantization (RVQ) Experiments

We experimented with replacing LaBraM’s single-stage tokenizer with multi-stage RVQ to compress EEG representations more effectively.

###  Key flags:

To use RVQ in tokenizer training and fine-tuning:

```bash
--res_quant
--num_quantizers [int]
```

Once trained, add `--num_quantizers` to downstream training commands.  
See the `residual_vector_quantization/` and `jobs/` folders for example scripts.

---

##  BCI Competition IV 2a

This section contains **replication scripts for LaBraM and EEGNet** on the BCI-IV2a motor imagery task dataset.

### EEGNet

#### ➔ Preprocess data:

```bash
python BCI-IV2a/EEGNet/make_BCI_IV2a.py
```

#### ➔ Run experiments:

```bash
python BCI-IV2a/EEGNet/BCI_IV2a_LOSO.py
python BCI-IV2a/EEGNet/BCI_IV2a_k_fold.py
```

###  LaBraM (LOSO & K-Fold)

#### ➔ Preprocess data:

```bash
python BCI-IV2a/LaBraM/dataset_maker/make_BCI_IV2a_LOSO.py
python BCI-IV2a/LaBraM/dataset_maker/make_BCI_IV2a_k_fold.py
```

#### ➔ Run Leave-One-Subject-Out (LOSO):

```bash
for SUBJECT_ID in {1..9}; do
  srun torchrun --nnodes=1 --nproc_per_node=1 run_class_finetuning.py \
    --output_dir /path/to/checkpoints/finetune_BCI_IV2a_base-subject_${SUBJECT_ID}/ \
    --log_dir /path/to/logs/finetune_BCI_IV2a_base-subject_${SUBJECT_ID} \
    --data_path /path/to/data/BCI_IV2a_LOSO/Subject_${SUBJECT_ID} \
    [other LaBraM args]
done
```

#### ➔ Run 3-Fold Cross-Validation:

```bash
for SUBJECT_ID in {1..9}; do
  for FOLD_ID in $(seq 0 2); do
    echo "[$(date)] Starting Subject ${SUBJECT_ID}, Fold ${FOLD_ID}"
    srun torchrun run_class_finetuning.py \
      --data_path /path/to/data/BCI_IV2a_K_FOLD/Subject_${SUBJECT_ID}/fold_${FOLD_ID} \
      --output_dir /checkpoints/... \
      --log_dir /logs/... \
      [other LaBraM args]
  done
done
```

---

## SEED-V Dataset

This folder contains code to replicate the LaBraM results on the **SEED-V emotion classification dataset**.

### Preprocessing:

```bash
python SEED-V/LaBraM/dataset_maker/make_SEED-V.py
```

### Fine-tuning:

```bash
srun torchrun run_class_finetuning.py \
  --dataset SEED-V \
  --model labram_base_patch200_200 \
  [other training arguments]
