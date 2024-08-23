
# DISCLAIMER
This repository is a work in progress and is not yet ready for public use. We are working on cleaning up the code and adding more documentation. Please check back later for updates.
# OHFT_triage
This repo provides the skeleton code to train and evaluate 

## Reference
Please cite the following paper:
```
    
```


## Requirements
* Python >= 3.6
* Install the required Python packages with `pip3 install -r requirements.txt`
* If the specific versions could not be found in your distribution, you could simple remove the version constraint. Our code should work with most versions.

## Dataset
Unfortunately, we are not allowed to redistribute the NHS dataset. 

### Training
1. `cd src`
2. Run the following command to train a model on MIMIC-3 full.
```
python3 run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/dev_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/RoBERTa-base-PM-M3-Voc-distill-align-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 20 \
    --num_warmup_steps 2000 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat
```

### Notes
- If you would like to train BERT-based or Longformer-base models, please set `--model_type [bert|longformer]`.
- If you would like to train models on MIMIC-3 top-50, please set `--code_50 --code_file ../data/mimic3/ALL_CODES_50.txt`
- If you would like to train models on MIMIC-2, please set `--code_file ../data/mimic2/ALL_CODES.txt`

### Inference
1. `cd src`
2. Run the following command to evaluate a model on the test set of MIMIC-3 full.
```
python3 run_icd.py \
    --train_file ../data/mimic3/train_full.csv \
    --validation_file ../data/mimic3/test_full.csv \
    --max_length 3072 \
    --chunk_size 128 \
    --model_name_or_path ../models/roberta-mimic3-full \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 0 \
    --output_dir ../models/roberta-mimic3-full \
    --model_type roberta \
    --model_mode laat
```
