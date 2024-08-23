
import argparse
import logging
import math
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
import random
from torch.utils.tensorboard import SummaryWriter
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
import torch
import numpy as np
import pandas as pd
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    AdamW,
    AutoConfig,
    PretrainedConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
########## setup PEFT #################################
from peft import get_peft_config, get_peft_model, LoraConfig, IA3Config,TaskType
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PeftConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    prepare_model_for_int8_training,
    # AutoPeftModel,
    prepare_model_for_kbit_training # only for latest dev version of peft
)
from modeling_bert import BertForMultilabelClassification
from modeling_roberta import RobertaForCombinedSequenceClassification
from modeling_longformer import LongformerForMultilabelClassification
from evaluation import all_metrics, compute_metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from scipy.special import softmax
from utils import get_model_save_name, get_dataset_directory_details, CommonLogger, TensorBoardLogger, WandbLogger, unfreeze_model, count_trainable_parameters

import pickle

logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help=(
            "The size of chunks that we'll split the inputs into"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="The type of model",
        required=True,
        choices=["bert", "roberta", "longformer"]
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        help="Specify how to aggregate output in the model",
        required=True,
        choices=["cls-sum", "cls-max", "laat", "laat-split"]
    )
    
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the results and plots.")
    parser.add_argument("--few_shot_n",
                        type=int,
                        default = None,
                        help = "Number of samples per class to use for few shot learning")
    parser.add_argument("--metric_logger",
                        type=str,
                        default = "tensorboard",
                        help = "How to log metrics - either tensorboard or wandb")
    parser.add_argument("--apply_lora",
                        action="store_true",
                        help="If passed, will apply lora to the model")
    parser.add_argument("--run_training_set",
                        action="store_true",
                        help="If passed, will apply lora to the model")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    # add additioanl dataset details
    args = get_dataset_directory_details(args)
    
    # set model classes
    MODELS_CLASSES = {
    'bert': BertForMultilabelClassification,
    'roberta': RobertaForCombinedSequenceClassification,
    'longformer': LongformerForMultilabelClassification
    }
       


    # get model name for saving
    # model_name = get_model_save_name(args.model_name_or_path)
    if "few_shot" in args.model_name_or_path:
        model_name = args.model_name_or_path.split("/")[6]
    else:
        if "all_data" not in args.model_name_or_path:
            model_name = args.model_name_or_path.split("/")[6] + "_all_data"
        else:
            model_name = args.model_name_or_path.split("/")[6]
        
    # we handle loading of lora models differently
    if args.apply_lora or "lora" in args.model_name_or_path or "LORA" in args.model_name_or_path:
        
        print(f"Got lora model!!! from path: {args.model_name_or_path}")
        
        
        # as this was trained on GPU instance, we need to set the local roberta-base-cris manually as its different local folder
        local_cris_roberta_path = "F:/31-CHRONOSIG_P2_OHFT/Experiments/language_modelling/mlm_only/transformers/roberta-base-cris/sampled_250000/24-04-2023--16-57/checkpoint-100000/"

        # config = AutoConfig.from_pretrained(local_cris_roberta_path, num_labels=5, finetuning_task="ohft_accepted_triage_subset_concat")


        # load peft config
        peft_config = PeftConfig.from_pretrained(args.model_name_or_path)
        config = AutoConfig.from_pretrained(args.model_name_or_path, finetuning_task="ohft_accepted_triage_subset_concat",)
        print(f"Config is: {config}")
        # load base model using the configs model_name_or_path
        model_class = MODELS_CLASSES[config.model_type]
        original_model = model_class.from_pretrained(
                # local_cris_roberta_path,
                peft_config.base_model_name_or_path,
                config=config,
            )
        # # sanity check the weights of a couple of attention layers and linear
        # org_layer_5_weights = original_model.roberta.encoder.layer[5].attention.self.query.weight.detach().clone()
        # org_layer_0_weights = original_model.roberta.encoder.layer[0].attention.self.query.weight.detach().clone()
        # print(f"Original_layer_0_weights: {org_layer_0_weights}")
        # print(f"Original_layer_5_weights: {org_layer_5_weights}")

        peft_model = PeftModel.from_pretrained(original_model, args.model_name_or_path)
        # print(f"PEFT model loaded: {peft_model}")
        # now merge the lora weights
        model = peft_model.merge_and_unload()
        # model = peft_model
        # unloaded_layer_5_weights = model.roberta.encoder.layer[5].attention.self.query.weight.detach().clone()
        # unloaded_layer_0_weights = model.roberta.encoder.layer[0].attention.self.query.weight.detach().clone()
        # print(f"unloaded_layer_0_weights: {unloaded_layer_0_weights}")
        # print(f"unloaded_layer_5_weights: {unloaded_layer_5_weights}")
        
        # assert not torch.equal(org_layer_0_weights, unloaded_layer_0_weights), "the reloaded weights should be different!"
        # assert not torch.equal(org_layer_5_weights, unloaded_layer_5_weights), "the reloaded weights should be different!"
    else:

        config = AutoConfig.from_pretrained(args.model_name_or_path,
                                            finetuning_task=args.task_name)
        # load in model
        model_class = MODELS_CLASSES[config.model_type]

        model = model_class.from_pretrained(
                args.model_name_or_path,
                config=config,
            )
    
    print(model)
        
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path,
                use_fast=True,
                )
    
    # set chunk size - need to do this dynamically based on model really
    chunk_size = args.chunk_size
    
    # make output dir if not existing
    save_path = f"{args.output_dir}/{args.task_name}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    
    
    
    raw_datasets = load_dataset("csv", 
                    data_files = {"train":f"{args.training_data_dir}/{args.training_file}",
                                    "valid":[f"{args.eval_data_dir}/{args.validation_file}",
                                             f"{args.eval_data_dir}/{args.test_file}"],
                                    },
                    cache_dir = None)    
    

    # check if label there, if not replace
    #NOTE - refactor this crap to use a yaml file
    if "label" not in raw_datasets["train"].column_names:
        raw_datasets = raw_datasets.rename_column(args.label_name, "label")
        # if args.task_name == "mimic-mp":
        #     raw_datasets["train"] = raw_datasets["train"].rename_column("hospital_expire_flag", "label")
        #     raw_datasets["validation"] = raw_datasets["validation"].rename_column("hospital_expire_flag", "label")
        # elif args.task_name == "mimic-los":
        #     raw_datasets["train"] = raw_datasets["train"].rename_column("los_label", "label")
        #     raw_datasets["validation"] = raw_datasets["validation"].rename_column("los_label", "label")
    # get num labels

    num_labels = len(np.unique(raw_datasets['train']['label']))
    
    # if we are doing few shot - we need to sample the training data
    if args.few_shot_n is not None:
        logger.info(f"Sampling {args.few_shot_n} samples per class")
        train_datasets = []
        for label in range(num_labels):
            label_dataset = raw_datasets['train'].filter(lambda x: x['label'] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= args.few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(args.few_shot_n))
            
            # add to list of datasets
            train_datasets.append(label_dataset)

        raw_datasets["train"] = concatenate_datasets(train_datasets)
        
    # define preprocess function
    sentence1_key, sentence2_key = "text", None
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=args.max_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    # remove_columns = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=None
    )
    
    def data_collator(features):
        # print(f"featurues:{features}")
        batch = dict()
        # print(f"Features inside data_collator: {len(features[0]['input_ids'])}")
        if "cls" in config.model_mode:
            for f in features:
                new_input_ids = []
                for i in range(0, len(f["input_ids"]), chunk_size - 2):
                    new_input_ids.extend([tokenizer.cls_token_id] + f["input_ids"][i:i+(chunk_size)-2] + [tokenizer.sep_token_id])
                f["input_ids"] = new_input_ids
                f["attention_mask"] = [1] * len(f["input_ids"])
                f["token_type_ids"] = [0] * len(f["input_ids"])

        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % chunk_size != 0:
            max_length = max_length - (max_length % chunk_size) + chunk_size

        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, chunk_size))
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([
                f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
                for f in features
            ]).contiguous().view((len(features), -1, chunk_size))
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([
                f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
                for f in features
            ]).contiguous().view((len(features), -1, chunk_size))

        batch["labels"] = torch.tensor([features[0]["labels"]])
        # add other features
        if "num_instance" in features[0]:
            batch["num_instance"] = torch.tensor([features[0]["num_instance"]])
            
        if "brc_id" in features[0]:
            
            batch["brc_id"] = [features[0]["brc_id"]]
        
        if "accepted_triage_team" in features[0]:
            batch["accepted_triage_team"] = [features[0]["accepted_triage_team"]]
        return batch
    
    # setup dataloaders
    train_dataloader = DataLoader(processed_datasets["train"], shuffle = False, batch_size=1, collate_fn=data_collator)
    eval_dataloader = DataLoader(processed_datasets["valid"], shuffle = False, batch_size=1, collate_fn=data_collator)
    
    
    # use accelerator package to setup device
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # # Prepare everything with our `accelerator`.
    # model,  eval_dataloader = accelerator.prepare(
    #     model, eval_dataloader
    # )
    
    # function to extract doc embeds
    def get_doc_embeds(batch, model_outputs):
        
        attn_mask = batch["attention_mask"].view(-1)
        last_hidden_state = model_outputs["hidden_states"][-1].view(-1,768).cpu()
        avg_hidden_state = torch.mean(last_hidden_state * attn_mask.unsqueeze(1), dim = 0)
        return avg_hidden_state
    
    def run_evaluation(model, 
                       dataloader,
                       split):
        
        print(f"Running pipeline on: {split}")

        # # double check eval metrics
        # set model to cuda
        model.to("cuda")
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        all_label_attentions = []
        all_embeddings = []
        all_input_ids = []
        all_results = {}
        probs_dfs = []
        results_dfs = []
        for step, batch in tqdm(enumerate(dataloader)):
            # print(f"Batch: {batch.keys()}")
                    
            with torch.no_grad():
                outputs = model(input_ids = batch["input_ids"].cuda(), 
                        attention_mask = batch["attention_mask"].cuda(),
                        labels = batch["labels"].cuda(), 
                        output_hidden_states = True)
                embeds = get_doc_embeds(batch, outputs)
            # apply softmax to the logits of the output - using the softmax function
            preds_raw = outputs.logits.softmax(dim=-1).cpu()           

            
            # get argmax of preds raw
            preds = np.argmax(preds_raw, axis = -1)             
            
            all_preds_raw.extend(list(preds_raw.cpu().numpy()))
            all_preds.extend(list(preds.cpu().numpy()))
            all_labels.extend(list(batch["labels"].cpu().numpy()))
            all_embeddings.append(embeds.cpu().numpy())
            
            # extract brc_id and num_instance
            
            
            # print(f'brc_id:{batch["brc_id"]} \n')
            # print(f"preds: {preds} \n\n")
            # print(f"preds_raw: {preds_raw} \n\n")
            # print(f"num instance: {batch['num_instance']}")
            results_df = pd.DataFrame({"brc_id":batch["brc_id"],
                                "num_instance":batch["num_instance"].detach().cpu(),                 
                                "probs":[preds_raw.squeeze(0).detach().cpu().numpy()],
                                "labels":batch["labels"].cpu().numpy(),
                                "preds":preds, 
                                })
            
            results_dfs.append(results_df)
            
            # get the attentions too
            label_attentions = outputs.label_attentions.squeeze(0).detach().cpu().numpy()
            # print(f"label attnetions shape: {label_attentions.shape}")
            all_label_attentions.append(label_attentions)
            
            # get reformed input_ids
            batch_size, num_chunks, chunk_size = batch["input_ids"].shape
            
            # get the attention mask to remove pad tokens from reformed input ids
            attention_mask = batch["attention_mask"].view(batch_size, num_chunks*chunk_size, -1).squeeze(-1)

            reformed_input_ids = batch["input_ids"].view(batch_size, num_chunks*chunk_size, -1).squeeze(-1)
            
            # now only get ones with full attention i.e. not pad tokens
            reformed_input_ids = reformed_input_ids[attention_mask==1].unsqueeze(0)

            reformed_input_ids = reformed_input_ids.cpu().numpy()
            all_input_ids.append(reformed_input_ids)
            
            
        all_dfs = pd.concat(results_dfs)
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        all_embeddings = np.array(all_embeddings)
        # print(f"all_preds_raw shape is: {all_preds_raw.shape}")
        # print(f"all_preds shape is: {all_preds.shape} \n\n {all_preds}")
        # print(f"all_labels shape is: {all_labels.shape} \n\n {all_labels}")
        # metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        # use compute metrics with args: predictions, pred_scores, labels
        metrics = compute_metrics(all_preds, all_preds_raw, all_labels)
        
        print(f"############Metrics are:\n {metrics}\n##################")
        
        # save metrics to file 
        with open(f'{save_path}/{model_name}_long_instance_{split}_results.pickle', 'wb') as f:
            pickle.dump(metrics, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        # save all_dfs too
        all_dfs.to_csv(f"{save_path}/{model_name}_{split}_results_df.csv", index = False)
        
        # save the label attentions and input ids too?
        with open(f'{save_path}/{model_name}_{split}-label-attentions.pickle', 'wb') as f:
            pickle.dump(all_label_attentions, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        with open(f'{save_path}/{model_name}_{split}-reformed-input-ids.pickle', 'wb') as f:
            pickle.dump(all_input_ids, f, protocol = pickle.HIGHEST_PROTOCOL)
            
        with open(f'{save_path}/{model_name}_{split}_embeddings.pickle', 'wb') as f:
            pickle.dump(all_embeddings, f, protocol = pickle.HIGHEST_PROTOCOL)
            
    
    # if run for training
    if args.run_training_set:
        run_evaluation(model, 
                       train_dataloader,
                       split = "train")
        
    
    # always run eval set
    run_evaluation(model,
                   eval_dataloader,
                   split = "eval")
            
            
if __name__ == "__main__":
    main()
