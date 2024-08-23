# coding=utf-8

""" Finetuning segment batch approach for sequence classification."""
import argparse
import logging
import math
import os
import random
import wandb
from torch.utils.tensorboard import SummaryWriter
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets, DatasetDict
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from modeling_bert import BertForMultilabelClassification
from modeling_roberta import RobertaForCombinedSequenceClassification
from modeling_longformer import LongformerForMultilabelClassification
from evaluation import all_metrics, compute_metrics
from scipy.special import softmax
from utils import get_model_save_name, get_dataset_directory_details, CommonLogger, TensorBoardLogger, WandbLogger, unfreeze_model, count_trainable_parameters
# os.environ["WANDB_MODE"] = "offline"
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

logger = logging.getLogger(__name__)


MODELS_CLASSES = {
    'bert': BertForMultilabelClassification,
    'roberta': RobertaForCombinedSequenceClassification,
    'longformer': LongformerForMultilabelClassification
}


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
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--cased",
        action="store_true",
        help="equivalent to do_lower_case=False",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
    parser.add_argument("--run_full_eval",
                        action="store_true",
                        help="If passed, run final evaluation on all data and save bunch of things.")
    parser.add_argument("--triage_team",
                        type = str,
                        default = None,
                        help = "the specific triage team to train on for the binary triage accept/reject task")
    parser.add_argument("--debug_run",
                        action = "store_true",
                        )
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
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.info(f"Arguments after getting dataset details are: {args}")
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    
    
    # get the save model name
    model_name = get_model_save_name(args.model_name_or_path)

    
    #TODO if we are looking at the binary reject/accept task - we need to grab the specific team
    if args.task_name == "triage_accept_reject_concat":
        args.training_data_dir = f"{args.training_data_dir}/{args.triage_team}/"
        args.eval_data_dir = f"{args.eval_data_dir}/{args.triage_team}/"
        args.task_name = f"triage_accept_reject_concat_{args.triage_team}"
        logger.info(f"Will be training binary triage accept reject for team: {args.triage_team}")
    
    # if apply lora change here
    if args.apply_lora:
        logger.warning(f"Applying LORA to model!")
        model_name = f"{model_name}-lora"    
   
    if args.few_shot_n is not None:
        args.output_dir = f"{args.output_dir}/{args.task_name}/{model_name}-{args.chunk_size}/few_shot_{args.few_shot_n}/"
    else:
        args.output_dir = f"{args.output_dir}/{args.task_name}/{model_name}-{args.chunk_size}/"
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        
    logger.info(f"Saving model to {args.output_dir}")
    
    # set up tensorboard or wandb
    if args.metric_logger == "tensorboard":
        logger.warning(f"Using tensorboard!")
        # setup tensorboard
        # writer = SummaryWriter(log_dir=args.output_dir)
        metric_logger = CommonLogger(TensorBoardLogger(log_dir=args.output_dir))
    elif args.metric_logger == "wandb":
        # setup wandb
        logger.warning(f"Using wandb!")
        metric_logger = CommonLogger(WandbLogger(project=f"long-sequence",                      
                        config=args,
                        job_type="train",
                        # change if few_shot_n is not None
                        name = f"{args.task_name}-{model_name}-{args.chunk_size}" if args.few_shot_n is None else f"{args.task_name}-{model_name}-{args.chunk_size}-few_shot_{args.few_shot_n}",
                        #  name=f"{args.task_name}-{model_name}",
                        dir = args.output_dir,))
        

    else:
        raise ValueError(f"Metric logger {args.metric_logger} not supported")
    logger.info(f"Loading dataset for task: {args.task_name}")
    

        
    raw_datasets = load_dataset("csv", 
                    data_files = {"train":f"{args.training_data_dir}/{args.training_file}",
                                    "validation":f"{args.eval_data_dir}/{args.validation_file}",
                                    "test":f"{args.eval_data_dir}/{args.test_file}"},
                    cache_dir = None)
    

    
    # take random sample of     
    
    print(f"Raw datasets: {raw_datasets}")
    # check if label there, if not replace
    #NOTE - refactor this crap to use a yaml file
    if "label" not in raw_datasets["train"].column_names:
        raw_datasets = raw_datasets.rename_column(args.label_name, "label")
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
    
    if args.debug_run:
        logger.warning(f"RUNNING DEBUG MODE")
        # take sample of validation and test
        val_datasets = []
        for label in range(num_labels):
            label_dataset = raw_datasets['validation'].filter(lambda x: x['label'] == label).shuffle(seed=42)
            num_samples = len(label_dataset)
            # if we have more samples than the few shot n - then we need to sample
            if num_samples >= args.few_shot_n:

                # select num_samples_per_class samples from the label
                label_dataset = label_dataset.select(range(args.few_shot_n))
            
            # add to list of datasets
            val_datasets.append(label_dataset)

        raw_datasets["validation"] = concatenate_datasets(val_datasets)
        raw_datasets["test"] = concatenate_datasets(val_datasets)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if args.model_type == "longformer":
        config.attention_window = args.chunk_size
    elif args.model_type in ["bert", "roberta"]:
        config.model_mode = args.model_mode
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        do_lower_case=not args.cased)
    model_class = MODELS_CLASSES[args.model_type]
    if args.num_train_epochs > 0:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        model = model_class.from_pretrained(
            args.output_dir,
            config=config,
        )
        
    # if we apply lora - then we need to get the config and model
    if args.apply_lora:
        peft_config = LoraConfig(task_type=None,
                        #  target_modules = ["query", "key", "value"],
                                    inference_mode=False,
                                    r=8,
                                    lora_alpha=0.8,
                                    lora_dropout=0.1,
                                    modules_to_save = ["first_linear","second_linear","third_linear"], #NOTE - this requires a janky fix in the forward pass of modelling_roberta
                            )
        model = get_peft_model(model, peft_config)
        logger.info(f"params before unfreezing linear layers:")
        model.print_trainable_parameters()
        
        # now unfreeze the linear layers
        # unfreeze the final linear layers
        unfreeze_model(model.base_model.model.first_linear)
        unfreeze_model(model.base_model.model.second_linear)
        unfreeze_model(model.base_model.model.third_linear)
        logger.info(f"params before unfreezing linear layers:")
        model.print_trainable_parameters()
        
        model.config.architectures = [model_class.__name__]
    print(f"Model: {model}")
    
    sentence1_key, sentence2_key = "text", None

    # label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = False

    sentence1_key, sentence2_key = "text", None
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result


    remove_columns = raw_datasets["train"].column_names if args.train_file is not None else raw_datasets["validation"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=None
    )

    print(f"Preprocessed datasets before passing to trainer: {processed_datasets}")
    eval_dataset = processed_datasets["validation"]    

    if args.num_train_epochs > 0:
        train_dataset = processed_datasets["train"]
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            logger.info(f"Original tokens: {tokenizer.decode(train_dataset[index]['input_ids'])}")

    def data_collator(features):
        batch = dict()
        # print(f"Features inside data_collator: {len(features[0]['input_ids'])}")
        if "cls" in args.model_mode:
            for f in features:
                new_input_ids = []
                for i in range(0, len(f["input_ids"]), args.chunk_size - 2):
                    new_input_ids.extend([tokenizer.cls_token_id] + f["input_ids"][i:i+(args.chunk_size)-2] + [tokenizer.sep_token_id])
                f["input_ids"] = new_input_ids
                f["attention_mask"] = [1] * len(f["input_ids"])
                f["token_type_ids"] = [0] * len(f["input_ids"])

        max_length = max([len(f["input_ids"]) for f in features])
        if max_length % args.chunk_size != 0:
            max_length = max_length - (max_length % args.chunk_size) + args.chunk_size

        batch["input_ids"] = torch.tensor([
            f["input_ids"] + [tokenizer.pad_token_id] * (max_length - len(f["input_ids"]))
            for f in features
        ]).contiguous().view((len(features), -1, args.chunk_size))
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([
                f["attention_mask"] + [0] * (max_length - len(f["attention_mask"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([
                f["token_type_ids"] + [0] * (max_length - len(f["token_type_ids"]))
                for f in features
            ]).contiguous().view((len(features), -1, args.chunk_size))

        batch["labels"] = torch.tensor([features[0]["labels"]])
        # add other features
        if "num_instance" in features[0]:
            batch["num_instance"] = torch.tensor([features[0]["num_instance"]])
            
        if "brc_id" in features[0]:
            
            batch["brc_id"] = [features[0]["brc_id"]]
        
        if "accepted_triage_team" in features[0]:
            batch["accepted_triage_team"] = [features[0]["accepted_triage_team"]]
        return batch

    if args.num_train_epochs > 0:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )
    if args.num_train_epochs > 0:
        train_dataloader = accelerator.prepare(train_dataloader)

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # # Get the metric function
    # if args.task_name is not None:
    #     metric = load_metric("glue", args.task_name)

    if args.num_train_epochs > 0:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        total_running_loss = 0.0
        running_best_metric = 0.0
        patience = 3
        patience_counter = 0
        for epoch in tqdm(range(args.num_train_epochs)):
            model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels = batch["labels"])
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                epoch_loss += loss.item()
                # add to total running loss
                total_running_loss += loss.item()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_postfix(loss=epoch_loss / completed_steps)
                
                    # log every 50 completed steps
                    if completed_steps % 10 == 0:
                        # log current step loss to wandb
                        # wandb.log({"step": completed_steps, "running_train_loss": total_running_loss / completed_steps})
                        metric_logger.log_metrics(completed_steps, {"running_train_loss": total_running_loss / completed_steps})

                if completed_steps >= args.max_train_steps:
                    break                        
            
            model.eval()
            all_preds = []
            all_preds_raw = []
            all_labels = []
            for step, batch in tqdm(enumerate(eval_dataloader)):                
                with torch.no_grad():
                    outputs = model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels = batch["labels"])
                # apply softmax to the logits of the output - using the softmax function
                preds_raw = outputs.logits.softmax(dim=-1).cpu()           

                
                # get argmax of preds raw
                preds = np.argmax(preds_raw, axis = -1)             
                
                all_preds_raw.extend(list(preds_raw))
                all_preds.extend(list(preds))
                all_labels.extend(list(batch["labels"].cpu().numpy()))
            
            all_preds_raw = np.stack(all_preds_raw)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
            print(f"all_preds_raw shape is: {all_preds_raw.shape}")
            print(f"all_preds shape is: {all_preds.shape} \n\n {all_preds}")
            print(f"all_labels shape is: {all_labels.shape} \n\n {all_labels}")
            # metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
            # use compute metrics with args: predictions, pred_scores, labels
            metrics = compute_metrics(all_preds, all_preds_raw, all_labels)
            logger.info(f"epoch {epoch} finished")
            logger.info(f"metrics: {metrics}")
            
            # log to wandb
            # wandb.log({"epoch": epoch, "train_loss": epoch_loss / completed_steps, "val_metrics": metrics})
            
            metric_logger.log_epoch_metrics(completed_steps,{"train_loss": epoch_loss / completed_steps})
            metric_logger.log_epoch_metrics(completed_steps, metrics, data_type = "valid")
            
            # check if model improves and save checkpoint if so
            if metrics["f1_macro"] > running_best_metric:
                logger.info(f"Model improved - will save checkpoint and carry on training!")
                running_best_metric = metrics["f1_macro"]
                patience_counter = 0
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(f"{args.output_dir}/checkpoint/", save_function=accelerator.save)
                # save tokenizer too
                tokenizer.save_pretrained(f"{args.output_dir}/checkpoint/")
                # also save the original model config
                if args.apply_lora:
                    
                    unwrapped_model.config.save_pretrained(f"{args.output_dir}/checkpoint/", save_function = accelerator.save)
                        # unwrapped_model.config.to_json_file(f"{args.output_dir}/checkpoint/")
                
            else:
                patience_counter += 1
            
            # early stopping if patient_counter >= patience
            if patience_counter >= patience:
                print(f"Patience reached, will be early stopping")
                break
                
    
    # if args.num_train_epochs == 0 and accelerator.is_local_main_process:
    #     model.eval()
    #     all_preds = []
    #     all_preds_raw = []
    #     all_labels = []
    #     for step, batch in enumerate(tqdm(eval_dataloader)):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #         preds_raw = outputs.logits.sigmoid().cpu()
    #         preds = (preds_raw > 0.5).int()
    #         all_preds_raw.extend(list(preds_raw))
    #         all_preds.extend(list(preds))
    #         all_labels.extend(list(batch["labels"].cpu().numpy()))
        
    #     all_preds_raw = np.stack(all_preds_raw)
    #     all_preds = np.stack(all_preds)
    #     all_labels = np.stack(all_labels)
    #     metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
    #     logger.info(f"evaluation finished")
    #     logger.info(f"metrics: {metrics}")
    #     for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    #         all_preds = (all_preds_raw > t).astype(int)
    #         metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw, k=[5,8,15])
    #         logger.info(f"metrics for threshold {t}: {metrics}")
    
    if args.run_full_eval:
        # # double check eval metrics
        # concat validation and test

        import pandas as pd
        import pickle
        save_path=f"/home/ec2-user/Experiments/plm_triage_models/results_plots/{args.task_name}/"
        all_eval_dataset = concatenate_datasets([processed_datasets["validation"], processed_datasets["test"]])
        final_eval_dataloader =  DataLoader(all_eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        final_eval_dataloader = accelerator.prepare(
                    final_eval_dataloader
        )
        print(f"running full eval!!!")
        model.eval()
        all_preds = []
        all_preds_raw = []
        all_labels = []
        all_label_attentions = []
        all_input_ids = []
        all_results = {}
        probs_dfs = []
        results_dfs = []
        for step, batch in tqdm(enumerate(final_eval_dataloader)):
            # print(f"Batch: {batch.keys()}")             
            with torch.no_grad():
                outputs = model(input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"],
                        labels = batch["labels"])
            # apply softmax to the logits of the output - using the softmax function
            preds_raw = outputs.logits.softmax(dim=-1).cpu()           

            
            # get argmax of preds raw
            preds = np.argmax(preds_raw, axis = -1)             
            
            all_preds_raw.extend(list(preds_raw.cpu().numpy()))
            all_preds.extend(list(preds.cpu().numpy()))
            all_labels.extend(list(batch["labels"].cpu().numpy()))
            
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

            reformed_input_ids = batch["input_ids"].view(batch_size, num_chunks*chunk_size, -1)

            reformed_input_ids = reformed_input_ids.squeeze(-1).cpu().numpy()
            all_input_ids.append(reformed_input_ids)
            
            
        all_dfs = pd.concat(results_dfs)
        all_preds_raw = np.stack(all_preds_raw)
        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)
        # print(f"all_preds_raw shape is: {all_preds_raw.shape}")
        # print(f"all_preds shape is: {all_preds.shape} \n\n {all_preds}")
        # print(f"all_labels shape is: {all_labels.shape} \n\n {all_labels}")
        # metrics = all_metrics(yhat=all_preds, y=all_labels, yhat_raw=all_preds_raw)
        # use compute metrics with args: predictions, pred_scores, labels
        metrics = compute_metrics(all_preds, all_preds_raw, all_labels)
        
        print(f"############Final Metrics are:\n {metrics}\n##################")
        
        # save metrics to file 
        with open(f'{save_path}/{model_name}-{args.chunk_size}-long_instance_results.pickle', 'wb') as f:
            pickle.dump(metrics, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        # save all_dfs too
        all_dfs.to_csv(f"{save_path}/{model_name}-{args.chunk_size}-results_df.csv", index = False)
        
        # save the label attentions and input ids too?
        with open(f'{save_path}/{model_name}-{args.chunk_size}-eval-label-attentions.pickle', 'wb') as f:
            pickle.dump(all_label_attentions, f, protocol = pickle.HIGHEST_PROTOCOL)
        
        with open(f'{save_path}/{model_name}-{args.chunk_size}-eval-reformed-input-ids.pickle', 'wb') as f:
            pickle.dump(all_input_ids, f, protocol = pickle.HIGHEST_PROTOCOL)
        

    # if args.output_dir is not None and args.num_train_epochs > 0:
        
    #     # # first save the model using native pytorch as the hf save_pretrained seems to mess up the peft modules
    #     # if not os.path.exists(f"{args.output_dir}/checkpoint_native/"):
    #     #     os.makedirs(f"{args.output_dir}/checkpoint_native", exist_ok = True)
        
    #     # save the state_dict 
    #     # torch.save(model.state_dict(), f"{args.output_dir}/checkpoint_native/pytorch_model.bin")
        
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(f"{args.output_dir}/checkpoint/", save_function=accelerator.save)
    #     # save tokenizer too
    #     tokenizer.save_pretrained(f"{args.output_dir}/checkpoint/")
    #     # also save the original model config
    #     if args.apply_lora:
            
    #         unwrapped_model.config.save_pretrained(f"{args.output_dir}/checkpoint/", save_function = accelerator.save)
    #         # unwrapped_model.config.to_json_file(f"{args.output_dir}/checkpoint/")


if __name__ == "__main__":
    main()
