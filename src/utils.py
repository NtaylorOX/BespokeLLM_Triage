import torch
import io
import argparse
import json
import yaml
import wandb
from torch.utils.tensorboard import SummaryWriter

# count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# freeze given neural network

def freeze_model(model):    
    '''
    Function to freeze the layers of a model
    
    '''
    
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
# unfreeze params of given model
def unfreeze_model(model):
    '''
    Function to unfreeze the layers of a model
    
    '''
    
    # unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        

def get_model_size(model):
    """Returns size of PyTorch model in bytes"""
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)
    print("Model size (MB):", model_size)
    return model_size


def get_full_model_size(model):
    # Parameter sizes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    # Buffer sizes
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # State dict sizes
    state_dict = model.state_dict()
    state_stream = io.BytesIO()
    torch.save(state_dict, state_stream)
    state_dict_size = len(state_stream.getbuffer())
    
    total_size = (param_size + buffer_size + state_dict_size) / (1024**2)
    print("Total size (MB):", total_size)
    return total_size


def get_model_save_name(model_name_or_path):
    # TODO clean this up/improve
    # THIS IS ALL VERY CRUDE AND DEPENDENT ON HAVING TRAINED USING THE SCRIPTS INSIDE THIS REPO - forward slashes really matter for the naming convention make sure to append the path with a forward slash
    # if "saved_models" in model_name_or_path:
    #     if "declutr" in model_name_or_path:
    #         if "few_epoch" in model_name_or_path:
    #             if "span_comparison" in model_name_or_path:
    #                 model_name = model_name_or_path.split("/")[9] + "/declutr/" + model_name_or_path.split("/")[-3]
    #             else:
    #                 model_name = model_name_or_path.split("/")[8] + "/declutr/" + model_name_or_path.split("/")[-3]

    #         else:
    #             model_name = model_name_or_path.split("/")[7] + "/declutr/" + model_name_or_path.split("/")[-3]
    #     elif "contrastive" in model_name_or_path or "custom_pretraining" in model_name_or_path:
    #         model_name = model_name_or_path.split("/")[7]
    #     else:
    #         model_name = model_name_or_path.split("/")[7]
    # else:    
    #     model_name = model_name_or_path.split("/")[-1]
        
    # return model_name
    
    #OHFT is a bit difference
    if "Experiments" in model_name_or_path:
        if "declutr" in model_name_or_path:
            if "few_epoch" in model_name_or_path:
                model_name = model_name_or_path.split("/")[8] + "-declutr/" + model_name_or_path.split("/")[-3]
            else:
                model_name = model_name_or_path.split("/")[7] + "-declutr/" + model_name_or_path.split("/")[-3]
            
                
        else:
            if "few_shot" in model_name_or_path:
                model_name = model_name_or_path.split("/")[6]
            else:
                model_name = model_name_or_path.split("/")[6] + "_all_data"
    elif "Language_models/HuggingFace" in model_name_or_path:
        num_subfolders = len(model_name_or_path.split("/"))
        # print(f"num sub folders: {num_subfolders}")
        # dumbest code i've ever seen but it works for this machine - this will use the 2nd folder name for the model name given models that have 2 folders as their id - e.g emilyalsentzer/Bio_ClinicalBERT/
        if num_subfolders == 8:
            model_name = model_name_or_path.split("/")[6]
        else:
            model_name = model_name_or_path.split("/")[5]
    else:
        # in this case hopefully model is default name/from online     
        model_name = model_name_or_path.split("/")[-1]
    return model_name

def get_dataset_directory_details(args:argparse.Namespace) -> argparse.Namespace:
    
    ''' Function to retrieve the dataset directory details from the datasets.yaml file'''

    with open('./data/datasets.yaml', 'r') as f:
        datasets = yaml.load(f, yaml.FullLoader)
    
    try:
        dataset_info = datasets[args.task_name]
        for k, v in dataset_info.items():
            setattr(args, k, v)
    except KeyError:
        print(f"Task name {args.task_name} not in datasets.yaml. Available tasks are: {list(datasets.keys())}")
        exit(0)

    return args

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
        return batch
    
from abc import ABC, abstractmethod
class CommonLogger:

    def __init__(self, logger):
        self.logger = logger

    def log_metrics(self, step, metrics, data_type='train'):
        tagged_metrics = {f"{data_type}/{k}": v for k, v in metrics.items()}
        self.logger.log(step, tagged_metrics)

    def log_epoch_metrics(self, epoch, metrics, data_type='train'):
        tagged_metrics = {f"{data_type}/{k}": v for k, v in metrics.items()}
        self.logger.log_epoch(epoch, tagged_metrics)


class Logger(ABC):

    @abstractmethod
    def log(self, step, tagged_metrics):
        pass

    @abstractmethod    
    def log_epoch(self, epoch, tagged_metrics):
        pass


class WandbLogger(Logger):

    def __init__(self, project, config=None, job_type=None, dir=None, name=None):
        import wandb
        wandb.init(project=project, config=config, job_type=job_type, dir=dir, name=name)

    def log(self, step, metrics):
        wandb.log(metrics, step=step)

    def log_epoch(self, epoch, metrics):
        wandb.log({k:v for k,v in metrics.items()}, step=epoch)


class TensorBoardLogger(Logger):
    
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(f"{log_dir}/tb_logs/")

    def log(self, step, tagged_metrics):
        for key, value in tagged_metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_epoch(self, epoch, tagged_metrics):
        for key, value in tagged_metrics.items():
            self.writer.add_scalar(key, value, epoch)
            

''' 
example usage:
logger = CommonLogger(WandbLogger()) 

# or 

logger = CommonLogger(TensorBoardLogger('logs'))

logger.log_metrics(step, metrics, data_type='train')
logger.log_metrics(step, metrics, data_type='valid') 

logger.log_epoch_metrics(epoch, epoch_metrics, data_type='train')
'''