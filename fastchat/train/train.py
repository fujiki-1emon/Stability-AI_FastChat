# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt="",
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"] if "content" in sentence else sentence["value"])
        format_conv = conv.get_prompt()
        # format_conv = format_conv.replace(conv.system, prompt)
        conversations.append(format_conv)
    # Tokenize conversations
    tokenizer.model_max_length = 2048
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    # Mask targets. Only compute loss on the assistant outputs.
    sep = "\n" + conv.roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        deli = conv.sep2 + "\n" + conv.roles[0] + "\n"
        turns_ = conversation.split(deli)
        if len(turns_) == 2 or len(turns_) == 1:
            turns = [conversation]
        else:
            turns = [conv.roles[0] + "\n" + x + conv.sep2 + "\n" for x in turns_[2:-1]]
            turns = turns + [conv.roles[0] + "\n" + turns_[-1]]
            turns = [turns_[0] + conv.sep2 + "\n" +  conv.roles[0] + "\n" + turns_[1] + conv.sep2 + "\n"] + turns
        assert "".join(turns) == conversation
        cur_len = 0
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len
            target[cur_len - 1: cur_len+1] = IGNORE_TOKEN_ID

        target[cur_len:] = IGNORE_TOKEN_ID
#    attention_mask = (input_ids.ne(tokenizer.pad_token_id) | targets.ne(IGNORE_TOKEN_ID)).long()
#    for (i, input_id) in enumerate(input_ids):
#        idx_max = (input_id != 0).nonzero(as_tuple=True)[0][-1].item()
#        attention_mask[i, idx_max + 1:] = 0
#        attention_mask[i, :idx_max + 1] = 1

    # check tensors all equal to -100
    for target in targets:
        assert (target == IGNORE_TOKEN_ID).sum() != len(target)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in tqdm(raw_data)]
        data_dict = preprocess(sources, tokenizer)
        self.cached_data_dict = {}
        for i in tqdm(range(len(sources))):
            ret = preprocess([sources[i]], tokenizer)
            self.cached_data_dict[i] = ret
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        from datasets import load_from_disk
        print("Load from disk")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_3b_except_orca")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_3b_except_orca_capybara")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_3b_glaive_slim_except_orca_canbara/")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/stablelm1.6b_instruct_glaive_math_slim_sharegpt_ultra_deita_2048/")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_4_jan_2048/")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_5_jan_2048/")
#        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/stablelm1.6b_instruct_glaive_math_slim_sharegpt_ultra_deita_wizard_capybara_2048")
        ds = load_from_disk("/fsx/codeai/FastChat/fastchat/train/data_hub/all_data_5_jan_2048_cleaned/")
        self.cached_data_dict = ds.to_pandas()

    def __len__(self):
        return len(self.cached_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.cached_data_dict['input_ids'][i]).long()
        labels = torch.tensor(self.cached_data_dict['labels'][i]).long()
        attention_mask = torch.tensor(self.cached_data_dict['attention_mask'][i]).long()
        # last value attention_mask equal to 1
        eos_id = 100257
        idx_max = (input_ids != eos_id).nonzero(as_tuple=True)[0][-1].item()
        attention_mask[:idx_max] = 1
        attention_mask[idx_max:] = 0
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        try:
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        except Exception as e:
            print("error", e)
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
            i = 0
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    from datasets import load_dataset, concatenate_datasets, Dataset
    import pandas as pd
    import os
    root = "/fsx/codeai/FastChat/fastchat/train/data_hub"
    # data_paths = [ "ultrachat_200k.jsonl",  "capybara.jsonl", "codeai.jsonl", "meta-math.jsonl", "orca_gpt4_1M.jsonl","wizard_196k.jsonl"]
    data_paths = [ "capybara.jsonl" ]
    data_paths = [os.path.join(root, x) for x in data_paths]
    train_json = []
    for path in data_paths:
        train_json += json.load(open(path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=train_dataset)


from p_tqdm import p_map

def preprocess_item(source):
    tokenizer = transformers.AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
    tokenizer.pad_token = tokenizer.eos_token
    try:
        res = preprocess([source], tokenizer)
    except Exception as e:
        print(e)
        res = None
    return res

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("preparing data")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_ds = data_module['train_dataset']
    print(len(train_ds))
    lst = [] 
    import random
    idxs = random.sample(range(0, len(train_ds)), 2000)
    error = 0
    for i in tqdm(idxs):
        try:
            lst.append(train_ds[i])
        except Exception as e:
            error += 1
            print(e)
    print("Number of error: ", error)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "/fsx/ckpts/stablelm-1b-step498k",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing = True
#    training_args.neftune_noise_alpha=5
    print("Start training")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
#    if
#    list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"train_dataset)):
#        trainer.train(resume_from_checkpoint=True)
#    else:
#        trainer.train()
#    trainer.save_state()
    print("Training now")
    trainer.train()
#    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
