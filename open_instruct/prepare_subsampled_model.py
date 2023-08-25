#!/usr/bin/env python
# coding=utf-8
"""
This file is modified from the huggingface example for finetuning language models
[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple
import datasets
from functools import reduce
import torch
import pickle
import numpy as np
import copy
import argparse
import subprocess
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel,
    set_seed,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from transformers.custom_utils import SelectorGenerator
from peft import LoraConfig, TaskType, get_peft_model
from mup import zip_infshapes
from pathlib import Path
import bitsandbytes as bnb
from custom_arguments import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)

def load_and_rescale(model, original_model, selector_generator, subsamp_ratio):
    infshapes = zip_infshapes(original_model, model)
    model_state = model.state_dict()
    original_state = original_model.state_dict()

    for name, param in model.named_parameters():
        param.infshape = infshapes[name]
        # rescale pretrained parameters
        if "lora" not in name:
            selectors = selector_generator.generate(
                original_state[name].shape, param.shape
            )
            model_state[name].copy_(original_state[name][np.ix_(*selectors)])
            if param.infshape.ninf() == 2:
                model_state[name].div_(subsamp_ratio)

def subsample_and_copy(model_args, tokenizer) -> Tuple[PreTrainedModel]:
    torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
    
    original_model = AutoModelForCausalLM.from_pretrained(
            model_args.pretrained_model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            use_auth_token=model_args.use_auth_token
        )   
    
    model_config = copy.deepcopy(original_model.config)
    model_config.subsamp_ratio = model_args.subsamp_ratio

    new_hidden_size = model_config.subsamp_ratio * model_config.hidden_size
    num_heads_dbl = 2 * model_config.num_attention_heads
    new_hidden_size = int(new_hidden_size / num_heads_dbl) * num_heads_dbl
    model_config.hidden_size = new_hidden_size
    intermediate_hidden_ratio = original_model.config.intermediate_size / original_model.config.hidden_size
    model_config.intermediate_size = int(new_hidden_size * intermediate_hidden_ratio)
    assert "llama" in model_config.model_type
    model = AutoModelForCausalLM.from_config(model_config)
    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = original_model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        original_model.resize_token_embeddings(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    selector_generator = SelectorGenerator(model_args.subsamp_ratio)
    if model_args.do_copy:
        logger.info('copy subsampled pretrained weights')
        load_and_rescale(model, original_model, selector_generator, model_args.subsamp_ratio)
    return model, original_model

def insert_modified_params(model, output_dir, subsamp_ratio):
    modified_param_names = ['base_model.model.model.embed_tokens.weight', 'base_model.model.lm_head.weight']
    modified_params_path = os.path.join(output_dir, f"modified_params_{subsamp_ratio}.pt")
    if os.path.exists(modified_params_path):
        modified_params = torch.load(modified_params_path)
        for name in modified_param_names:
            module_names = name.split(sep=".")[:-1]
            module = reduce(getattr, module_names, model)
            with torch.no_grad():
                module.weight.copy_(modified_params[name])
    else:
        state_dict = model.state_dict()
        modified_params = {}
        for name in modified_param_names:
            modified_params[name] = state_dict[name]
        torch.save(modified_params, modified_params_path)

def find_all_linear_names(args, model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def generate_subsamp_model(model_args, training_args):
    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.pretrained_model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this finetuning script."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this finetuning script."
        )

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    if model_args.pretrained_model_name_or_path:
        model, original_model = subsample_and_copy(model_args, tokenizer)
    else:
        logger.warning("No pretrained model_name_or_path is given. Training new model from scratch.")
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    # TODO DEBUG
    model.save_pretrained(model_args.subsamp_save_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_args.subsamp_save_dir)

    if model_args.use_lora:
        logger.info("Initializing LORA model...")
        modules = find_all_linear_names(model_args, model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=True, 
            r=model_args.lora_rank, 
            lora_alpha=model_args.lora_alpha, 
            target_modules=modules,
            lora_dropout=model_args.lora_dropout,
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        # insert_modified_params(model, training_args.output_dir, model_args.subsamp_ratio)
        # model.print_trainable_parameters()
        original_model = get_peft_model(original_model, peft_config)
    infshapes = zip_infshapes(original_model, model)
    infshapes_file_path = os.path.join(training_args.output_dir, "infshapes.pkl")
    with open(infshapes_file_path, "wb") as infshapes_file:
        pickle.dump(infshapes, infshapes_file)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if model_args.subsamp_save_dir is None:
        model_args.subsamp_save_dir = os.path.join(training_args.output_dir, "subsamp_model")
    os.makedirs(model_args.subsamp_save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    completion_file = Path(os.path.join(training_args.output_dir, "subsamp_complete"))
    
    if not completion_file.exists():
        generate_subsamp_model(model_args, training_args)
        completion_file.touch()
    
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    args.deepspeed = args.with_deepspeed
    args.model_name_or_path = args.subsamp_save_dir
    args.tokenizer_name = args.subsamp_save_dir
    parsed_args_path = os.path.join(args.output_dir, "parsed_args.pkl")

    with open(parsed_args_path, "wb") as args_file:
        pickle.dump(vars(args), args_file)
    
    if args.launcher == "python":
        launcher = ["python"]
    elif args.launcher == "accelerate":
        launcher = ["accelerate", "launch"]
    elif args.launcher == "deepspeed":
        assert args.deepspeed
        launcher = ["deepspeed"]
    else:
        raise ValueError
    command = launcher + ["open_instruct/custom_finetune_trainer.py", parsed_args_path]
    subprocess.run(command)

if __name__ == "__main__":
    main()