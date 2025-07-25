import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg):
    model_path = "/scratch/wd04/sm0074/models/open_llama_3b_v2_clean_real"

    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    # if model_cfg.quantization == 4:
    #     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # elif model_cfg.quantization == 8:
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # else:
    #     raise ValueError(
    #         f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
    #     )

    if model_cfg.quantization in [4, 8]:
        print("⚠️ Quantization requested, but BitsAndBytes is unavailable on Gadi. Loading full-precision model.")
        quantization_config = None
    else:
        raise ValueError("Use quantization=4 or 8 only.")

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     quantization_config=quantization_config,
    #     torch_dtype=torch.bfloat16,
    #     local_files_only=True,
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]
