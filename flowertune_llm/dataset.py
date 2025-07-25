import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"


from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner
from datasets import load_from_disk

LOCAL_DATASET_PATH = "/scratch/wd04/sm0074/preftune/datasets/alpaca_gpt4_local"

FDS = None  # Cache FederatedDataset

def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(model_name: str):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, use_fast=True, padding_side="right"
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    global FDS
    if FDS is None:
        full_dataset = load_from_disk(LOCAL_DATASET_PATH)
        train_data = full_dataset["train"]

        print(train_data.column_names)
        
        train_data = train_data.rename_column("output", "response")  
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = partitioner(train_data)

    return FDS.load_partition(partition_id)

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
