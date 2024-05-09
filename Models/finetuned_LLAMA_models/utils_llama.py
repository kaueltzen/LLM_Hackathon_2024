""" Utility functions for finetuning of LLAMA model with robocrys and / or LobsterPy text descriptions. """

import numpy as np
from unsloth import FastLanguageModel


def get_prediction_all_info(dataset: list, model, tokenizer) -> (list, list):
    """ Get model prediction for dataset. Give full instruction.
        Return two lists of y_true, y_pred. """
    y_pred, y_true = [], []
    for entry in dataset:
        y_true.append(float(entry["output"].split(" ")[-2]))
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                entry["instruction"],
                entry["input"],
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        output = tokenizer.batch_decode(outputs)
        num = output[0].split(" ")[-2]
        try:
            y_pred.append(float(num))
            if abs(float(num)) > 20000.0:
                print("VERY HIGH ERROR!: ", entry)
        except (TypeError, ValueError):
            print("Faulty response, append 0!")
            print(entry)
            print(output)
            y_pred.append(0.0)
    return y_true, y_pred


def get_prediction_subset_info(dataset: list, model, tokenizer, which_info: str = "robocrys") -> (list, list):
    """ Get model prediction for dataset. Give only part of instruction (for now, only Robocrys or LobsterPy part).
        Return two lists of y_true, y_pred. """
    y_pred, y_true = [], []
    for entry in dataset:
        y_true.append(float(entry["output"].split(" ")[-2]))
        new_input = get_subset_input_string(full_input=entry["input"], which_info=which_info)
        inputs = tokenizer(
        [
            alpaca_prompt.format(
                entry["instruction"],
                entry["input"],
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")
        
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
        output = tokenizer.batch_decode(outputs)
        num = output[0].split(" ")[-2]
        try:
            y_pred.append(float(num))
        except ValueError:
            y_pred.append(None)
    return y_true, y_pred


def get_subset_input_string(full_input: str, which_info: str = "robocrys") -> str:
    """ Gets subset of input string (either only lobsterpy or robocrys info
        for querying model."""
    # TODO hacky, improve
    div = " bonds. "
    fragments = full_input.split(div)  # see https://github.com/JaGeo/LobsterPy/blob/fd06eee7aed1ea97fadad271b09e7f3d5dc07da3/lobsterpy/cohp/describe.py#L238C27-L238C36
    lobsterpy_string, robocrys_string = "", ""
    for f_idx, f in enumerate(fragments):
        print(f_idx, f)
        if "mean ICOHP" in f:
            lobsterpy_string += f + div
        elif f_idx + 1 == len(fragments):
            robocrys_string += f
        else:
            robocrys_string += f + div
            
    if which_info == "robocrys":
        return robocrys_string
    return lobsterpy_string
    

def compute_mae(y_true, y_pred):
    output_errors = np.average(np.abs(y_pred - y_true))
    return np.average(output_errors)


def compute_rmse(y_true, y_pred):
    output_errors = np.average((y_true - y_pred) ** 2)
    return np.sqrt(output_errors)


# Tokenizer not modified in our apprach
_, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 800,
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass
    