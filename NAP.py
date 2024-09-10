from nnsight import LanguageModel
from nnsight.models import UnifiedTransformer
from collections import defaultdict
from tqdm import tqdm
from dataset import HFEAPDataset

import pickle
import csv
import torch
import random
import argparse
import numpy as np

def load_prompts_hf(datapath, tokenizer, num_examples=100, max_length=64, batch_size=8):
    if 'ioi' in datapath:
        task = 'ioi'
    elif 'mcqa' in datapath:
        task = 'mcqa'
    else:
        raise ValueError(f"Unrecognized task: {datapath}")
    
    dataset = HFEAPDataset(datapath, tokenizer, task=task)
    dataloader = dataset.to_dataloader(batch_size)
    examples = []
    tokenizer.padding_side = "left"

    for row in dataset:
        clean, corrupted, labels = row
        clean_tok = tokenizer(clean, return_tensors="pt", add_special_tokens=False,
                              padding="max_length", max_length=max_length).input_ids
        corrupted_tok = tokenizer(corrupted, return_tensors="pt", add_special_tokens=False,
                                  padding="max_length", max_length=max_length).input_ids
    
        example = {
            "clean_prompt": clean_tok,
            "corrupted_prompt": corrupted_tok,
            "clean_answer": torch.LongTensor([labels[0]]),
            "corrupted_answer": torch.LongTensor([labels[1]])
        }
        examples.append(example)
    return examples


def load_prompts(datapath, tokenizer, num_examples=100):
    examples = []
    
    with open(datapath, 'r') as datafile:
        reader = csv.reader(datafile)
        next(reader)    # skip header
        for row in reader:
            clean, corrupted, target_year_str = row
            target_year = int(target_year_str)
            clean_tok = tokenizer(clean, return_tensors="pt", add_special_tokens=False).input_ids
            corrupted_tok = tokenizer(corrupted, return_tensors="pt", add_special_tokens=False).input_ids
            target_year_tok = tokenizer(target_year_str, return_tensors="pt", add_special_tokens=False).input_ids
            if clean_tok.shape[-1] != corrupted_tok.shape[-1]:
                continue

            target_year_prefix = corrupted.split()[-1]
            less_than_tok = torch.Tensor([[0]])
            while less_than_tok.shape[-1] != 2:
                less_than_target = random.randint(0, target_year)
                less_than_str = str(less_than_target) if less_than_target > 10 else "0" + str(less_than_target)
                less_than_str = target_year_prefix + less_than_str
                less_than_tok = tokenizer(less_than_str, return_tensors="pt", add_special_tokens=False).input_ids
            greater_than_tok = torch.Tensor([[0]])
            while greater_than_tok.shape[-1] != 2:
                greater_than_target = random.randint(target_year+1, 99)
                greater_than_str = str(greater_than_target) if greater_than_target > 10 else "0" + str(greater_than_target)
                greater_than_str = target_year_prefix + greater_than_str
                greater_than_tok = tokenizer(greater_than_str, return_tensors="pt", add_special_tokens=False).input_ids
            
            example = {
                "clean_prompt": clean_tok,
                "corrupted_prompt": corrupted_tok,
                "greater_than": greater_than_tok[:, -1],
                "less_than": less_than_tok[:, -1],
                "correct_idx": target_year_tok[:, -1]
            }
            examples.append(example)

    return examples


def attribution_patching(model, batch, submodules):
    clean_cache = {}
    patch_cache = {}
    effects = {}

    clean_examples = torch.cat([e["clean_prompt"] for e in batch], dim=0)
    patch_examples = torch.cat([e["corrupted_prompt"] for e in batch], dim=0)
    clean_ans_id = torch.cat([e["clean_answer"] for e in batch], dim=0)
    patch_ans_id = torch.cat([e["corrupted_answer"] for e in batch], dim=0)

    with model.trace(clean_examples):
        for submodule in submodules:
            hidden_state = submodule.output
            if type(hidden_state.shape) == tuple:
                hidden_state = hidden_state[0]
            clean_cache[submodule] = hidden_state.save()
            hidden_state.retain_grad()
        # logits = model.lm_head.output.save()
        logits = model.unembed.output.save()
    
    logit_diff = logits.value[:, -1, patch_ans_id] - logits.value[:, -1, clean_ans_id]
    logit_diff.sum().backward()
    # if input is non-contrastive
    # logprob = -1 * torch.nn.functional.log_softmax(logits, dim=-1)[:, -1, gt_id]
    # logprob.sum().backward()

    with model.trace(patch_examples):
        for submodule in submodules:
            hidden_state = submodule.output
            if type(hidden_state.shape) == tuple:
                hidden_state = hidden_state[0]
            patch_cache[submodule] = hidden_state.save()
    
    for submodule in submodules:
        effects[submodule] = (clean_cache[submodule].value.grad * (patch_cache[submodule].value - clean_cache[submodule].value)).detach()
        effects[submodule] = effects[submodule].sum(dim=1).sum(dim=0)
        effects[submodule] = effects[submodule].view(-1, model.cfg.d_model)

    return effects


def get_top_neurons(effects, k=100):
    stacked_effects = torch.stack([effects[e] for e in effects], dim=0)
    top_values, indices = torch.topk(stacked_effects.flatten(), k=k)
    indices = np.array(np.unravel_index(indices.numpy(), stacked_effects.shape)).T
    return indices, top_values


def get_neurons_above_threshold(effects, t=0.1):
    neurons_list = []
    filtered_effects = []
    stacked_effects = torch.stack([effects[e] for e in effects], dim=0)
    for i in range(stacked_effects.shape[0]):
        idxs_above_t = (stacked_effects[i] > t).nonzero().flatten().tolist()
        neurons_list.extend([f"{i}-{idx}" for idx in idxs_above_t])
        filtered_effects.extend([stacked_effects[i][idx] for idx in idxs_above_t])
    return neurons_list, filtered_effects


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="gpt2")
    parser.add_argument("--dataset", type=str, default="danaarad/ioi_dataset")
    parser.add_argument("--use_attn", "-a", action="store_true")
    parser.add_argument("--use_inputs", "-i", action="store_true")
    parser.add_argument("--num_examples", "-n", type=int, default=1000)
    parser.add_argument("--threshold", "-t", type=float, default=0.002)
    args = parser.parse_args()

    model_name = args.model_name
    num_examples = args.num_examples

    model = UnifiedTransformer.UnifiedTransformer(model_name).to("cuda")
    submodules = [layer.mlp for layer in model.blocks]
    num_layers = len(submodules)

    if args.use_attn:
        submodules.extend([layer.attn.hook_result for layer in model.blocks])
        model.cfg.use_attn_result = True
    if args.use_inputs:
        submodules.append(model.hook_embed)
    t = args.threshold

    examples = load_prompts_hf(args.dataset, model.tokenizer,
                            num_examples=num_examples)
    num_examples = min(num_examples, len(examples))
    batches = []
    batch_size = 8
    prev_idx = 0
    for i in range(batch_size, num_examples, batch_size):
        batches.append(examples[prev_idx:i])
        prev_idx = i
    if num_examples % batch_size != 0:
        batches.append(examples[prev_idx:])
    
    mean_effects = {}
    for batch in tqdm(batches, desc="Batches", total=len(batches)):
        effects = attribution_patching(model, batch, submodules)
        for submodule in submodules:
            if submodule not in mean_effects:
                mean_effects[submodule] = effects[submodule]
            else:
                mean_effects[submodule] += effects[submodule]

    for submodule in submodules:
        mean_effects[submodule] /= num_examples

    nodesname = "mlp"
    if args.use_attn:
        nodesname += "attn"
    if args.use_inputs:
        nodesname += "embed"

    model_basename = model_name.split("/")[-1]
    dataset_basename = args.dataset.split("/")[-1]
    with open(f"NAP_effects/{model_basename}_{dataset_basename}_NAP_{nodesname}_allscores.pt", "wb") as neurons_effects:
        effects_list = []
        stacked_mlp_effects = torch.stack([mean_effects[e] for e in submodules[:num_layers]], dim=0)
        effects_list.append(stacked_mlp_effects)
        if args.use_attn:
            stacked_attn_effects = torch.stack([mean_effects[e] for e in submodules[num_layers:num_layers*2]], dim=0)
            effects_list.append(stacked_attn_effects)
        if args.use_inputs:
            effects_list.append(mean_effects[submodules[-1]])
        torch.save(effects_list, neurons_effects)