from typing import Optional, List, Union, Literal, Tuple
from functools import partial 

import pandas as pd
import torch 
from torch.nn.functional import kl_div
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer


def get_metric(metric_name: str, 
               task: str, 
               tokenizer: Optional[PreTrainedTokenizer] = None, 
               model: Optional[HookedTransformer] = None):
    """
    Get a metric function based on the metric name and task.
    The metric function basic signature is:
        metric(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False)

        TODO DOCUMENT SPECIFIC PARAMETERS AND MAKE KL/JS PARAMS MATCH THESE

    Args:
        metric_name (str): The name of the metric. Currently supported are: 
            - 'kl_divergence' or 'kl' for KL Divergence between the model logits and clean logits.
            - 'js_divergence' or 'js' for JS Divergence between the model logits and clean logits.
            - 'normalized_logit' for taking the label logit and normalizing it with the highest logit.
            - 'acc' for mean accuracy.
            - 'logit_diff' or 'prob_diff' for Logit difference between 
        task (str): The task of the model
        tokenizer (Optional[PreTrainedTokenizer]): The tokenizer of the model
        model (Optional[HookedTransformer]): The model
    """
    if metric_name == 'kl_divergence' or metric_name == 'kl':
        return partial(divergence, divergence_type='kl')
    elif metric_name == 'js_divergence' or metric_name == 'js':
        return partial(divergence, divergence_type='js')
    elif metric_name == 'normalized_logit':
        return normalized_logit
    elif metric_name.startswith('acc'):
        # Find the k for top-k accuracy
        if metric_name == 'acc':
            k = 1
        else:
            k = int(metric_name.split('-')[-1])
        return partial(topk_accuracy, k=k)
    elif metric_name == 'logit_diff' or metric_name == 'prob_diff':
        prob = (metric_name == 'prob_diff')
        if 'greater-than' in task:
            assert tokenizer is not None or model is not None, "Either tokenizer or model must be set for greater-than and prob / logit diff"
            tokenizer = tokenizer or model.tokenizer
            logit_diff_fn = partial(logit_diff_greater_than, year_indices=get_year_indices(tokenizer))
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    elif metric_name == "normalized_logit":
        normalized_logit_fn = normalized_logit
        return partial(normalized_logit_fn)
    elif metric_name == "sequence_logprob":
        return sequence_logprob
    else: 
        raise ValueError(f"got bad metric_name: {metric_name}")

# TODO TEST THIS
def normalized_logit(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean: bool=True, loss: bool=False):
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    circuit_outputs = circuit_logits
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_logits = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))[:, 0]
    max_logits = circuit_outputs.max(dim=-1).values
    results = good_logits / max_logits
    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results

# TODO TEST THIS
def topk_accuracy(logits: torch.Tensor, corrupted_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean: bool=True, loss: bool=False, k: int=1):
    if type(labels) == tuple:
        labels = torch.Tensor(labels).to("cuda")
    if labels.shape[-1] == 2:
        # If labels have both clean (good) and corrupt (bad) labels, take only the clean one
        labels = labels[:, 0]
    labels = labels.view(-1, 1)

    logits = get_logit_positions(logits, input_length)
    predictions = logits.topk(k=k, dim=-1).indices # Get top-k predictions
    topk_acc = torch.any(predictions.eq(labels.expand_as(predictions)), dim=-1).float() # Check if any of the top-k predictions match the label
    if loss:
        topk_acc = 1 - topk_acc
    if mean:
        topk_acc = topk_acc.mean()
    return topk_acc


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits = logits[idx, input_length - 1]
    return logits


def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (kl_div(m, p.log(), log_target=True, reduction='none').mean(-1) + kl_div(m, q.log(), log_target=True, reduction='none').mean(-1))


def divergence(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, divergence_type: Union[Literal['kl'], Literal['js']]='kl', mean=True, loss=True):
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    circuit_probs = torch.softmax(circuit_logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if divergence_type == 'kl':
        results = kl_div(circuit_probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    elif divergence_type == 'js':
        results = js_div(circuit_probs, clean_probs)
    else: 
        raise ValueError(f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'")
    return results.mean() if mean else results


def logit_diff(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False):
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    circuit_outputs = torch.softmax(circuit_logits, dim=-1) if prob else circuit_logits
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    labels = torch.tensor(labels, dtype=torch.long, device=circuit_outputs.device)
    good_bad = torch.gather(circuit_outputs, -1, labels) 
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean: 
        results = results.mean()
    return results


def get_year_indices(tokenizer: PreTrainedTokenizer):
    return torch.tensor([tokenizer(f'{year:02d}', add_special_tokens=False).input_ids[0] for year in range(100)])


def logit_diff_greater_than(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, prob=False, loss=False, year_indices=None):
    # Prob diff (negative, since it's a loss)
    circuit_logits = get_logit_positions(circuit_logits, input_length)
    circuit_outputs = torch.softmax(circuit_logits, dim=-1) if prob else circuit_logits
    # print(circuit_outputs)
    # print(circuit_outputs[:, tuple(year_indices)])
    circuit_outputs = circuit_outputs[:, year_indices]

    results = []
    if prob:
        for prob, year in zip(circuit_outputs, labels):
            results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
    else:
        for logit, year in zip(circuit_outputs, labels):
            results.append(logit[year + 1 :].mean() - logit[: year + 1].mean())

    results = torch.stack(results)
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


def sequence_logprob(circuit_logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    circuit_outputs = torch.nn.functional.log_softmax(circuit_logits, dim=-1)
    # good_bad = torch.gather(circuit_outputs, -1, labels.to(circuit_outputs.device))
    padded_labels = []
    max_label_len = 0
    for labelset in labels:
        max_label_len = max(max_label_len, len(labelset[0]))
    for labelset in labels:
        padded_labels.append(labelset)
        if len(labelset[0]) < max_label_len:
            for i in range(max_label_len - len(labelset[0])):
                padded_labels[-1][0].append(0)
                padded_labels[-1][1].append(0)

    labels = torch.tensor(padded_labels, dtype=torch.long, device=circuit_outputs.device)[:, 0, :]
    print(labels)
    logprobs = torch.zeros(circuit_outputs.shape[0])
    for token_position in range(circuit_outputs.shape[1] - labels.shape[1], circuit_outputs.shape[1] - 1):
        next_tokens = labels[:, token_position+1 - labels.shape[1]]
        logits = circuit_outputs[:, token_position]
        next_tokens_logprobs = torch.gather(logits, -1, next_tokens)
        logprobs += next_tokens_logprobs
    if loss:
        # remember it's reversed to make it a loss
        logprobs = -logprobs
    if mean: 
        logprobs = logprobs.mean()
    return logprobs