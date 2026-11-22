# Outputs: what gets captured and saved

`src/main.py` saves each run as one `.pt` file. This page explains which token positions are
recorded, what the file contains, and how to load it.

## The capture convention

A model generates an answer as a sequence of tokens, and a prompt has many token positions. Where do
we read activations? **At every token of the answer** (plus any prompt positions you name in
`find_positions_of_interest`).

If the model answers `42` and that is two tokens `['4', '2']`, the run records activations at both,
named by the order they were written: `answer_0` (the `'4'`) and `answer_1` (the `'2'`). Each is
stored under `result["activations"][name]`, and `result["positions"][name]` gives its token index.
The answer's **last** token gets a shortcut too, `result["answer"]`: by then the model has committed
to its answer, so it is often the most informative single place to look. Every tensor recorded at a
position (residual, MLP neurons, attention heads, and the token / positional embeddings) is read at
that *same* position, so they all describe the same token.

> [!NOTE]
> How this works under the hood: after generating, we run the *whole* output sequence (prompt +
> generation) back through the model in a single pass and read its activation cache. Because that one
> pass covers every position, the answer's last token always has its activations computed, so there's no
> "ran out of tokens" edge case to worry about, whatever `--max-new-tokens` you choose.

Which substring counts as "the answer" is decided by `find_answer_span` in `src/inference.py`. It
only ever sees what the model wrote *before* its first special token: a trained model ends its answer
with `<eos>`, and that `<eos>` is never part of the answer. The default grabs the first
whitespace-delimited chunk of that text; override it for your task.

## What a capture run saves

A run **with `--capture-activations`** saves a dictionary like this (tensors are PyTorch tensors on the
CPU). The shapes shown are GPT-2's; a toy model you trained will have whatever sizes you configured
(e.g. a `[256]` residual instead of `[768]`):

```python
{
  "baseline": [                     # one entry per prompt (intervention mode uses this key too)
    {
      "prompt": "7+5=",
      "prompt_length": 4,           # number of prompt tokens
      "metadata": { ... },           # that prompt's ground truth, from PromptDataset
      "result": {
        "text": "7+5=12<eos>",       # prompt + generation, decoded (special tokens included)
        "completion": "12<eos>",     # just the generation
        "output_ids": Tensor[7],     # prompt + generated token ids
        "correct": True,             # exactly the answer's tokens, then <eos> (None without an "answer")
        "positions": {"answer_0": 4, "answer_1": 5},   # token index of each captured position
        "activations": {             # activations at every captured position
          "answer_0": {-1: {...}, 0: {"residual": Tensor[768], "mlp": Tensor[3072], "heads": Tensor[12, 64]}, ...},
          ...                        # -1 = before block 0 (embeddings); find_positions_of_interest's names too
        },
        "logits": {"answer_0": ..., "answer_1": ..., "eos": ...},   # with --capture-logits only
        "answer": {
          "token": "12",             # the answer (string), without the <eos> after it
          "token_id": Tensor[2],     # its token ids
          "position": 5,             # absolute index of its LAST token
          "residual":        {0: Tensor[768],  1: ...},   # per layer
          "mlp_neurons":     {0: Tensor[3072], 1: ...},   # per layer (the neurons)
          "attn_heads":      {0: Tensor[12, 64], ...},    # per layer (per head)
          "token_embedding": Tensor[768],   # content embedding at the answer token
          "pos_embedding":   Tensor[768],   # positional embedding (None for rotary models, e.g. Llama)
          "resid_pre":       Tensor[768],   # block 0's input (token + positional embedding)
          "logits":          Tensor[50257],   # what predicted the last answer token (--capture-logits)
        },
      },
    },
    ...
  ],
  "metadata": { "model_path": "...", "layer_indices": [...], "num_attention_heads": 12, ... },
}
```

To explore it in a notebook:

```python
import torch

data = torch.load("your_results.pt", weights_only=False)  # the .pt file main.py wrote
row = data["baseline"][0]
print(row["prompt"], "->", row["result"]["answer"]["token"])
answer = row["result"]["answer"]
layer = data["metadata"]["layer_indices"][0]  # a layer you actually captured
neurons = answer["mlp_neurons"][layer]  # MLP neurons at the answer token
print(neurons.shape)  # e.g. torch.Size([3072]) for GPT-2
```

## What an ablation sweep saves

An **intervention** run ([step 7](analysing-a-model.md#7-ablate-them-optional)) saves a slightly
different layout, `{"baseline": [rows], "baseline_accuracy": float, "ablations": [{"layer_idx",
"feature_idx", "type", "local_idx", "accuracy", "accuracy_drop", ...}, ...], "metadata": {...}}`,
where each ablation records only its accuracy and its drop versus baseline (the heavy per-ablation
rows are discarded; the unablated `baseline` is kept once). Each ablation also records
`step_accuracy` and `step_accuracy_drop`, plus `correct`: a small `[prompts × steps]` matrix of which
answer steps were right, so you can later split the drops by any property of the prompts (say, carry
vs no carry) without re-running anything. The file also keeps `baseline_step_accuracy` and
`baseline_correct`, and in `metadata` the `step_names`, the `ablation_type` and
`tf_generation_mismatches` (how many baseline prompts teacher forcing and generation disagreed on;
it should be 0). `src/analysis/plot_ablations.py` reads these files and turns the drops into heatmaps
and cross-run scatter plots.

### Splitting a drop by any property of the prompts

Row `i` of every `correct` matrix is prompt `i` of `baseline`, so you can recompute a drop on any
subset of the prompts. For example, does the most important component of the addition example
matter equally for sums that carry and sums that don't? Run this from the repository root:

```python
import sys

import torch

sys.path.insert(0, "src")
import inference
from utils.ablations import neuron_label

data = torch.load("sweeps/zero.pt", weights_only=False)  # a sweep from main.py --intervention
rows = data["baseline"]
carry = torch.tensor([int(row["metadata"]["answer"]) >= 10 for row in rows])  # one bool per prompt

entry = max(data["ablations"], key=lambda a: a["accuracy_drop"])  # the component that matters most
print("component:", neuron_label(entry["layer_idx"], entry["type"], entry["local_idx"]))
for name, subset in [("carry", carry), ("no carry", ~carry)]:
    baseline = inference.accuracy(inference.whole_answer_correct(data["baseline_correct"][subset]))
    ablated = inference.accuracy(inference.whole_answer_correct(entry["correct"][subset]))
    print(f"  {name}: accuracy drop {baseline - ablated:.2f} ({int(subset.sum())} prompts)")
```

On the model saved in the quickstart notebook, this prints that head `L0A1` is needed for both (a drop of 1.00 with carry,
0.87 without): it isn't specific to carrying.
