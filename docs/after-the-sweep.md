# After the sweep: from *which* to *how*

The ablation sweep tells you **which** components your model needs. This page helps you read those
results critically, then suggests where to look to find out **how** those components do their job,
which is the research the template leaves to you ([concepts](concepts.md#what-this-template-does-and-doesnt-do)).

The snippets below are runnable from the repository root after the
[quickstart notebook](../examples/quickstart.ipynb) (they use its `saved_models/`, `captures/run.pt`
and `sweeps/zero.pt`). Change the paths, layers and components for your own runs. The numbers quoted
are what they print on the model saved in the quickstart notebook; a fresh run, especially on other
hardware, trains a slightly different model, so its numbers (and which components come out on top)
can differ.

- [Reading the results](#reading-the-results)
- [Sanity checks before you trust a result](#sanity-checks-before-you-trust-a-result)
- [Finding out how a component works](#finding-out-how-a-component-works)

## Reading the results

- **The heatmaps** ([plotting the drops](analysing-a-model.md#plotting-the-drops)). Each cell is one
  component's accuracy drop. Expect most cells to be near 0 and a handful to stand out: the model
  relies on a few components, and those are your circuit candidates. Red means removing the
  component hurt; blue would mean removing it *helped*, which is rare and worth a closer look.
- **The circuit diagram.** It keeps, per layer, the components in the top percentile of drop and
  links survivors in adjacent layers. The links mean "both matter", **not** "one feeds the other":
  whether information actually flows between them is what
  [activation patching](#3-activation-patching-does-this-component-carry-the-information) tests.
- **The scatter** (with two or more runs). Points on the diagonal matter equally in both settings;
  points off it matter in one setting only. Comparing zero and mean ablation this way is a quick
  robustness check.
- **Per-step drops** (`--metric step:<name>`). `step0` drops mean the component is needed for the
  answer's first token; a drop only at `step:eos` means the model still writes the answer but no
  longer stops.
- **The split-half value** (`src/analysis/split_half.py`). Near 1 for the meaningful components means
  the ranking doesn't depend on which prompts you sampled. A low value means the ranking is noise:
  run more prompts before reading anything into it.

## Sanity checks before you trust a result

- **Baseline accuracy is high.** An ablation can only show a drop if the model was right to begin
  with. If the baseline is low, improve the model or the prompts first.
- **Teacher forcing agrees with generation.** `main.py` prints
  `Teacher-forced vs generated baseline: 0 of N prompts differ`, and the sweep file stores it as
  `metadata["tf_generation_mismatches"]`. Anything above 0 means the sweep's scores don't match what
  the model actually generates (often `--max-new-tokens` is too small).
- **The result holds under both ablation types.** Run the sweep with `--ablation-type zero` and with
  `mean`. A component that matters only under zero ablation may just be reacting to an unusual value
  ([zero vs mean ablation](concepts.md#zero-vs-mean-ablation)).
- **The result holds on other prompts.** Re-run with a different `--seed`, or with prompts from your
  validation split, and check the same components come out on top.

## Finding out how a component works

Four standard techniques, from the cheapest to the most direct. Each is a starting point; the
[reading list](reading.md) has the papers that develop them.

### 1. Attention patterns: what does a head look at?

An attention head decides which earlier positions to read from. Its attention pattern shows, for each
position, how much weight it puts on every earlier one:

```python
import sys

sys.path.insert(0, "src")
from model import load_model

model = load_model("saved_models")
prompt = "3+4=7\n1+8=9\n6+2=8\n5+0=5\n7+5="
_, cache = model.run_with_cache(model.to_tokens(prompt))
tokens = model.to_str_tokens(prompt)

layer, head = 0, 1
pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # [query position, key position]
weights = pattern[-1]  # where the last position ("=", which writes the answer) looks
for position in weights.argsort(descending=True)[:3].tolist():
    print(f"  {tokens[position]!r} at position {position}: {weights[position]:.2f}")
```

On the quickstart's model, head `L0A1` (the most important component) puts most of its attention on
the question's `7`, `5` and `+`: it gathers the operands at the position that has to produce the sum.

### 2. Direct logit attribution: what does a component push the output towards?

The model's output logits are (up to the final layer norm) a sum of what every component writes into
the residual stream. So you can ask how much each component *directly* raises or lowers the logit of
the correct answer:

```python
import sys

sys.path.insert(0, "src")
from model import load_model

model = load_model("saved_models")
prompt = "3+4=7\n1+8=9\n6+2=8\n5+0=5\n7+5="
_, cache = model.run_with_cache(model.to_tokens(prompt))
answer = model.tokenizer("1", add_special_tokens=False)["input_ids"][0]  # first token of "12"

# The final layer norm centres the residual stream, divides by its scale and multiplies by a weight;
# with the scale frozen at its value in this run, that is linear, so each head's share adds up.
scale = cache["ln_final.hook_scale"][0, -1]
direction = model.ln_final.w * model.W_U[:, answer]  # [d_model]
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        output = cache[f"blocks.{layer}.attn.hook_z"][0, -1, head] @ model.W_O[layer, head]  # [d_model]
        effect = ((output - output.mean()) / scale) @ direction
        print(f"  L{layer}A{head}: {effect.item():+.2f}")
```

Here `L0A1` comes out at about −0.7: its *direct* effect on the answer is small and even negative,
although removing it destroys the answer. That's the lesson of this technique: a component can matter
**indirectly**, by writing information that later components read, and direct attribution only sees
the last step. (The same sum over every component, plus the biases, gives back the model's exact
logit.)

### 3. Activation patching: does this component carry the information?

Run the model on two prompts that differ in one detail, a *clean* one and a *corrupted* one, then
copy one component's output from the clean run into the corrupted run. If the prediction flips to the
clean answer, that component carries the information that differs:

```python
import sys

sys.path.insert(0, "src")
from model import load_model

model = load_model("saved_models")
shots = "3+4=7\n1+8=9\n6+2=8\n5+0=5\n"
clean, corrupted = model.to_tokens(shots + "7+5="), model.to_tokens(shots + "7+1=")  # answers 12 and 8
_, clean_cache = model.run_with_cache(clean)

layer, head = 0, 1
name = f"blocks.{layer}.attn.hook_z"


def patch(z, hook):
    z[:, -1, head] = clean_cache[name][:, -1, head]  # this head's output, taken from the clean run
    return z


with model.hooks(fwd_hooks=[(name, patch)]):
    logits = model(corrupted)
print("corrupted run predicts:", repr(model.to_string(model(corrupted)[0, -1].argmax())))
print("with the head patched: ", repr(model.to_string(logits[0, -1].argmax())))
```

The corrupted prompt `7+1=` predicts `'8'`; with just `L0A1`'s output at the last position patched in
from `7+5=`, it predicts `'1'`, the first digit of `12`. This one head carries the operand information
the rest of the model computes the sum from. Patching is the standard way to go from "this matters" to
"this is where the information flows"; the activation-patching papers in the
[reading list](reading.md#methods-for-causal-analysis) cover how to choose prompts and metrics.

### 4. Inspecting a neuron: when does it fire?

A neuron's activation across many prompts often reveals what it responds to. The capture file already
has every neuron's activation at every answer token:

```python
from collections import defaultdict

import torch

data = torch.load("captures/run.pt", weights_only=False)  # a capture from main.py --capture-activations
layer, neuron = 1, 140
by_answer = defaultdict(list)
for row in data["baseline"]:
    activations = row["result"]["activations"].get("answer_0")  # the answer's first token
    if activations is not None:
        by_answer[row["metadata"]["answer"]].append(activations[layer]["mlp"][neuron].item())
for answer in sorted(by_answer, key=int):
    values = by_answer[answer]
    print(f"  answer {answer:>2}: mean activation {sum(values) / len(values):+.2f} ({len(values)} prompts)")
```

On the quickstart's model, neuron 140 in layer 1 (one of the sweep's top components) is clearly
active (between about +1.7 and +3.7) whenever the sum is below 10, and about 0 whenever it is 10 or more: it
behaves like a "the answer has one digit" detector. Group by whatever your task's ground truth
offers (operands, categories, answer length) to find such patterns. Keep in mind that a neuron can
mix several features ([superposition](concepts.md#concepts-youll-need)), so a clean pattern like this
one is a lucky find, not a guarantee.

## Putting it together

A convincing explanation usually combines these: the sweep says *which* components matter, attention
patterns and neuron inspection suggest *what* each one does, patching confirms *where* the
information flows, and a final check is to predict the model's behaviour on new prompts from your
explanation. When your story survives that, you've found a circuit.
