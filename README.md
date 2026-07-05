# Mechanistic Interpretability Template

A starting point for **mechanistic interpretability** ("mechinterp") projects on small transformer
models. The intended workflow is:

> [!NOTE]
> **Pick a task → train a tiny ("toy") transformer on it from scratch → find which of its internal
> components causally drive the behaviour**, by recording its activations and switching pieces off.

Because you design the task and train the model yourself, you know *exactly* what behaviour you're
explaining — and a tiny model is small enough to actually understand. That control is what makes toy
models such a good interpretability playground.

> [!IMPORTANT]
> **What this template does and doesn't do.** It identifies the **causal components**: it tells
> you *which* neurons and attention heads matter for your task, and *how much* (by measuring what
> breaks when you switch them off). It does **not** explain *how* a given component does its job.
> Turning "this head is important" into "this head does *X*" is task-specific, creative work with no
> standard recipe in mechinterp — that interpretation is the research you do next. The template's job
> is to get you reliably to that starting line.

Everything generic already works; the task-specific parts are left as clearly marked `TODO`s for you
to fill in. The template is built around **GPT-2**, the smallest, best-studied open architecture in
this field, and uses [**TransformerLens**](https://github.com/TransformerLensOrg/TransformerLens) to
read and edit the model's internals — so the same analysis code runs on GPT-2 or any toy model you
train, just by changing `--model-path`.

This README assumes you've seen the transformer architecture once (you roughly know what "attention"
and "layers" are) but are **new to interpretability**. It defines every concept you need.

## Contents

- [The big picture](#the-big-picture)
- [Concepts you'll need (a short primer)](#concepts-youll-need-a-short-primer)
- [The pipeline at a glance](#the-pipeline-at-a-glance)
- [Setup](#setup)
- [Project structure](#project-structure)
- [The workflow, step by step](#the-workflow-step-by-step)
- [The capture convention (which token we read)](#the-capture-convention-which-token-we-read)
- [What gets saved (and how to look at it)](#what-gets-saved-and-how-to-look-at-it)
- [Everything you need to fill in](#everything-you-need-to-fill-in)
- [Tips and common pitfalls](#tips-and-common-pitfalls)
- [Shortcut: analysing a pretrained model](#shortcut-analysing-a-pretrained-model)
- [Using a non-GPT-2 model](#using-a-non-gpt-2-model)
- [Further reading](#further-reading)
- [Acknowledgements](#acknowledgements)

## The big picture

A language model takes some text and predicts the next token. Standard machine learning measures
*whether* it gets the answer right. **Mechanistic interpretability** asks the harder question:
**how, internally, does it do that?** — i.e. it tries to reverse-engineer the actual step-by-step
algorithm the network learned, in terms of its concrete parts (neurons, attention heads), rather
than treating the model as a black box. The goal is to find the small "circuit" of components that
implements a given behaviour.

### How mechinterp research usually gets started

Real models like the Llama family have billions of parameters spread across dozens of layers — far
too much to read by hand. So the field's standard entry point is to go small *on purpose*: pick a
narrow, well-understood task, train (or find) the smallest model that can do it, and study that
model in full. Two things make this tractable in a way the full-scale version isn't:

- **You know the ground truth.** Because *you* designed the task and generated the training data,
  you know exactly what algorithm would solve it — giving you something concrete to check the
  model's internals against.
- **The model is small enough to look at everything.** A few layers and a handful of heads means you
  can inspect *every* neuron and *every* head, not just a sample, and causal interventions
  (switching a component off and re-running) are cheap enough to try on all of them.

This "design a task → train a toy model → intervene on it" recipe is how some of the field's
best-known results were found — e.g. induction heads (discovered in tiny attention-only
transformers) and the grokking/modular-addition circuit (see [Further reading](#further-reading)).
This template packages exactly that recipe into a runnable pipeline:

1. You choose a small, well-defined task (arithmetic, sorting, copying, parity, ...).
2. You generate synthetic training data and **train a small GPT-2 from scratch** on it.
3. You run the trained model on probe prompts and **record its internal activations**.
4. You **find** the neurons and attention heads that seem to carry the work, then **switch them off**
   to test whether the model really needs them.

If zeroing out a particular neuron reliably breaks the behaviour, you have *causal* evidence that the
neuron is part of the mechanism. The template takes you exactly this far — a ranked set of components
that demonstrably matter, and by how much. Working out *what each one actually computes*, and how they
combine into a "circuit," is the open-ended, task-specific part you do from there (there's no
push-button method for it — that's the interesting research).

## Concepts you'll need (a short primer)

A decoder-only transformer (GPT-2 is one) reads a sequence of **tokens** and pushes them through a
stack of identical **blocks** (layers). Here are the terms used throughout this repo — light
definitions, just enough to follow along:

- **Token** — a small chunk of text (a character, word-piece, or symbol) the model treats as one
  unit. The model reads and writes sequences of tokens.

- **Activation** — any intermediate vector of numbers the model computes while running. "Capturing
  activations" means saving these vectors so we can study them.

- **Residual stream** — the model keeps one running vector per token position that flows through every
  block from start to finish; think of it as that token's evolving "notes." Each block reads it and
  writes an update back into it. Its size is `hidden_size` (768 for GPT-2). After the last block, the
  residual vector at the final position is turned into the next-token prediction.

- **MLP** — one of the two parts inside every block: a small feed-forward network applied to each
  position on its own. It widens the vector to a larger hidden layer, applies a non-linearity, then
  shrinks it back.

- **Neuron** — a single entry of that wide MLP hidden layer (there are `intermediate_size = 3072` of
  them per layer in GPT-2). Each neuron is just one number that turns on or off depending on the
  input. Neurons are a favourite unit of study because individual ones sometimes fire for a clean,
  human-readable feature (e.g. "this is a number" or "the previous token was a verb").

- **Attention head** — the other part of every block is *attention*, which lets a position pull in
  information from earlier positions. Attention is split into several independent **heads**, each
  attending to a different mix of earlier tokens. Heads often implement reusable operations (a famous
  example is an "induction head" that continues a repeated pattern). Each head outputs a small vector
  of size `head_dim` (64 for GPT-2), and there are `num_heads = 12` per layer.

- **Token embedding** — the raw vector a token starts as, before any block processes it (we store this
  under layer index `-1`).

- **Hook** — a small function the model calls whenever a particular activation is computed. We use
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) hooks to read out
  activations (without changing anything) or to edit them — e.g. to zero out a neuron. Each
  activation has a stable *name* (like `blocks.5.mlp.hook_post`), so the same code works on any
  supported model.

- **Ablation** — an intervention where we force some activations (a neuron or a head) to **zero** and
  see if the output changes. If it does, that component mattered — this is our causal test.

- **Logits** — the raw scores over the whole vocabulary that get turned into next-token probabilities.

- **Lasso** — a linear model with an "L1 penalty." Given many activations, it predicts a target and
  drives almost all of its weights to exactly zero, leaving a **short list** of the activations that
  actually matter. We use it to shortlist which neurons/heads are worth ablating. (You don't need the
  math — just know it gives you a ranked shortlist.)

- **Grokking** — a striking training phenomenon where a small model first *memorises* its training
  data and then, often much later, *suddenly* starts to *generalise*. Watching the circuit form during
  grokking is a classic mechinterp experiment (see the training note below).

### The three components we capture, side by side

Of all those concepts, three are the actual things this template records and lets you switch off.
They all live inside every transformer block, and each is a candidate "unit" of the model's
computation you can study:

| Component | What it is | Shape (per token, in GPT-2) | Why study it |
| --- | --- | --- | --- |
| **Residual stream** | the block's output vector — a token's running "notes" | `[768]` | the model's overall state at that depth |
| **MLP neurons** | the wide hidden layer inside the block's MLP | `[3072]` | a single neuron sometimes encodes one clean, readable feature |
| **Attention heads** | each attention head's output | `[12, 64]` = heads × head_dim | a head often implements a reusable operation (e.g. "copy an earlier token") |

(The **token embedding** — the input vector before any block runs — is also captured, stored under
layer index `-1`.)

### The two moves

Everything in this template is one of two operations on those components:

- **Extraction (observe):** attach hooks that copy activations out. Changes nothing about the model —
  we're just watching.
- **Ablation (intervene):** attach hooks that force chosen activations to zero, then watch how the
  answer changes. This is the causal test that turns a correlation ("this neuron is active when…")
  into evidence ("the model *needs* this neuron to…").

## The pipeline at a glance

Steps 1–3 build your toy model; steps 4–7 study it. Each name is a script you run, and each arrow is
what it produces and hands to the next one:

```
  train_tokenizer.py  ──▶  create_dataset.py  ──▶  train_model.py  ──▶  your toy model
   (build a tokenizer)      (make training data)    (train it)          (saved locally, or to the Hub)
                                                                                │
                                                                                ▼
  PromptDataset  ──▶  main.py  ──▶  results file  ──▶  lasso.py  ──▶  analysis.json
   (probe prompts)    (capture)     (activations)       (rank parts)    (what to ablate)
        │                                                                     │
        └──────────▶  main.py --intervention  ◀───────────────────────────────┘
                      (switch the parts off, save baseline + ablated runs)
                              │
                              ▼
                      plot_ablations.py  (visualise the drops: heatmaps + scatter)
```

Follow it like a snake: build the model along the top, drop down and run it on your own prompts to
**record what happens inside**, **shortlist** the neurons/heads that look important, then feed that
shortlist back into `main.py` to **switch them off** and see how the answers change. The last two
steps (`lasso.py` and `--intervention`) are optional — you can also stop after capturing and explore
the saved activations yourself.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync                       # create the virtual environment and install dependencies
uv run python src/main.py -h  # check it works (prints the command-line options)
```

(You can also call the environment directly, e.g. `.venv/bin/python src/main.py -h`.)

Training saves your tokenizer/dataset/model locally by default — no account needed. If you'd
rather push them to the Hugging Face Hub (pass `--hub-name`), log in once:

```bash
uv run huggingface-cli login
```

## Project structure

```
src/
  train/                 # build your toy model
    train_tokenizer.py   #   build a character-level tokenizer            [fill in vocab]
    create_dataset.py    #   generate a synthetic training dataset        [fill in build_dataset]
    train_model.py       #   train a small GPT-2, saved locally by default [fill in repo names]
  model.py               # load_model(): boots the model into a TransformerLens bridge (generic)
  utils/
    dataset.py           # PromptDataset: the probe prompts you run            [fill in — required]
    parser.py            # command-line arguments for main.py                  [optional extra args]
    dir.py               # builds the output filename                          [optional]
    scoring.py           # is_correct(): whether a generated answer is right   [optional]
    ablations.py         # shared loaders for intervention .pt files (used by the plot scripts)
  inference.py           # the activation-capture loop — the heart of the analysis
  main.py                # ENTRY POINT: run capture (+ optional intervention) and save results
  lasso.py               # OPTIONAL: rank neurons/heads by importance, write analysis.json
  plot_ablations.py      # OPTIONAL: heatmaps + cross-run scatter of ablation accuracy drops
  plot_circuits.py       # OPTIONAL: circuit diagrams of the surviving components per setting
```

The two files you'll spend the most time *reading* are **`inference.py`** (how activations are
captured) and **`main.py`** (how a run is orchestrated). They are heavily commented.

## The workflow, step by step

> [!NOTE]
> **Local by default.** `--hub-name` is optional for `train_tokenizer.py`, `create_dataset.py`, and
> `train_model.py` — omit it and the output is saved locally (`./artifacts/tokenizer`,
> `./artifacts/dataset`, `--output-dir` for the model, default `./saved_models`) with no Hugging
> Face login needed. Pass `--hub-name <user>/...` to push to the Hub instead, and point `--model-path`
> (step 5) at whichever location you used.

### 1. Build a tokenizer

A **tokenizer** maps text to integer ids. For small synthetic tasks the simplest, most transparent
choice is one token per character. Set `VOCAB_CHARS` (every character your task can produce) and
`SAMPLE_TEXTS` in `src/train/train_tokenizer.py`, then:

```bash
uv run python src/train/train_tokenizer.py --hub-name <user>/my-tokenizer
```

### 2. Generate a dataset

Implement `build_dataset` in `src/train/create_dataset.py` to produce a Hugging Face dataset with
`prompt` and `answer` columns. The file documents a *leakage-safe* recipe (so your test examples
never sneak into training).

```bash
uv run python src/train/create_dataset.py --hub-name <user>/my-dataset
```

### 3. Train the toy model

Set `TOKENIZER_NAME` and `DATASET_NAME` at the top of `src/train/train_model.py`, then train. You
control the size with flags like `--num-hidden-layers`, `--num-attention-heads`, `--hidden-size`:

```bash
uv run python src/train/train_model.py --hub-name <user>/my-model --num-epochs 5
```

> [!NOTE]
> **A note on grokking.** Training defaults to a strong weight decay (`--weight-decay 1.0`). Heavy
> weight decay is a well-known trigger for **grokking** — memorise first, generalise suddenly later.
> Watching a circuit form during grokking is a classic experiment, so it's on by default. Train for
> plenty of steps to give it a chance to appear, or lower `--weight-decay` to turn it down.

> [!TIP]
> **Smaller is better for interpretability.** A model with a couple of layers and a handful of heads
> is *far* easier to fully understand than full GPT-2. Start as small as the task allows.

### 4. Define your probe prompts

Open `src/utils/dataset.py` and implement `PromptDataset.generate_prompts`. A prompt is just the
input string you feed the model; end it right where you want the answer to begin (e.g. `"7+5="`).
These prompts *are* your experiment — they decide what behaviour you get to study. **This is the one
thing you must implement to run the analysis.** A self-contained example (single-digit addition) is
included in comments — uncomment it to see the pipeline run end to end, then replace it.

### 5. Capture activations

```bash
uv run python src/main.py -m <user>/my-model --num-prompts 200 --capture-geometry
```

This runs your model on each prompt, records the activations at the answer, and saves a `.pt` file in
the current directory (its name is generated automatically from the run's parameters; pass
`--output DIR` to choose where it goes). Useful options (`python src/main.py -h` lists them all):

| Option | Meaning |
| --- | --- |
| `-m, --model-path` | model to load (your HF repo id, or a local path) — **required** |
| `-p, --num-prompts` | how many prompts to run (default 1000) |
| `-l, --layers` | which layer indices to record from (default: all) |
| `--capture-geometry` | record the activations: residual stream, MLP neurons, heads, embeddings. **Without this flag only the answer token and its logits are saved — pass it for any real analysis (`lasso.py` needs it).** |
| `--max-new-tokens` | how many tokens to generate per prompt (default 200) |
| `--intervention FILE` | run an ablation sweep from an `analysis.json` (step 7) |

### 6. Find the important components *(optional)*

```bash
uv run python src/lasso.py --dir . --output analysis.json
```

Reads the `.pt` files, fits the Lasso, and writes `analysis.json` — per layer and per condition, the
shortlist of important neurons/heads, each with its Lasso **weight** (signed effect size), plus a
top-level `conditions` block recording how many rows backed each condition. (See the `lasso.py`
module docstring for the exact JSON layout.)

### 7. Ablate them *(optional)*

```bash
uv run python src/main.py -m <user>/my-model --num-prompts 200 --intervention analysis.json
```

Re-runs the prompts, switching off each important component in turn, and for each one records the
**accuracy drop** it causes (`baseline_accuracy - ablated_accuracy`) into a single `.pt` file — the
bigger the drop, the more the model relied on that component. Accuracy is scored by `is_correct` in
`src/utils/scoring.py` (it compares the model's answer to the `"answer"` in each prompt's metadata;
adjust it for your task). `src/plot_ablations.py` turns those drops into figures. Like `lasso.py`, it reads a
whole **directory** of intervention runs — each `.pt` in it is one *setting* — and reads the layer
count, head count and MLP size from the files themselves, so there is nothing else to pass:

```bash
uv run python src/plot_ablations.py --dir <dir with your intervention .pt files> --output plots
```

Into `plots/` it writes:

- **per-run heatmaps** — for each run, a `layer × MLP-neuron` and a `layer × attention-head` heatmap
  of the accuracy drop (grey = a component that run didn't ablate);
- **across-run comparisons** *(only when the directory has ≥2 runs)* — `setting × component` heatmaps
  (read one neuron/head *down a column* to compare it across runs) and, for every pair of runs, a
  **scatter** placing each component at (`its drop in run A`, `its drop in run B`): points on the
  `y=x` line matter equally in both runs, points off it matter in one but not the other. A point's
  shape marks its type (circle = neuron, star = head) and its colour marks its layer.

So a single intervention file already gives you the two per-run heatmaps; to get the comparisons,
capture-and-ablate under two conditions (say, two prompt distributions or two models) and drop both
`.pt` files in the same directory.

`src/plot_circuits.py` reads the same directory of intervention runs and draws a **circuit diagram**
instead: one column of nodes per layer (all MLP neurons + attention heads), with the components in
the top-p percentile of accuracy drop ("survivors") coloured and connected to the survivors of the
next layer. It writes `S + 1` images — one `circuit_<setting>.png` per setting plus a
`circuit_combined.png` overlaying every setting's edges in its own colour — and rings the components
that survive in *every* setting (the shared-circuit candidates):

```bash
uv run python src/plot_circuits.py --dir <dir with your intervention .pt files> --output plots --percentile 99
```

## The capture convention (which token we read)

A model generates an answer as a sequence of tokens, and a prompt has many token positions. Where do
we read activations? **Always at the last token of the answer.**

If the model answers `42` and that is two tokens `['4', '2']`, we read at `'2'`. If it answers `-63`
as `['-', '6', '3']`, we read at `'3'`. The intuition: by the answer's last token the model has
committed to its answer, so that position's residual stream is the most informative single place to
look. Every recorded tensor (residual, MLP neurons, attention heads, and the token / positional
embeddings) is read at this *same* position, so they all describe the same token.

> [!NOTE]
> How this works under the hood: after generating, we run the *whole* output sequence (prompt +
> generation) back through the model in a single pass and read its activation cache. Because that one
> pass covers every position, the answer's last token always has its activations computed — there's no
> "ran out of tokens" edge case to worry about, whatever `--max-new-tokens` you choose.

Which substring counts as "the answer" is decided by `find_answer_span` in `src/inference.py`. The
default grabs the first whitespace-delimited chunk of the generation; override it for your task.

## What gets saved (and how to look at it)

A run **with `--capture-geometry`** saves a dictionary like this (tensors are PyTorch tensors on the
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
        "completion": "12",          # what the model generated
        "answer": {
          "token": "12",             # the answer (string)
          "position": 5,             # absolute index of its LAST token
          "residual":        {0: Tensor[768],  1: ...},   # per layer
          "mlp_neurons":     {0: Tensor[3072], 1: ...},   # per layer (the neurons)
          "attn_heads":      {0: Tensor[12, 64], ...},    # per layer (per head)
          "token_embedding": Tensor[768],   # content embedding at the answer token
          "pos_embedding":   Tensor[768],   # positional embedding at the answer token
          "resid_pre":       Tensor[768],   # block 0's input (token + positional embedding)
          "logits":          Tensor[50257],   # next-token distribution at the answer
        },
        "geometry": { ... },          # extra positions from find_positions_of_interest
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
data = torch.load("your_results.pt", weights_only=False)   # the .pt file main.py wrote
row = data["baseline"][0]
print(row["prompt"], "->", row["result"]["answer"]["token"])
answer = row["result"]["answer"]
layer = data["metadata"]["layer_indices"][0]                # a layer you actually captured
neurons = answer["mlp_neurons"][layer]                      # MLP neurons at the answer token
print(neurons.shape)                                        # e.g. torch.Size([3072]) for GPT-2
```

An **intervention** run (step 7) saves a slightly different layout — `{"baseline": [rows],
"baseline_accuracy": float, "ablations": [{"layer_idx", "feature_idx", "type", "local_idx",
"accuracy", "accuracy_drop"}, ...], "metadata": {...}}` — where each ablation records only the
*scalar* accuracy and its drop versus baseline (the heavy per-ablation rows are discarded; the
unablated `baseline` is kept once). `src/plot_ablations.py` reads these files and turns the drops
into heatmaps and cross-run scatter plots (step 7).

## Everything you need to fill in

Every spot you might edit is marked with a `TODO`. List them all at any time:

```bash
grep -rn TODO src/
```

| File | What to implement | When |
| --- | --- | --- |
| `src/train/train_tokenizer.py` | `VOCAB_CHARS`, `SAMPLE_TEXTS` | building the model |
| `src/train/create_dataset.py` | `build_dataset` — your training data | building the model |
| `src/train/train_model.py` | `TOKENIZER_NAME`, `DATASET_NAME` | building the model |
| `src/utils/dataset.py` | `PromptDataset.generate_prompts` — the probe prompts | **required to analyse** |
| `src/inference.py` | `find_answer_span` — which substring is "the answer" | optional (good default) |
| `src/inference.py` | `find_positions_of_interest` — extra prompt positions to record | optional (default: none) |
| `src/utils/parser.py` | extra task-specific command-line arguments | optional |
| `src/utils/dir.py` | `generate_output_path` — the output filename | optional (good default) |
| `src/lasso.py` | `assign_condition`, `build_target` — what to compare/predict | optional (good defaults) |
| `src/utils/scoring.py` | `is_correct` — whether a generated answer is right (scores ablation accuracy drops) | optional (good default) |

Each `TODO` explains *what* to do, *why*, and shows a worked example in comments. Once your model is
trained, the analysis side **runs as soon as you implement `PromptDataset.generate_prompts`** — every
other extension point has a working default.

## Tips and common pitfalls

- **Always pass `--capture-geometry` for analysis.** Without it, *no* activations are recorded — you
  only get the generated answer and its logits. (That lightweight mode exists to make large ablation
  sweeps fast, where you only care about how the answer changes.)
- **Keep answers short** (one or two tokens). Long, variable answers make "the answer position" fuzzy
  and the analysis noisier.
- **Start with a tiny model.** Fewer layers and heads means fewer things to understand.
- **Reproducibility is built in:** `main.py` seeds all RNGs from `--seed`, so reruns are identical.
- **GPU vs CPU:** the model is placed automatically; everything also runs on CPU, just slower.

## Shortcut: analysing a pretrained model

You don't *have* to train your own model. To skip steps 1–3 and study an existing GPT-2-style
checkpoint instead, point `--model-path` at any HF repo id or local path:

```bash
uv run python src/main.py -m gpt2 --num-prompts 200 --capture-geometry
```

The rest of the analysis (steps 4–7) is identical. (Training your own model is recommended for
learning, because you control and understand the task completely.)

## Using a non-GPT-2 model

There is **no architecture-specific code to edit**. The model is loaded through
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)'s `TransformerBridge`
(`src/model.py`), which wraps any supported decoder-only HuggingFace model and re-exposes it with
*uniform* hook names (`blocks.{i}.hook_resid_post`, `blocks.{i}.mlp.hook_post`,
`blocks.{i}.attn.hook_z`, `hook_embed`). The capture loop and ablation code in `src/inference.py`
reference only those names, so switching models is just a matter of pointing `--model-path` at a
different Hub repo or local directory — the bridge reads the new model's config and `cfg.n_layers /
n_heads / d_head` update automatically.

## Further reading

To go deeper into the ideas this template puts into practice:

- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021) — the residual-stream
  view and attention-head analysis.
- **Neel Nanda's TransformerLens tutorials** and **"200 Concrete Open Problems in Mechanistic
  Interpretability"** — a friendly on-ramp to the field.
- **Progress measures for grokking via mechanistic interpretability** (Nanda et al., 2023) — a worked
  example of reverse-engineering a grokked toy model (very close in spirit to this template).
- **Toy Models of Superposition** (Elhage et al., 2022) — why a single neuron doesn't always
  correspond to a single human-readable feature, and what to expect when one of your "important"
  neurons doesn't have a clean story.
- **Locating and Editing Factual Associations in GPT** (Meng et al., 2022, "ROME") — causal tracing
  and targeted ablation/editing applied to a real model, the same style of evidence this template
  produces at toy-model scale.
- **ARENA: Mechanistic Interpretability curriculum** (Callum McDougall et al.) — a hands-on,
  exercise-based course built on TransformerLens; a natural next step once you're comfortable with
  this template's workflow.

## Acknowledgements

This template is built entirely on top of the open-source mechanistic interpretability and
machine-learning ecosystem:

- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** (originally by Neel
  Nanda, now maintained by the TransformerLens community) provides the hook-based interface this
  template uses for all activation capture and ablation.
- **Hugging Face's `transformers`, `datasets`, `tokenizers`, and `accelerate`** libraries provide the
  tokenizer, dataset, and training infrastructure in `src/train/`.
- **GPT-2** (Radford et al., 2019, OpenAI) is the architecture every toy model in this template is
  based on.
- The overall workflow — train a toy model on a task you understand, then use causal interventions to
  find the circuit behind it — follows the approach popularised by the mechanistic interpretability
  research community (see [Further reading](#further-reading)).
