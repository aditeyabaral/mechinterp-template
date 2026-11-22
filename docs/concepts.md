# Concepts: a mechinterp primer

This page assumes you've seen the transformer architecture once (you roughly know what "attention"
and "layers" are) but are **new to interpretability**. It defines every concept the rest of the
documentation uses.

- [The big picture](#the-big-picture)
- [What this template does and doesn't do](#what-this-template-does-and-doesnt-do)
- [Concepts you'll need](#concepts-youll-need)
- [The three components we capture, side by side](#the-three-components-we-capture-side-by-side)
- [The two moves](#the-two-moves)
- [Zero vs mean ablation](#zero-vs-mean-ablation)
- [Probes: where information can be read vs what the model uses](#probes-where-information-can-be-read-vs-what-the-model-uses)

## The big picture

A language model takes some text and predicts the next token. Standard machine learning measures
*whether* it gets the answer right. **Mechanistic interpretability** asks the harder question:
**how, internally, does it do that?** It tries to reverse-engineer the actual step-by-step
algorithm the network learned, in terms of its concrete parts (neurons, attention heads), rather
than treating the model as a black box. The goal is to find the small "circuit" of components that
implements a given behaviour.

### How mechinterp research usually gets started

Real models like the Llama family have billions of parameters spread across dozens of layers: far
too much to read by hand. So the field's standard entry point is to go small *on purpose*: pick a
narrow, well-understood task, train (or find) the smallest model that can do it, and study that
model in full. Two things make this tractable in a way the full-scale version isn't:

- **You know the ground truth.** Because *you* designed the task and generated the training data,
  you know exactly what algorithm would solve it, giving you something concrete to check the
  model's internals against.
- **The model is small enough to look at everything.** A few layers and a handful of heads means you
  can inspect *every* neuron and *every* head, not just a sample, and causal interventions
  (switching a component off and re-running) are cheap enough to try on all of them.

This "design a task → train a toy model → intervene on it" recipe is how some of the field's
best-known results were found, e.g. induction heads (discovered in tiny attention-only
transformers) and the grokking/modular-addition circuit (see the [reading list](reading.md)).
This template packages exactly that recipe into a runnable pipeline:

1. You choose a small, well-defined task (arithmetic, sorting, copying, parity, ...).
2. You generate synthetic training data and **train a small GPT-2 from scratch** on it.
3. You run the trained model on probe prompts and **record its internal activations**.
4. You **find** the neurons and attention heads that seem to carry the work, then **switch them off**
   to test whether the model really needs them.

If zeroing out a particular neuron reliably breaks the behaviour, you have *causal* evidence that the
neuron is part of the mechanism. The template takes you exactly this far: a ranked set of components
that demonstrably matter, and by how much. Working out *what each one actually computes*, and how they
combine into a "circuit," is the open-ended, task-specific part you do from there (there's no
push-button method for it; that's the interesting research). [After the sweep](after-the-sweep.md)
gives you a head start.

## What this template does and doesn't do

> [!IMPORTANT]
> It identifies the **causal components**: it tells you *which* neurons and attention heads matter
> for your task, and *how much* (by measuring what breaks when you switch them off). It does **not**
> explain *how* a given component does its job. Turning "this head is important" into "this head
> does *X*" is task-specific, creative work with no standard recipe in mechinterp; that
> interpretation is the research you do next. The template's job is to get you reliably to that
> starting line.

Everything generic already works; the task-specific parts are left as clearly marked `TODO`s for you
to fill in. The template is built around **GPT-2**, the smallest, best-studied open architecture in
this field, and uses [**TransformerLens**](https://github.com/TransformerLensOrg/TransformerLens) to
read and edit the model's internals, so the same analysis code runs on GPT-2 or any toy model you
train, just by changing `--model-path`.

## Concepts you'll need

A decoder-only transformer (GPT-2 is one) reads a sequence of **tokens** and pushes them through a
stack of identical **blocks** (layers). Here are the terms used throughout this repo, with light
definitions, just enough to follow along:

- **Token**: a small chunk of text (a character, word-piece, or symbol) the model treats as one
  unit. The model reads and writes sequences of tokens.

- **Activation**: any intermediate vector of numbers the model computes while running. "Capturing
  activations" means saving these vectors so we can study them.

- **Residual stream**: the model keeps one running vector per token position that flows through every
  block from start to finish; think of it as that token's evolving "notes." Each block reads it and
  writes an update back into it. Its size is `hidden_size` (768 for GPT-2). After the last block, the
  residual vector at the final position is turned into the next-token prediction.

- **MLP**: one of the two parts inside every block, a small feed-forward network applied to each
  position on its own. It widens the vector to a larger hidden layer, applies a non-linearity, then
  shrinks it back.

- **Neuron**: a single entry of that wide MLP hidden layer (there are `intermediate_size = 3072` of
  them per layer in GPT-2). Each neuron is just one number that turns on or off depending on the
  input. Neurons are a favourite unit of study because individual ones sometimes fire for a clean,
  human-readable feature (e.g. "this is a number" or "the previous token was a verb").

- **Attention head**: the other part of every block is *attention*, which lets a position pull in
  information from earlier positions. Attention is split into several independent **heads**, each
  attending to a different mix of earlier tokens. Heads often implement reusable operations (a famous
  example is an "induction head" that continues a repeated pattern). Each head outputs a small vector
  of size `head_dim` (64 for GPT-2), and there are `num_heads = 12` per layer.

- **Token embedding**: the raw vector a token starts as, before any block processes it (we store this
  under layer index `-1`).

- **Hook**: a small function the model calls whenever a particular activation is computed. We use
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) hooks to read out
  activations (without changing anything) or to edit them, e.g. to zero out a neuron. Each
  activation has a stable *name* (like `blocks.5.mlp.hook_post`), so the same code works on any
  supported model.

- **Ablation**: an intervention where we force some activations (a neuron or a head) to **zero** (or
  to their average value) and see if the output changes. If it does, that component mattered; this
  is our causal test. See [zero vs mean ablation](#zero-vs-mean-ablation).

- **Logits**: the raw scores over the whole vocabulary that get turned into next-token probabilities.

- **Lasso**: a linear model with an "L1 penalty." Given many activations, it predicts a target (here
  a class, such as the next answer token: a *probe*) and pushes most of its weights to exactly zero,
  leaving a **short list** of the activations it relies on, ranked by weight. We use it to shortlist
  which neurons/heads are worth ablating. (You don't need the math; just know it gives you a ranked
  shortlist.) See [probes](#probes-where-information-can-be-read-vs-what-the-model-uses).

- **Grokking**: a striking training phenomenon where a small model first *memorises* its training
  data and then, often much later, *suddenly* starts to *generalise*. Watching the circuit form during
  grokking is a classic mechinterp experiment (see the
  [note on grokking](building-a-model.md#3-train-the-toy-model)).

- **Circuit**: the small set of components (and the connections between them) that together carry
  out one behaviour, e.g. "these two heads move the operands to the `=` sign, and this neuron decides
  whether the sum carries." Finding which components belong to it is what this template does;
  explaining how they work together is the research after.

- **Teacher forcing**: feeding the model the *correct* answer as input and checking, at every
  position, whether its top prediction is the right next token. One forward pass scores a whole
  answer, which is how the ablation sweep stays fast (see
  [how the sweep is scored](analysing-a-model.md#how-the-sweep-is-scored-teacher-forcing)).

- **Superposition**: a model can represent more features than it has neurons by spreading each
  feature over many neurons, and packing several features into each. A consequence: a neuron that
  matters doesn't always have one clean meaning (see *Toy Models of Superposition* in the
  [reading list](reading.md)).

## The three components we capture, side by side

Of all those concepts, three are the actual things this template records and lets you switch off.
They all live inside every transformer block, and each is a candidate "unit" of the model's
computation you can study:

| Component | What it is | Shape (per token, in GPT-2) | Why study it |
| --- | --- | --- | --- |
| **Residual stream** | the block's output vector, a token's running "notes" | `[768]` | the model's overall state at that depth |
| **MLP neurons** | the wide hidden layer inside the block's MLP | `[3072]` | a single neuron sometimes encodes one clean, readable feature |
| **Attention heads** | each attention head's output | `[12, 64]` = heads × head_dim | a head often implements a reusable operation (e.g. "copy an earlier token") |

(The **token embedding**, the input vector before any block runs, is also captured, stored under
layer index `-1`.)

## The two moves

Everything in this template is one of two operations on those components:

- **Extraction (observe):** attach hooks that copy activations out. Changes nothing about the model;
  we're just watching.
- **Ablation (intervene):** attach hooks that force chosen activations to zero (or their average),
  then watch how the answer changes. This is the causal test that turns a correlation ("this neuron
  is active when…") into evidence ("the model *needs* this neuron to…").

## Zero vs mean ablation

Ablating a component means replacing its output with something that carries no information about the
current prompt. The question is *what* to replace it with, and the choice changes what a drop means.

- **Zero ablation** (`--ablation-type zero`, the default) sets the output to 0. It is the simplest
  knock-out, but 0 is often a value the component *never* produces. The later layers then see an
  input unlike anything in training, and the model can break for that reason alone. A large drop
  under zero ablation can mean "the model needs this component's information" *or* "the model can't
  cope with this component suddenly reading 0".
- **Mean ablation** (`--ablation-type mean`) sets the output to its *average* over every token of
  every prompt and its answer. The later layers still get a typical value, so what disappears is only
  what the component says about *this particular* prompt. A drop under mean ablation is the cleaner
  evidence that the model uses the component's prompt-specific information.

In practice, run both. Components that matter under both are the safest bets. A component that
matters only under zero ablation is a warning sign: its "importance" may just be the shock of an
unusual value. Neither method tells you *what* the component computes; for that, see
[after the sweep](after-the-sweep.md). The activation-patching papers in the
[reading list](reading.md#methods-for-causal-analysis) discuss these choices in depth.

## Probes: where information can be read vs what the model uses

A **probe** is a small classifier trained to predict something (e.g. the next answer token) from a
layer's activations. If it succeeds, that information is *present* there, in a form a linear model can
read. It does **not** show that the model *uses* it: the information could be a by-product that no
later layer reads. That's why the lasso's shortlist is a set of *candidates*, and the ablation sweep
is the test.

Two habits keep probes honest, and `lasso.py` follows both:

- **Score on held-out prompts.** A probe with thousands of features can memorise its training rows.
  `lasso.py` fits on 80% of the prompts and reports accuracy on the other 20%.
- **Compare with chance.** On a skewed target, always guessing the most common class already scores
  well. `lasso.py` reports that baseline and shortlists nothing unless the probe clearly beats it.

*Probing Classifiers: Promises, Shortcomings, and Advances* in the
[reading list](reading.md#methods-for-causal-analysis) covers the pitfalls in more depth.
