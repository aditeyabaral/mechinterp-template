# Reading list

A path through the ideas this template puts into practice, roughly in the order to read them. Each
entry says why it matters here. You don't need all of it to use the template; [concepts](concepts.md)
covers the essentials.

## Start here

- **[A Comprehensive Mechanistic Interpretability Explainer & Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)**
  (Neel Nanda): definitions of nearly every term you'll meet, with intuitions. Keep it open while
  reading the papers below.
- **[Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)** (Olah et al.,
  2020): the case for studying networks as circuits of understandable features, which is the premise
  of this whole field. Written about vision models, but the ideas carry over.
- **[How To Become A Mechanistic Interpretability Researcher](https://www.neelnanda.io/mechanistic-interpretability/getting-started)**
  (Neel Nanda): a practical guide to learning the field and starting your own projects.

## Foundations: how transformers compute

- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)**
  (Elhage et al., 2021): the residual-stream view and attention-head analysis that this template's
  vocabulary (residual stream, heads reading and writing) comes from.
- **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)**
  (Olsson et al., 2022): the classic example of a small circuit found in small models: two heads that
  together let a model continue repeated patterns.
- **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)** (Elhage
  et al., 2022): why a single neuron doesn't always correspond to a single human-readable feature,
  and what to expect when one of your "important" neurons doesn't have a clean story.

## Worked examples on small models

- **[Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)**
  (Nanda et al., 2023): a complete reverse-engineering of a small model trained on modular addition,
  including how the circuit forms during grokking. The closest in spirit to this template.
- **[Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)**
  (Wang et al., 2022): a circuit of more than 20 heads in a real model, found with the path-patching
  version of the techniques in [after the sweep](after-the-sweep.md).

## Methods for causal analysis

- **[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)** (Meng et al.,
  2022, "ROME"): causal tracing and targeted editing in a real model, the same style of evidence this
  template produces at toy-model scale.
- **[Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042)**
  (Zhang & Nanda, 2023): how the choice of corruption and metric changes patching and ablation
  results; read it before trusting any single ablation type.
- **[How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)** (Heimersheim &
  Nanda, 2024): a practical guide to patching experiments and the mistakes to avoid.
- **[Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)**
  (Conmy et al., 2023, "ACDC"): automating what this template does by hand: finding the edges between
  components, not just the components.
- **[Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452)**
  (Belinkov, 2021): what probes like `lasso.py`'s can and can't tell you.
- **[interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)**
  (nostalgebraist, 2020): reading the residual stream through the output layer at every layer, a
  cheap way to see when the model "knows" the answer.

## Courses and tools

- **[ARENA](https://www.arena.education/)**: a hands-on, exercise-based curriculum built on
  TransformerLens; a natural next step once you're comfortable with this template's workflow.
- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** ([documentation](https://transformerlensorg.github.io/TransformerLens/)):
  the library this template uses to read and edit model internals; its tutorials are a friendly
  on-ramp to the field.

## Open problems: where to go next

- **[200 Concrete Open Problems in Mechanistic Interpretability](https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj)**
  (Neel Nanda, 2022): project ideas sorted by difficulty, many of them doable with a toy model like
  yours.
- **[Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)** (Sharkey et al.,
  2025): a survey of what the field still can't do, from many of its researchers.
