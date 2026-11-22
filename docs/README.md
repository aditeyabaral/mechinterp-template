# MechInterpLens documentation

| Page | What it covers |
| --- | --- |
| [Concepts](concepts.md) | A mechinterp primer: the big picture, every term the template uses, zero vs mean ablation, and what probes can and can't tell you |
| [Building a model](building-a-model.md) | Steps 1–3: the tokenizer, the dataset and training |
| [Analysing a model](analysing-a-model.md) | Steps 4–7: prompts, capturing activations, the lasso shortlist, ablation sweeps and their plots |
| [Outputs](outputs.md) | Which token positions are recorded, what each run saves, and how to load it |
| [Customising](customising.md) | What to fill in for your task, tips and pitfalls, pretrained and non-GPT-2 models |
| [After the sweep](after-the-sweep.md) | Reading the results critically, and techniques for finding out *how* a component works |
| [Reading list](reading.md) | Papers, guides and courses, in the order to read them |

## Where to start

- **New to mechanistic interpretability?** Read [concepts](concepts.md), then follow
  [building a model](building-a-model.md) and [analysing a model](analysing-a-model.md) on your task.
  [After the sweep](after-the-sweep.md) and the [reading list](reading.md) take it from there.
- **Just want to see it run?** Open the [quickstart notebook](../examples/quickstart.ipynb) (or
  [in Colab](https://colab.research.google.com/github/aditeyabaral/MechInterpLens/blob/main/examples/quickstart.ipynb)).
- **Looking something up?** [Outputs](outputs.md) has the saved-file layout;
  [customising](customising.md) has every `TODO` you can fill in.
