# Building a model (steps 1–3)

Steps 1–3 build your toy model: a tokenizer, a dataset, and a small GPT-2 trained on it. Then
[analysing a model](analysing-a-model.md) (steps 4–7) studies it. If you'd rather study an existing
model, see [analysing a pretrained model](customising.md#analysing-a-pretrained-model).

> [!NOTE]
> **Local by default.** `--hub-name` is optional for `train_tokenizer.py`, `create_dataset.py`, and
> `train_model.py`: the output is always saved locally (`./artifacts/tokenizer`,
> `./artifacts/dataset`, `--output-dir` for the model, default `./saved_models`) with no Hugging
> Face login needed. Pass `--hub-name <user>/...` to also push it to the Hub (log in once with
> `uv run hf auth login`), and point `--model-path` (step 5) at whichever location you want to use.

## 1. Build a tokenizer

A **tokenizer** maps text to integer ids. For small synthetic tasks the simplest, most transparent
choice is one token per character. Set `VOCAB_CHARS` (every character your task can produce) and
`SAMPLE_TEXTS` in `src/train/train_tokenizer.py`, then:

```bash
uv run python src/train/train_tokenizer.py --hub-name <user>/my-tokenizer
```

The script saves the tokenizer, loads it back exactly as training will, and checks it: every
character is one token, text decodes back exactly, ids are unique and contiguous, and so on. It is
pushed to the Hub only if every check passes. Add `--add-bos` to put a `<bos>` token at the start of
every sequence (a fixed first position many models use as a "resting place" for attention); the
rest of the pipeline follows the tokenizer, so training and analysis always agree on it.

## 2. Generate a dataset

Implement `build_dataset` in `src/train/create_dataset.py` to produce a Hugging Face dataset with
`prompt` and `answer` columns and two splits: `train` (learned from) and `validation` (scored during
training to pick the best checkpoint, and later a source of held-out prompts for the analysis). The
file documents a *leakage-safe* recipe, and before anything is saved, `check_dataset` verifies that
no question appears in two splits (fill in `item_key` to say what "the same question" means for your
task, e.g. the same two operands).

```bash
uv run python src/train/create_dataset.py --hub-name <user>/my-dataset
```

The dataset is always saved to `./artifacts/dataset`; `--hub-name` also pushes it. For big datasets,
build the splits with `Dataset.from_generator` as the example does: the rows go to disk as they are
generated instead of filling your memory.

## 3. Train the toy model

Set `TOKENIZER_NAME` and `DATASET_NAME` at the top of `src/train/train_model.py`, then train. You
control the size with flags like `--num-hidden-layers`, `--num-attention-heads`, `--hidden-size`:

```bash
uv run python src/train/train_model.py --hub-name <user>/my-model --num-epochs 5
```

What happens during training:

- **The data.** Each row becomes one sequence, `prompt + answer + <eos>`. The loss counts only the
  answer and `<eos>`, so the model learns to answer and to *stop*, and nothing else: the prompt's
  tokens (random operands, mostly) would only add noise and side behaviours to the circuits you'll
  study. The model's `config.json` records what it was trained on (the dataset, its config and the
  tokenizer) and how (`training_args`: every option you passed), so anyone can reproduce it.
- **Evaluation.** Every `--eval-steps` the model generates an answer for every row of the
  validation split. An answer is right only if its tokens are exactly the answer's, followed by
  `<eos>`. Fill in `eval_categories` for a per-category breakdown (e.g. carry vs no carry).
- **The end.** The best checkpoint (by validation accuracy) is kept and saved. The final numbers
  (`eval_results.json`) and the training curves (`trainer_state.json`) are saved with the model.
- **Interrupted runs.** With `--hub-name`, the latest full checkpoint (weights *and* optimizer
  state) is pushed to the repo's `last-checkpoint/` folder, so a run that dies (e.g. a lost Colab
  machine) continues on any machine with `--resume-from-checkpoint hub` (or pass a local
  `checkpoint-*` directory). The repo's root holds the *best* model, which is what
  `--model-path <user>/my-model` loads. `--init-from DIR` starts from a model's weights with a fresh
  optimizer instead, and `--max-steps N` stops after N steps.

> [!NOTE]
> **A note on grokking.** Training defaults to a strong weight decay (`--weight-decay 1.0`). Heavy
> weight decay is a well-known trigger for **grokking**: memorise first, generalise suddenly later.
> Watching a circuit form during grokking is a classic experiment, so it's on by default. Train for
> plenty of steps to give it a chance to appear, or lower `--weight-decay` to turn it down.

> [!TIP]
> **Smaller is better for interpretability.** A model with a couple of layers and a handful of heads
> is *far* easier to fully understand than full GPT-2. Start as small as the task allows.

`python src/train/train_model.py -h` lists every training option.
