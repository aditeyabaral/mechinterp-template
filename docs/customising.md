# Customising the template

What you fill in for your task, practical tips, and how to use a model you didn't train.

## Everything you need to fill in

Every spot you might edit is marked with a `TODO`. List them all at any time:

```bash
grep -rn TODO src/
```

| File | What to implement | When |
| --- | --- | --- |
| `src/train/train_tokenizer.py` | `VOCAB_CHARS`, `SAMPLE_TEXTS` | building the model |
| `src/train/create_dataset.py` | `build_dataset`: your training data; `item_key` for the leakage check (optional) | building the model |
| `src/train/train_model.py` | `TOKENIZER_NAME`, `DATASET_NAME`; `eval_categories` for per-category accuracy (optional) | building the model |
| `src/utils/dataset.py` | `PromptDataset.generate_prompts`: the probe prompts | **required to analyse** |
| `src/inference.py` | `find_answer_span`: which substring is "the answer" | optional (good default) |
| `src/inference.py` | `find_positions_of_interest`: extra prompt positions to record (gets each prompt's metadata) | optional (default: none) |
| `src/utils/parser.py` | extra task-specific command-line arguments | optional |
| `src/utils/dir.py` | `generate_output_path`: the output filename | optional (good default) |
| `src/analysis/lasso.py` | `keep_row`, `assign_condition`, `build_target`: which rows, what to compare/predict | optional (good defaults) |

Each `TODO` explains *what* to do, *why*, and shows a worked example in comments. Once your model is
trained, the analysis side **runs as soon as you implement `PromptDataset.generate_prompts`**; every
other extension point has a working default.

To see the whole pipeline run before writing anything, fill everything in with the template's own
example task (single-digit addition): `uv run python scripts/fill_example.py .` uncomments every
example in place, so run it **in a copy** of the repository, not the one you'll work in.

## Tips and common pitfalls

- **Always pass `--capture-activations` for analysis.** Without it, *no* activations are recorded; you
  only get the generated answers and the accuracy, which is all an ablation sweep needs.
- **Keep answers short** (one or two tokens). Long, variable answers make "the answer position" fuzzy
  and the analysis noisier.
- **Start with a tiny model.** Fewer layers and heads means fewer things to understand.
- **Set `--seed` for repeatable runs.** `main.py` seeds prompt sampling and PyTorch. Numerical
  results can vary across devices.
- **Prompts are tokenized exactly as in training.** TransformerLens normally puts a BOS token in front
  of every prompt, even for models that never saw one, which shifts every position and can change the
  answers. `load_model` (`src/model.py`) instead adds a BOS only if your tokenizer does. So when you
  count token positions (e.g. in `find_positions_of_interest`), count them as
  `model.to_str_tokens(prompt)` shows them.
- **Lint like CI does.** `uv sync` also installs the pinned dev tools (`ruff`, `pre-commit`), so your
  editor, the pre-commit hook and CI all run the same `ruff` version. `uv run pre-commit install`
  runs them on every commit.

## Analysing a pretrained model

You don't *have* to train your own model. To skip steps 1–3 and study an existing GPT-2-style
checkpoint instead, point `--model-path` at any HF repo id or local path:

```bash
uv run python src/main.py -m gpt2 --num-prompts 200 --capture-activations
```

Capturing activations (steps 4–5) works the same. Anything scored for correctness does not: an
answer only counts if the model ends it with `<eos>`, and GPT-2 was never trained to (in a few-shot
prompt it ends answers with a newline, copying the examples). So every GPT-2 answer is scored wrong:
the lasso's default `keep_row` keeps no rows, and the ablation sweep (step 7) has no accuracy to
drop. For steps 6–7, train your own model, which is recommended for learning anyway, because you
control and understand the task completely.

## Using a non-GPT-2 model

There is **no architecture-specific code to edit**. The model is loaded through
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)'s `TransformerBridge`
(`src/model.py`), which wraps any supported decoder-only HuggingFace model and re-exposes it with
*uniform* hook names (`blocks.{i}.hook_resid_post`, `blocks.{i}.mlp.hook_post`,
`blocks.{i}.attn.hook_z`, `hook_embed`). The capture loop and ablation code in `src/inference.py`
reference only those names, so switching models is just a matter of pointing `--model-path` at a
different Hub repo or local directory; the bridge reads the new model's config and `cfg.n_layers /
n_heads / d_head` update automatically. Models without a learned positional embedding (e.g. Llama,
whose rotary embedding acts inside attention) simply store `pos_embedding` as `None`. Only training
(`src/train/train_model.py`) is GPT-2-specific: it builds a GPT-2 from scratch.
