"""Entry point for running inference that captures MLP neuron and attention head activations.

Normal mode:   run N prompts, save the answers (and, with --capture-activations, the activations).
Intervention mode (--intervention <lasso_shortlist.json>, or --intervention all for every neuron and head):
    Run the same N prompts without hooks (baseline), then for each important
    neuron/head in the JSON re-run the same N prompts with that feature zeroed,
    recording how far task accuracy drops when it's gone (src/analysis/plot_ablations.py
    can turn those drops into a heatmap). Each ablated model is scored by teacher forcing
    (see inference.score_teacher_forced): one forward pass per batch of prompts instead
    of generating every answer token by token, which also gives the accuracy at each answer step.

================================ ADAPTING THIS TEMPLATE ================================
This file runs out of the box once you implement your task's prompts. Every place you
need to edit is marked with a `TODO` comment -- run `grep -rn TODO src/` to list them.
The extension points, in the order you'll likely touch them:

  1. src/utils/dataset.py  PromptDataset.generate_prompts  -- build the prompts to run.   [required]
  2. src/inference.py      find_answer_span                -- locate the answer token in the generation.
  3. src/inference.py      find_positions_of_interest      -- prompt positions to capture (--capture-activations).
  4. src/utils/parser.py   add_arguments                   -- add any task-specific CLI arguments.
  5. src/utils/dir.py      generate_output_path            -- name your saved output files.
  6. "Intervention mode" below                             -- the format of your --intervention spec file.

To train a small model from scratch first, see src/train/ (tokenizer -> dataset -> model),
which has its own TODO markers.
=======================================================================================
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformer_lens.model_bridge import TransformerBridge

import inference
import utils
from model import load_model
from utils.dataset import PromptDataset


def _default_max_new_tokens(model: TransformerBridge, dataset: PromptDataset) -> int:
    """The longest ground-truth answer in tokens, plus one for <eos>; 16 if no prompt carries an answer.

    Enough for any correct answer: generating longer can't make a wrong one right.
    """
    answers = [entry.get("metadata", {}).get("answer") for entry in dataset.prompts]
    answers = [answer for answer in answers if answer is not None]
    if not answers:
        return 16
    return max(len(ids) for ids in model.tokenizer(answers, add_special_tokens=False)["input_ids"]) + 1


if __name__ == "__main__":
    # 1. Parse CLI arguments (defined in src/utils/parser.py).
    parser = argparse.ArgumentParser(
        description="Run inference capturing MLP neuron and attention head activations.",
    )
    utils.parser.add_arguments(parser)
    args = parser.parse_args()
    print(args)

    # 2. Seed every RNG so a run is reproducible (same seed -> same prompts and same outputs).
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 3. Load the model into a TransformerLens bridge (works for real gpt2 AND a toy model you
    #    trained from scratch -- just point --model-path at a Hub repo or local directory).
    model = load_model(args.model_path, device)

    # 4. Choose which decoder layers to capture from (--layers; None = all of them).
    layers = args.layers if args.layers else list(range(model.cfg.n_layers))

    # 5. Build the prompts to run on, with PromptDataset.generate_prompts (src/utils/dataset.py), which
    #    you implement. If your task needs extra parameters (operator, few-shot count, ...), add them as
    #    CLI args in src/utils/parser.py and forward them here.
    dataset = utils.dataset.PromptDataset.generate_prompts(num_prompts=args.num_prompts)
    # An ablation sweep scores answers, so check for them before the (possibly long) baseline run.
    missing_answer = any(entry.get("metadata", {}).get("answer") is None for entry in dataset.prompts)
    if args.intervention is not None and missing_answer:
        raise ValueError("--intervention needs a ground-truth 'answer' in every prompt's metadata")
    if args.max_new_tokens is None:
        args.max_new_tokens = _default_max_new_tokens(model, dataset)
        print(f"--max-new-tokens: {args.max_new_tokens} (the longest answer plus <eos>)")

    # 6. Baseline run: generate on every prompt and capture activations per --capture-activations (no
    #    ablation here). This runs in BOTH modes -- it's the saved output in normal mode, and the
    #    unablated accuracy reference each ablated run is compared against in intervention mode.
    result = inference.run(
        model,
        dataset,
        layers,
        args.max_new_tokens,
        capture_activations=args.capture_activations,
        capture_logits=args.capture_logits,
    )

    # How many baseline answers are right (src/train/scoring.py).
    # Skipped when the prompts carry no ground-truth "answer" to score against.
    scored = [row["result"]["correct"] for row in result if row["result"]["correct"] is not None]
    if scored:
        print(f"Baseline accuracy: {sum(scored)}/{len(scored)}")

    # 7. Decide where to save (src/utils/dir.py builds a filename from the run's parameters).
    output_path = Path(utils.dir.generate_output_path(args))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Every run parameter, saved with the results so a .pt file always says how it was made.
    run_metadata = {
        "model_path": args.model_path,
        "num_prompts": len(dataset),
        "num_layers": len(layers),
        "layer_indices": layers,
        "max_new_tokens": args.max_new_tokens,
        "capture_activations": args.capture_activations,
        "capture_logits": args.capture_logits,
        "seed": args.seed,
        "d_model": model.cfg.d_model,
        "num_attention_heads": model.cfg.n_heads,
        "head_dim": model.cfg.d_head,
        "num_mlp_neurons": model.cfg.d_mlp,
    }

    # 8a. Normal mode: just save the baseline run plus run metadata. Saved under "baseline"
    #     (the same key intervention mode uses below) since both are the same thing: an unablated
    #     inference.run() pass.
    if args.intervention is None:
        save_data = {"baseline": result, "metadata": run_metadata}

    # 8b. Intervention mode: re-run the prompts many times, each with one component knocked out,
    #     to measure which components causally matter. Results are saved alongside the baseline.
    else:
        # The spec is the lasso_shortlist.json written by src/analysis/lasso.py: analysis["layers"][L][condition]
        # is {"features": [indices], "weight": [coeffs]}. Feature indices below num_mlp_neurons are
        # MLP neurons; the rest are attention heads (offset by num_mlp). We ablate each flagged
        # feature individually (a sweep) and record which conditions flagged it.
        # With --intervention all, the spec is built here instead: every MLP neuron and head of every
        # layer in --layers, under one condition "all". For a small model this exhaustive sweep is
        # affordable, and it can't miss a component that a shortlist (e.g. the lasso's) left out.
        # TODO: if you write your own spec format, adapt this block to build the
        # {layer_idx: [indices]} ablation dicts passed to the scoring below.
        if args.intervention == "all":
            analysis = {
                "num_mlp_neurons": model.cfg.d_mlp,
                "layers": {
                    str(layer): {"all": {"features": list(range(model.cfg.d_mlp + model.cfg.n_heads))}}
                    for layer in layers
                },
            }
        else:
            with open(args.intervention) as f:
                analysis = json.load(f)

        num_mlp = analysis["num_mlp_neurons"]
        ablations: list[dict] = []

        # Baseline accuracy (no ablation): the reference each ablated run is compared against to get
        # its accuracy_drop. Teacher forcing scores every prompt in one batched forward pass per
        # ablation. As a check, it is compared prompt by prompt with the generated baseline, which is
        # scored by the same rule (src/train/scoring.py): the two should agree on every prompt.
        batches, names = inference.build_teacher_forced_batches(model, dataset, args.batch_size)
        baseline_correct = inference.score_teacher_forced(model, batches, len(dataset), len(names))
        baseline_whole = inference.whole_answer_correct(baseline_correct)
        generated = [bool(row["result"]["correct"]) for row in result]
        mismatches = sum(tf != gen for tf, gen in zip(baseline_whole.tolist(), generated))
        print(f"Teacher-forced vs generated baseline: {mismatches} of {len(result)} prompts differ")
        if mismatches:
            print(
                "WARNING: teacher forcing and generation disagree on some baseline prompts. Check that "
                "--max-new-tokens fits every answer and <eos>."
            )
        baseline_accuracy = inference.accuracy(baseline_whole)
        baseline_step_accuracy = inference.step_accuracy(baseline_correct, names)
        run_metadata.update(
            {"step_names": names, "tf_generation_mismatches": mismatches, "batch_size": args.batch_size}
        )

        layers_iter = tqdm(list(analysis["layers"].items()), desc="Layers", position=0)
        for layer_str, layer_conditions in layers_iter:
            layer_idx = int(layer_str)
            # The features to test are those flagged by ANY condition at this layer (their union).
            all_features = sorted({f for entry in layer_conditions.values() for f in entry["features"]})

            for feat_idx in tqdm(all_features, desc=f"Layer {layer_idx} features", position=1, leave=False):
                if feat_idx < num_mlp:
                    mlp_abl = {layer_idx: [feat_idx]}
                    head_abl = None
                    feat_type, local_idx = "mlp", feat_idx
                else:
                    mlp_abl = None
                    head_abl = {layer_idx: [feat_idx - num_mlp]}
                    feat_type, local_idx = "head", feat_idx - num_mlp

                entry = {
                    "layer_idx": layer_idx,
                    "feature_idx": feat_idx,
                    "type": feat_type,
                    "local_idx": local_idx,
                    # which conditions flagged this feature as important
                    "conditions": [
                        name for name, entry in layer_conditions.items() if feat_idx in set(entry["features"])
                    ],
                }
                correct = inference.score_teacher_forced(
                    model, batches, len(dataset), len(names), mlp_ablation=mlp_abl, head_ablation=head_abl
                )
                ablated_accuracy = inference.accuracy(inference.whole_answer_correct(correct))
                step_accuracy = inference.step_accuracy(correct, names)
                step_accuracy_drop = {step: baseline_step_accuracy[step] - acc for step, acc in step_accuracy.items()}
                entry.update(
                    {
                        # Per answer step (metadata["step_names"]).
                        "step_accuracy": step_accuracy,
                        "step_accuracy_drop": step_accuracy_drop,
                        # [prompts, steps] uint8, kept so the drops can be split by any property of the
                        # prompts later (e.g. carry vs no carry), without re-running the sweep.
                        "correct": correct,
                    }
                )
                # Keep only the accuracy stats (and the small per-prompt step matrix), not the heavy
                # ablated rows: the baseline is saved once below, and that drop is all the downstream
                # analysis (e.g. src/analysis/plot_ablations.py) needs.
                entry.update({"accuracy": ablated_accuracy, "accuracy_drop": baseline_accuracy - ablated_accuracy})
                ablations.append(entry)

        run_metadata.update(
            {
                "intervention": args.intervention,
                "baseline_generation_accuracy": inference.accuracy(torch.tensor(generated)),
            }
        )
        save_data = {
            "baseline": result,
            "baseline_accuracy": baseline_accuracy,
            "baseline_step_accuracy": baseline_step_accuracy,
            # [prompts, steps] uint8: 1 right, 0 wrong, inference.NOT_APPLICABLE for missing steps.
            "baseline_correct": baseline_correct,
            "ablations": ablations,
            "metadata": run_metadata,
        }

    torch.save(save_data, output_path)
    print(f"Saved results to {output_path}")
