"""Entry point for running inference that captures MLP neuron and attention head activations.

Normal mode:   run N prompts, save activations.
Intervention mode (--intervention <analysis.json>):
    Run the same N prompts without hooks (baseline), then for each important
    neuron/head in the JSON re-run the same N prompts with that feature zeroed,
    recording how far task accuracy drops when it's gone (src/plot_ablations.py
    can turn those drops into a heatmap).

================================ ADAPTING THIS TEMPLATE ================================
This file runs out of the box once you implement your task's prompts. Every place you
need to edit is marked with a `TODO` comment -- run `grep -rn TODO src/` to list them.
The extension points, in the order you'll likely touch them:

  1. src/utils/dataset.py  PromptDataset.generate_prompts  -- build the prompts to run.   [required]
  2. src/inference.py      find_answer_span                -- locate the answer token in the generation.
  3. src/inference.py      find_positions_of_interest      -- prompt positions to capture (--capture-geometry).
  4. src/utils/parser.py   add_arguments                   -- add any task-specific CLI arguments.
  5. src/utils/dir.py      generate_output_path            -- name your saved output files.
  6. "Intervention mode" below                             -- the format of your --intervention spec file.
  7. src/main.py           is_correct                      -- how an answer is scored right (ablation accuracy).

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

import inference
import utils
from model import load_model


def is_correct(row: dict) -> bool:
    """Whether the model's generated answer matches the ground truth for one result row.

    Used to score intervention runs (baseline vs ablated accuracy). The default compares the answer
    token inference.py captured (row["result"]["answer"]["token"]) against an "answer" stored in
    that prompt's metadata (see PromptDataset.generate_prompts in src/utils/dataset.py). It
    therefore only does something useful if your prompts carry a ground-truth "answer" in their
    metadata; otherwise every row counts as wrong and the accuracies come out 0.

    TODO (optional): adjust this if "correct" means something else for your task, or if your
    metadata stores the truth under a different key.
    """
    return row["result"]["answer"]["token"] == row["metadata"].get("answer")


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

    # 5. Build the prompts to run on. TODO: implement PromptDataset.generate_prompts (src/utils/dataset.py).
    # If your task needs extra parameters (operator, few-shot count, ...), add them as CLI args in
    # src/utils/parser.py and forward them here.
    dataset = utils.dataset.PromptDataset.generate_prompts(num_prompts=args.num_prompts)

    # 6. Baseline run: generate on every prompt and capture activations (no ablation here).
    result = inference.run(
        model,
        dataset,
        layers,
        args.max_new_tokens,
        capture_geometry=args.capture_geometry,
    )

    # 7. Decide where to save (src/utils/dir.py builds a filename from the run's parameters).
    output_path = Path(utils.dir.generate_output_path(args))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 8a. Normal mode: just save the baseline activations plus run metadata. Saved under "baseline"
    #     (the same key intervention mode uses below) since both are the same thing: an unablated
    #     inference.run() pass.
    if args.intervention is None:
        save_data = {
            "baseline": result,
            "metadata": {
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "num_layers": len(layers),
                "layer_indices": layers,
                "max_new_tokens": args.max_new_tokens,
                "num_attention_heads": model.cfg.n_heads,
                "head_dim": model.cfg.d_head,
            },
        }

    # 8b. Intervention mode: re-run the prompts many times, each with one component knocked out,
    #     to measure which components causally matter. Results are saved alongside the baseline.
    else:
        # The spec is the analysis.json written by src/lasso.py: analysis["layers"][L][condition]
        # is {"features": [indices], "weight": [coeffs]}. Feature indices below num_mlp_neurons are
        # MLP neurons; the rest are attention heads (offset by num_mlp). We ablate each flagged
        # feature individually (a sweep) and record which conditions flagged it.
        # TODO: if you write your own spec format, adapt this block to build the
        # {layer_idx: [indices]} ablation dicts passed to inference.run.
        with open(args.intervention) as f:
            analysis = json.load(f)

        num_mlp = analysis["num_mlp_neurons"]
        ablations: list[dict] = []

        # Baseline accuracy (no ablation): the reference each ablated run is compared against to get
        # its accuracy_drop. See `is_correct` above for how "correct" is decided.
        baseline_accuracy = sum(1 for row in result if is_correct(row)) / len(result) if result else 0.0

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

                ablated = inference.run(
                    model,
                    dataset,
                    layers,
                    args.max_new_tokens,
                    mlp_ablation=mlp_abl,
                    head_ablation=head_abl,
                    capture_geometry=args.capture_geometry,
                )
                ablated_accuracy = sum(1 for row in ablated if is_correct(row)) / len(ablated) if ablated else 0.0
                # Keep only the scalar accuracy stats, not the heavy `ablated` rows: the baseline is
                # saved once below, and that drop is all the downstream analysis (e.g.
                # src/plot_ablations.py) needs.
                ablations.append(
                    {
                        "layer_idx": layer_idx,
                        "feature_idx": feat_idx,
                        "type": feat_type,
                        "local_idx": local_idx,
                        # which conditions flagged this feature as important
                        "conditions": [
                            name for name, entry in layer_conditions.items() if feat_idx in set(entry["features"])
                        ],
                        "accuracy": ablated_accuracy,
                        "accuracy_drop": baseline_accuracy - ablated_accuracy,
                    }
                )

        save_data = {
            "baseline": result,
            "baseline_accuracy": baseline_accuracy,
            "ablations": ablations,
            "metadata": {
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "num_layers": len(layers),
                "layer_indices": layers,
                "max_new_tokens": args.max_new_tokens,
                "num_attention_heads": model.cfg.n_heads,
                "head_dim": model.cfg.d_head,
                "intervention": args.intervention,
            },
        }

    torch.save(save_data, output_path)
    print(f"Saved results to {output_path}")
