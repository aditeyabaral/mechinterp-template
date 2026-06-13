"""Entry point for running inference that captures MLP neuron and attention head activations.

Normal mode:   run N prompts, save activations.
Intervention mode (--intervention <analysis.json>):
    Run the same N prompts without hooks (baseline), then for each important
    neuron/head in the JSON run the same N prompts with that feature zeroed.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

import inference
import utils
from model import LargeLanguageModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference capturing MLP neuron and attention head activations.",
    )
    utils.parser.add_arguments(parser)
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bnb_config = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        if args.quantize
        else None
    )
    llm = LargeLanguageModel(
        model_path=args.model_path,
        bnb_config=bnb_config,
        device=device,
    )

    layers = llm.get_layers(args.layers)

    dataset = utils.dataset.PromptDataset.generate_prompts(
        num_prompts=args.num_prompts,
        few_shot_examples=args.few_shot_examples,
        overloading=True,
        operator=args.operator,
        overloading_operator=args.overloading_operator,
        max_digits=args.max_digits,
        reverse=args.reverse,
        pad_zero=args.pad_zero,
    )

    # Baseline run
    result = inference.run(
        llm,
        dataset,
        layers,
        args.max_new_tokens,
        operator=args.operator,
        capture_geometry=args.capture_geometry,
    )

    # Output path
    model_name = args.model_path.replace("/", "--")
    op = utils.dataset.PromptDataset.OPERATOR_SYMBOL_TO_NAME_MAP.get(args.operator, args.operator)
    oop = utils.dataset.PromptDataset.OPERATOR_SYMBOL_TO_NAME_MAP.get(
        args.overloading_operator, args.overloading_operator
    )
    intervention = args.intervention is not None
    fname = (
        f"[m={model_name}]_[p={args.num_prompts}]_[fs={args.few_shot_examples}]"
        f"_[rv={args.reverse}]_[pz={args.pad_zero}]"
        f"_[op={op}]_[oop={oop}]_[int={intervention}].pt"
    )
    if args.output is not None:
        output_path = Path(args.output) / fname
    else:
        output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normal mode
    if args.intervention is None:
        save_data = {
            "result": result,
            "metadata": {
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "few_shot_examples": args.few_shot_examples,
                "num_layers": len(layers),
                "layer_indices": list(layers.keys()),
                "operator": args.operator,
                "overloading_operator": args.overloading_operator,
                "max_new_tokens": args.max_new_tokens,
                "max_digits": args.max_digits,
                "reverse": args.reverse,
                "pad_zero": args.pad_zero,
                "num_attention_heads": llm.num_attention_heads,
                "head_dim": llm.head_dim,
            },
        }

    # Intervention mode
    else:
        with open(args.intervention) as f:
            analysis = json.load(f)

        num_mlp = analysis["num_mlp_neurons"]
        ablations: list[dict] = []

        layers_iter = tqdm(list(analysis["layers"].items()), desc="Layers", position=0)
        for layer_str, layer_data in layers_iter:
            layer_idx = int(layer_str)
            lasso = layer_data.get("lasso", {})
            std_features = set(lasso.get("std", []))
            over_features = set(lasso.get("over", []))
            all_features = sorted(std_features | over_features)

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
                    llm,
                    dataset,
                    layers,
                    args.max_new_tokens,
                    operator=args.operator,
                    mlp_ablation=mlp_abl,
                    head_ablation=head_abl,
                    capture_geometry=args.capture_geometry,
                )
                ablations.append(
                    {
                        "layer_idx": layer_idx,
                        "feature_idx": feat_idx,
                        "type": feat_type,
                        "local_idx": local_idx,
                        "in_std": feat_idx in std_features,
                        "in_over": feat_idx in over_features,
                        "result": ablated,
                    }
                )

        save_data = {
            "baseline": result,
            "ablations": ablations,
            "metadata": {
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "few_shot_examples": args.few_shot_examples,
                "num_layers": len(layers),
                "layer_indices": list(layers.keys()),
                "operator": args.operator,
                "overloading_operator": args.overloading_operator,
                "max_new_tokens": args.max_new_tokens,
                "max_digits": args.max_digits,
                "reverse": args.reverse,
                "pad_zero": args.pad_zero,
                "num_attention_heads": llm.num_attention_heads,
                "head_dim": llm.head_dim,
                "intervention": args.intervention,
            },
        }

    torch.save(save_data, output_path)
    print(f"Saved results to {output_path}")
