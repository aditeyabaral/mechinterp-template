"""Train a small LM from scratch on a synthetic dataset, then push it to the HF Hub.

Uses a GPT-2 architecture; only the size parameters (n_embd, n_layer, n_head, ...)
are configurable, so you get a tiny randomly-initialised model.

Loads a pre-built tokenizer from the HF Hub (see train_tokenizer.py) and trains on
a dataset with "prompt" and "answer" columns (see create_dataset.py), then pushes
the trained model to the HF Hub.

Evaluation is done by greedy generation + exact-match accuracy on the "answer"
column. If your task needs finer-grained metrics (per-category accuracy, etc.),
see the commented example inside GenerationEvalTrainer.evaluate.
"""

import argparse
import random

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------------------------------- #
# TODO: fill these in for your task.
# --------------------------------------------------------------------------- #

# HF Hub repo holding your trained tokenizer (see train_tokenizer.py).
# Example: TOKENIZER_NAME = "your-username/your-project-tokenizer"
TOKENIZER_NAME: str = ""

# HF Hub dataset repo with train/validation splits (see create_dataset.py).
# Example: DATASET_NAME = "your-username/your-dataset-name"
DATASET_NAME: str = ""

# Dataset config/subset name, or None if the dataset has a single default config.
DATASET_CONFIG: str | None = None

# --------------------------------------------------------------------------- #


def run_inference_batch(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list[str], max_new_tokens: int = 16
) -> list[str]:
    """Run batched greedy inference; return completions up to the first newline per prompt."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    inputs.pop("token_type_ids", None)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = []
    for out in output_ids:
        completion = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        completions.append(completion.split("\n")[0].strip())
    return completions


class GenerationEvalTrainer(Trainer):
    """Trainer that replaces the default eval loop with generation-based exact-match accuracy.

    The eval dataset is expected to be the raw (untokenized) validation set with
    "prompt" and "answer" columns. For each prompt the model greedily generates a
    completion, which is compared for exact string equality against "answer".
    """

    def __init__(
        self, *args: object, eval_batch_size: int = 128, eval_max_new_tokens: int = 16, **kwargs: object
    ) -> None:
        """Store eval-time generation settings, then defer to the base Trainer."""
        super().__init__(*args, **kwargs)
        self.eval_batch_size = eval_batch_size
        self.eval_max_new_tokens = eval_max_new_tokens

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run greedy generation on the validation set and return exact-match accuracy."""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        was_training = self.model.training
        self.model.eval()
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"

        try:
            results = []
            for i in tqdm(range(0, len(eval_dataset), self.eval_batch_size), desc="Evaluating"):
                batch = eval_dataset[i : i + self.eval_batch_size]
                predictions = run_inference_batch(
                    self.model, self.processing_class, batch["prompt"], self.eval_max_new_tokens
                )
                for j, predicted in enumerate(predictions):
                    results.append({"predicted": predicted, "answer": batch["answer"][j]})

            accuracy = sum(r["predicted"] == r["answer"] for r in results) / len(results)
            metrics = {f"{metric_key_prefix}_accuracy": accuracy}

            # ------------------------------------------------------------------- #
            # Optional: per-category accuracy breakdown.
            # ------------------------------------------------------------------- #
            # `results` is in the same order as `eval_dataset`, so you can split it
            # by any category column on the val rows. This block is drop-in: paste
            # it right here. Example from the operator-overloading study, where rows
            # carry "base_operation"/"target_operation" and a row is "overloaded"
            # when they differ:
            #
            #     base = eval_dataset["base_operation"]
            #     target = eval_dataset["target_operation"]
            #     standard = [r for k, r in enumerate(results) if base[k] == target[k]]
            #     overloaded = [r for k, r in enumerate(results) if base[k] != target[k]]
            #     metrics[f"{metric_key_prefix}_standard_accuracy"] = (
            #         sum(r["predicted"] == r["answer"] for r in standard) / len(standard) if standard else float("nan")
            #     )
            #     metrics[f"{metric_key_prefix}_overloaded_accuracy"] = (
            #         sum(r["predicted"] == r["answer"] for r in overloaded) / len(overloaded)
            #         if overloaded else float("nan")
            #     )
            # ------------------------------------------------------------------- #

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        finally:
            self.processing_class.padding_side = original_padding_side
            if was_training:
                self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small LM from scratch on a synthetic dataset")

    # Model size (GPT-2 config attribute names)
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size (GPT-2 n_embd)")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Number of transformer layers (GPT-2 n_layer)")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads (GPT-2 n_head)")
    parser.add_argument("--intermediate-size", type=int, default=1024, help="MLP/FFN inner size (GPT-2 n_inner)")
    parser.add_argument(
        "--max-position-embeddings", type=int, default=256, help="Max sequence length (GPT-2 n_positions)"
    )

    # Hub + Training
    parser.add_argument(
        "--hub-name",
        type=str,
        required=True,
        help="HF Hub repo to push the model to",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./saved_models", help="Output directory for model checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per device for train and eval")
    parser.add_argument("--eval-max-new-tokens", type=int, default=16, help="Max tokens to generate per eval prompt")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of steps used for LR warmup")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="AdamW weight decay. Defaults to a strong 1.0: heavy weight decay is a well-known "
        "trigger for 'grokking' (the model first memorises the training data, then later "
        "suddenly generalises), which is a phenomenon worth being able to study in small models.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints to keep on disk")
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type (e.g. 'linear', 'cosine', 'cosine_with_restarts')",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        help="Reporting integration (e.g. 'wandb', 'tensorboard', 'none')",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for the experiment tracker")

    args = parser.parse_args()

    if not TOKENIZER_NAME or not DATASET_NAME:
        raise ValueError(
            "TOKENIZER_NAME and DATASET_NAME are not set. Fill these in at the top of this script for your task."
        )

    hub_name = args.hub_name

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    print(f"Loading tokenizer from '{TOKENIZER_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Tokenizer vocab size : {len(tokenizer)}")
    print(f"  max token ID       : {max(tokenizer.get_vocab().values())}")
    print(f"  pad_token_id       : {tokenizer.pad_token_id}")
    print(f"  bos_token_id       : {tokenizer.bos_token_id}")
    print(f"  eos_token_id       : {tokenizer.eos_token_id}")

    # Config -- a GPT-2 architecture sized down to the values below. Tune these (via CLI args)
    # to control how big your model is: more layers/heads/width = more capacity but slower.
    config = GPT2Config(
        vocab_size=len(tokenizer),  # must match the tokenizer so the embedding/output sizes line up
        n_embd=args.hidden_size,  # width of the residual stream (hidden_size)
        n_inner=args.intermediate_size,  # width of each MLP's hidden layer
        n_layer=args.num_hidden_layers,  # number of transformer blocks (depth)
        n_head=args.num_attention_heads,  # attention heads per block
        n_positions=args.max_position_embeddings,  # max sequence length the model can handle
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # from_config builds the model with RANDOM weights (training "from scratch"), as opposed to
    # from_pretrained which would download trained weights. We want a fresh model to train ourselves.
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nDevice: {device}")
    print(f"Seed:   {args.seed}\n")

    print("Model Configuration:")
    print(f"  vocab_size:    {config.vocab_size}")
    print(f"  n_embd:        {config.n_embd}")
    print(f"  n_inner:       {config.n_inner}")
    print(f"  n_layer:       {config.n_layer}")
    print(f"  n_head:        {config.n_head}")
    print(f"  n_positions:   {config.n_positions}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Model size (fp32): ~{total_params * 4 / 1e9:.2f} GB")

    # Dataset
    print(f"\nLoading dataset '{DATASET_NAME}' (config: {DATASET_CONFIG})...")
    raw_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    raw_val = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")

    def tokenize(batch: dict) -> dict:
        """Turn each (prompt, answer) pair into one token sequence the model learns to predict.

        We train on the full string "prompt + answer + EOS". The model learns next-token
        prediction over the whole thing, so it learns both to continue prompts and to stop (EOS).
        """
        texts = [p + a + tokenizer.eos_token for p, a in zip(batch["prompt"], batch["answer"])]
        return tokenizer(texts, truncation=True, max_length=args.max_position_embeddings)

    tokenized_train = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)

    # The collator pads each batch to equal length and (mlm=False -> causal LM) creates the
    # training labels as the input shifted by one, i.e. "predict the next token at every position".
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    cuda_available = torch.cuda.is_available()
    supports_bf16 = cuda_available and torch.cuda.get_device_capability()[0] >= 8
    print(f"\nCUDA available: {cuda_available}\nSupports bf16: {supports_bf16}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        eval_on_start=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        push_to_hub=True,
        hub_model_id=hub_name,
        hub_strategy="every_save",
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name if args.run_name is not None else hub_name.split("/")[-1],
        bf16=supports_bf16,
    )

    trainer = GenerationEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=raw_val,
        data_collator=collator,
        processing_class=tokenizer,
        eval_batch_size=args.batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nPushing model and tokenizer to {hub_name}...")
    trainer.push_to_hub()

    # Post-training evaluation
    print("\nRunning post-training evaluation on validation split...")
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    if wandb.run is not None:
        wandb.run.summary["val/accuracy"] = metrics["eval_accuracy"]
        wandb.run.summary["val/n_examples"] = len(raw_val)
