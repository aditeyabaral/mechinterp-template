"""Train a small LM from scratch on a synthetic dataset (locally by default, or push to the HF Hub).

Uses a GPT-2 architecture; the size parameters (n_embd, n_layer, n_head, n_positions) are
configurable via CLI args, so you get a tiny randomly-initialised model. The MLP inner size
(n_inner) is always 4x hidden size (standard GPT-2 convention).

Loads a pre-built tokenizer (see train_tokenizer.py; TOKENIZER_NAME below can be a
local path such as "./artifacts/tokenizer" or an HF Hub repo ID) and trains on a
dataset with "prompt" and "answer" columns (see create_dataset.py; DATASET_NAME can likewise
be a local path such as "./artifacts/dataset" or an HF Hub repo ID). The dataset, its config and
the tokenizer are saved in the model's config.json, so anyone using the model knows what it was
trained on, together with how it was trained ("training_args": every option of this script).

Evaluation is done by greedy generation on the whole validation split. An answer is correct only if the
generated tokens are exactly the answer's tokens followed by <eos>, the stop the model is
trained to produce. For a per-category breakdown (e.g. carry vs no carry), fill in
eval_categories. The selected (best) model is the one with the highest validation accuracy.

Long runs can be resumed, even on another machine: with --hub-name, the latest full checkpoint
(weights, optimizer, LR schedule, RNG state) is pushed to the repo's "last-checkpoint/" folder,
and --resume-from-checkpoint hub continues from it. The repo's root holds the BEST checkpoint
(by validation accuracy), which is what from_pretrained(repo) and src/main.py load.
"""

import argparse
import os
import random

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GPT2Config,
    Trainer,
    TrainingArguments,
)

# (src/train/scoring.py): the rule for a correct answer, shared with inference and analysis.
from scoring import is_correct

# --------------------------------------------------------------------------- #
# TODO: fill these in for your task.
# --------------------------------------------------------------------------- #

# HF Hub repo holding your trained tokenizer (see train_tokenizer.py).
# Example: TOKENIZER_NAME = "your-username/your-project-tokenizer"
TOKENIZER_NAME: str = ""

# Dataset with "train"/"validation" splits (see create_dataset.py). A
# local path such as "./artifacts/dataset" or an HF Hub repo ID both work.
# Example: DATASET_NAME = "your-username/your-dataset-name"
DATASET_NAME: str = ""

# Dataset config/subset name, or None if the dataset has a single default config.
DATASET_CONFIG: str | None = None

# --------------------------------------------------------------------------- #


def _run_inference_batch(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: list[str], max_new_tokens: int = 16
) -> list[list[int]]:
    """Run batched greedy inference; return the generated token ids for each prompt.

    The tokenizer must pad on the LEFT, so that every prompt ends right before its generated tokens.
    """
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
    return output_ids[:, prompt_len:].tolist()


def eval_categories(row: dict) -> list[str]:
    """Return the categories an eval row belongs to, for a per-category accuracy breakdown.

    The overall accuracy is always reported as {prefix}_accuracy (e.g. "eval_accuracy"); every
    category returned here also gets its own {prefix}_{category}_accuracy (e.g.
    "eval_carry_accuracy"). The default returns no categories.

    Args:
        row: One eval row as a dict of its columns (at least "prompt" and "answer", plus any extra
            columns your create_dataset.py writes).
    """
    # ----------------------------------------------------------------------- #
    # TODO (optional): return the categories this row belongs to.
    # ----------------------------------------------------------------------- #
    #
    # Example (a single-digit addition task), reporting accuracy separately for sums that carry
    # (answer >= 10) and those that don't:
    #
    #     return ["carry" if int(row["answer"]) >= 10 else "no_carry"]
    #
    # ----------------------------------------------------------------------- #
    return []


class GenerationEvalTrainer(Trainer):
    """Trainer that replaces the default eval loop with generation-based exact-match accuracy.

    The eval dataset is expected to be raw (untokenized) rows with "prompt" and "answer"
    columns. For each prompt the model greedily generates a completion, which is checked by
    is_correct from src/train/scoring.py (the answer's exact tokens, then <eos>).
    """

    def __init__(
        self,
        *args: object,
        eval_batch_size: int = 128,
        eval_max_new_tokens: int = 16,
        **kwargs: object,
    ) -> None:
        """Store eval-time settings, then defer to the base Trainer."""
        super().__init__(*args, **kwargs)
        self.eval_batch_size = eval_batch_size
        self.eval_max_new_tokens = eval_max_new_tokens

    def _accuracy_metrics(self, dataset: Dataset, prefix: str) -> dict[str, float]:
        """Return {prefix}_accuracy over every row, plus {prefix}_{category}_accuracy per category."""
        tokenizer = self.processing_class
        # Running [correct, total] counts per category ("" = every row) rather than one result per
        # row: a large eval split would otherwise fill the memory with millions of small dicts.
        counts: dict[str, list[int]] = {"": [0, 0]}
        for i in tqdm(range(0, len(dataset), self.eval_batch_size), desc=f"Evaluating {prefix}"):
            batch = dataset[i : i + self.eval_batch_size]
            predictions = _run_inference_batch(self.model, tokenizer, batch["prompt"], self.eval_max_new_tokens)
            # The answers tokenized as in training (see tokenize below): no special tokens.
            answer_ids = tokenizer(batch["answer"], add_special_tokens=False)["input_ids"]
            for j, predicted in enumerate(predictions):
                right = is_correct(predicted, answer_ids[j], tokenizer.eos_token_id)
                row = {column: values[j] for column, values in batch.items()}
                for category in ["", *eval_categories(row)]:
                    count = counts.setdefault(category, [0, 0])
                    count[0] += right
                    count[1] += 1

        return {
            f"{prefix}_{category + '_' if category else ''}accuracy": correct / total if total else float("nan")
            for category, (correct, total) in sorted(counts.items())
        }

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run greedy generation and return exact-match accuracy.

        Scores eval_dataset if given, otherwise the default eval dataset. ignore_keys is unused, and
        is here only to match Trainer.evaluate, which calls it by name.
        """
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        was_training = self.model.training
        self.model.eval()
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"

        try:
            metrics = self._accuracy_metrics(eval_dataset, metric_key_prefix)

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        finally:
            self.processing_class.padding_side = original_padding_side
            if was_training:
                self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small LM from scratch on a synthetic dataset")

    # Model size (GPT-2 config attribute names). The MLP/FFN inner size (n_inner) is not an
    # independent knob -- it's always 4x hidden size, the standard GPT-2 convention.
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size (GPT-2 n_embd)")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Number of transformer layers (GPT-2 n_layer)")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads (GPT-2 n_head)")
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=256,
        help="Max sequence length (GPT-2 n_positions). Sequences are never truncated, so it must fit the longest one.",
    )

    # Hub + Training
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub repo to push the trained model to. If omitted, the model is only saved "
        "locally to --output-dir (no Hugging Face login needed).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./saved_models", help="Output directory for model checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N steps; overrides --num-epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per device for train and eval")
    parser.add_argument("--eval-max-new-tokens", type=int, default=16, help="Max tokens to generate per eval prompt")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.05, help="Fraction of steps used for LR warmup (below 1)"
    )
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
        default="none",
        help="Reporting integration (e.g. 'wandb', 'tensorboard', 'none'). Defaults to 'none' so "
        "training works without a wandb account; pass --report-to wandb to enable it.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for the experiment tracker")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume an interrupted run (weights, optimizer, LR schedule, RNG and data position) from a "
        "checkpoint directory (e.g. ./saved_models/checkpoint-500), or pass 'hub' to download the latest "
        "checkpoint pushed to --hub-name",
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Start from the weights in this checkpoint directory or Hub repo, with a fresh optimizer and LR "
        "schedule (unlike --resume-from-checkpoint), e.g. to keep training a model on new data. The "
        "architecture comes from the checkpoint, so the model-size options are ignored.",
    )

    args = parser.parse_args()

    if not TOKENIZER_NAME or not DATASET_NAME:
        raise ValueError(
            "TOKENIZER_NAME and DATASET_NAME are not set. Fill these in at the top of this script for your task."
        )
    if args.init_from is not None and args.resume_from_checkpoint is not None:
        parser.error("--init-from and --resume-from-checkpoint cannot be combined: resuming restores its own weights")

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
        n_inner=4 * args.hidden_size,  # width of each MLP's hidden layer -- standard GPT-2 4x ratio
        n_layer=args.num_hidden_layers,  # number of transformer blocks (depth)
        n_head=args.num_attention_heads,  # attention heads per block
        n_positions=args.max_position_embeddings,  # max sequence length the model can handle
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Disable dropout (GPT2Config defaults all of these to 0.1). For a tiny model trained from
        # scratch on a synthetic task, weight decay is usually a better sole regularizer; dropout
        # mostly adds noise that hurts these small setups. Remove these lines if you
        # want the standard GPT-2 dropout back.
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        summary_first_dropout=0.0,
    )
    # Saved in config.json, so anyone using the model knows what it was trained on. A resumed run must match it.
    provenance = {
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "tokenizer": TOKENIZER_NAME,
    }
    # How the model was trained, also saved in config.json, so a model on the Hub says how to reproduce it:
    # every option of this script. Only recorded, never checked, so a resumed run may change e.g. --max-steps.
    provenance["training_args"] = vars(args)
    config.update(provenance)

    # from_config builds the model with RANDOM weights (training "from scratch"), as opposed to
    # from_pretrained which would download trained weights. We want a fresh model to train ourselves.
    model = AutoModelForCausalLM.from_config(config)
    if args.init_from is not None:
        model = AutoModelForCausalLM.from_pretrained(args.init_from)
        config = model.config
        config.update(provenance)
        print(f"\nInitialized weights from {args.init_from}")
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
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)
    raw_train, raw_val = raw["train"], raw["validation"]

    def tokenize(batch: dict) -> dict:
        """Turn each (prompt, answer) pair into one token sequence the model learns to predict.

        We train on the full sequence "prompt + answer + EOS". The model learns next-token
        prediction over the whole thing, so it learns both to continue prompts and to stop (EOS).

        The prompt and the answer are tokenized SEPARATELY and then joined, so the prompt's tokens
        are exactly those the model sees at evaluation and inference time, where the prompt is
        tokenized on its own (with a subword tokenizer, tokenizing "prompt + answer" as one string
        can merge characters across the boundary). The prompt gets the tokenizer's special tokens
        (e.g. a BOS, if it adds one); the answer gets none.

        The labels (what the loss compares each prediction with) are the sequence itself; the model
        shifts them by one internally, so each position learns to predict the NEXT token. The
        prompt's labels are -100, which the loss ignores, so the model only learns to produce the
        answer and then stop: the prompt's tokens (random operands, mostly) would only add noise
        and side behaviours to the circuits you'll study.
        """
        prompts = tokenizer(batch["prompt"])["input_ids"]
        answers = tokenizer(batch["answer"], add_special_tokens=False)["input_ids"]
        input_ids = [prompt + answer + [tokenizer.eos_token_id] for prompt, answer in zip(prompts, answers)]
        # Never truncate silently: a cut-off sequence loses its answer or <eos>, and the model would
        # quietly learn something else. (n_positions is --max-position-embeddings, or the --init-from model's.)
        longest = max(len(ids) for ids in input_ids)
        if longest > model.config.n_positions:
            raise ValueError(
                f"A training sequence has {longest} tokens, more than the model's {model.config.n_positions} "
                "positions. Raise --max-position-embeddings to fit the longest sequence."
            )
        labels = [[-100] * len(prompt) + ids[len(prompt) :] for prompt, ids in zip(prompts, input_ids)]
        return {"input_ids": input_ids, "labels": labels}

    # Tokenized lazily, batch by batch as training reads it (with_transform), rather than all up front
    # with .map: on a dataset of millions of rows, .map would take long and store a second copy.
    tokenized_train = raw_train.select_columns(["prompt", "answer"]).with_transform(tokenize)

    # The collator pads each batch to equal length: input_ids with the pad token, labels with -100
    # so padding is ignored by the loss.
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    cuda_available = torch.cuda.is_available()
    supports_bf16 = cuda_available and torch.cuda.get_device_capability()[0] >= 8
    print(f"\nCUDA available: {cuda_available}\nSupports bf16: {supports_bf16}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        # The tokenize transform needs the raw "prompt" and "answer" columns.
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        # A float below 1 is a fraction of the total steps. (TrainingArguments' own warmup_ratio is
        # deprecated in transformers 5, and warns that it will be removed.)
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        eval_on_start=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Reloading the best checkpoint may print "missing keys: ['lm_head.weight']" -- this is
        # expected and harmless: GPT-2 ties its output head to the input embedding, so that
        # weight is shared rather than saved twice.
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        push_to_hub=hub_name is not None,
        hub_model_id=hub_name,
        # Push the best model to the repo root and the latest FULL checkpoint (with optimizer state)
        # to "last-checkpoint/", so training can resume on another machine (--resume-from-checkpoint hub).
        hub_strategy="checkpoint",
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name if args.run_name is not None else (hub_name.split("/")[-1] if hub_name else None),
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

    resume = args.resume_from_checkpoint
    if resume == "hub":
        if hub_name is None:
            raise ValueError("--resume-from-checkpoint hub needs --hub-name")
        resume = os.path.join(snapshot_download(hub_name, allow_patterns="last-checkpoint/*"), "last-checkpoint")
    if resume is not None:
        # A resumed run must train on the same data, with the same tokenizer, as the run it continues.
        saved = GPT2Config.from_pretrained(resume).to_dict()
        for key, value in provenance.items():
            if key != "training_args" and key in saved and saved[key] != value:
                raise ValueError(f"This run's {key} is {value!r}, but the checkpoint was trained with {saved[key]!r}")
        print(f"\nResuming from {resume}")

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=resume)

    # Save the selected (best) model before the final evaluations, so a failure in them cannot lose it.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel and tokenizer saved to {args.output_dir}")

    # Post-training evaluation
    print("\nRunning post-training evaluation on the validation split...")
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    # Keep the final results and the full train/validation curves (trainer_state.json) with the model.
    trainer.save_metrics("eval", metrics)
    trainer.save_state()

    if hub_name:
        print(f"\nPushing model, tokenizer and results to {hub_name}...")
        trainer.push_to_hub()
    else:
        print("(pass --hub-name to also push them to the Hub)")

    if wandb.run is not None:
        # Every post-training accuracy, e.g. "val/accuracy", "val/carry_accuracy".
        for key, value in metrics.items():
            if key.startswith("eval_") and key.endswith("accuracy"):
                wandb.run.summary[f"val/{key[len('eval_') :]}"] = value
        wandb.run.summary["val/n_examples"] = len(raw_val)
