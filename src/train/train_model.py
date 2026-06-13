"""Train a small LM from scratch on arithmetic circuit overloading data.

Architecture is inherited directly from the base model's config (default: Llama 3.3 70B).
Only the size parameters (hidden_size, num_hidden_layers, etc.) are overridden.

Loads a pre-built tokenizer from HF Hub (see train_tokenizer.py) and trains the model
on the specified dataset config, then pushes both to the same HF Hub repo.
"""

import argparse
import random

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


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


class ArithmeticTrainer(Trainer):
    """Trainer subclass that replaces the default eval loop with generation-based accuracy metrics."""

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run greedy generation on the validation set and return standard/overloaded accuracy."""
        was_training = self.model.training
        self.model.eval()
        original_padding_side = self.processing_class.padding_side
        self.processing_class.padding_side = "left"

        try:
            results = []
            for i in tqdm(range(0, len(raw_val), args.batch_size), desc="Evaluating"):
                batch = raw_val[i : i + args.batch_size]
                predictions = run_inference_batch(self.model, self.processing_class, batch["prompt"])
                for j, predicted in enumerate(predictions):
                    example = {k: batch[k][j] for k in batch}
                    results.append({**example, "predicted": predicted, "correct": predicted == example["answer"]})

            standard = [r for r in results if r["base_operation"] == r["target_operation"]]
            overloaded = [r for r in results if r["base_operation"] != r["target_operation"]]

            metrics = {
                f"{metric_key_prefix}_overall_accuracy": sum(r["correct"] for r in results) / len(results),
                f"{metric_key_prefix}_standard_accuracy": sum(r["correct"] for r in standard) / len(standard)
                if standard
                else float("nan"),
                f"{metric_key_prefix}_overloaded_accuracy": sum(r["correct"] for r in overloaded) / len(overloaded)
                if overloaded
                else float("nan"),
            }

            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics
        finally:
            self.processing_class.padding_side = original_padding_side
            if was_training:
                self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on arithmetic expressions")

    # Tokenizer
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="arithmetic-circuit-overloading/tokenizer-full",
        help="HF Hub repo to load the tokenizer from (see train_tokenizer.py)",
    )

    # Architecture
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Base model to inherit architecture config from (only size params are overridden)",
    )

    # Dataset
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="HF Hub dataset repo",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="99",
        help="Dataset config/subset to train the model on (e.g. '100', '99', '95')",
    )

    # Model size
    parser.add_argument("--hidden-size", type=int, default=256, help="Dimension of hidden representations")
    parser.add_argument("--num-hidden-layers", type=int, default=4, help="Number of transformer decoder layers")
    parser.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads per layer")
    parser.add_argument(
        "--num-key-value-heads",
        type=int,
        default=None,
        help="Number of KV heads for GQA (default: same as --num-attention-heads, i.e. MHA); must divide evenly",
    )
    parser.add_argument("--intermediate-size", type=int, default=1024, help="Dimension of the MLP representations")
    parser.add_argument("--max-position-embeddings", type=int, default=256, help="Maximum sequence length")

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
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Fraction of steps used for LR warmup")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
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

    hub_name = args.hub_name

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer
    print(f"Loading tokenizer from '{args.tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    print(f"Tokenizer vocab size : {len(tokenizer)}")
    print(f"  max token ID       : {max(tokenizer.get_vocab().values())}")
    print(f"  pad_token_id       : {tokenizer.pad_token_id}")
    print(f"  bos_token_id       : {tokenizer.bos_token_id}")
    print(f"  eos_token_id       : {tokenizer.eos_token_id}")

    # Config
    # Load config as-is, then override only the size parameters.
    config = AutoConfig.from_pretrained(args.base_model)
    config.vocab_size = len(tokenizer)
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.num_hidden_layers = args.num_hidden_layers
    config.num_attention_heads = args.num_attention_heads
    config.num_key_value_heads = args.num_key_value_heads or args.num_attention_heads
    config.max_position_embeddings = args.max_position_embeddings
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    # Model
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nDevice: {device}")
    print(f"Seed:   {args.seed}\n")

    print("Model Configuration:")
    print(f"  vocab_size:              {config.vocab_size}")
    print(f"  hidden_size:             {config.hidden_size}")
    print(f"  intermediate_size:       {config.intermediate_size}")
    print(f"  num_hidden_layers:       {config.num_hidden_layers}")
    print(f"  num_attention_heads:     {config.num_attention_heads}")
    print(f"  num_key_value_heads:     {config.num_key_value_heads}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Model size (fp32): ~{total_params * 4 / 1e9:.2f} GB")

    # Dataset
    print(f"\nLoading dataset '{args.dataset_name}' (config: {args.dataset_config})...")
    raw_train = load_dataset(args.dataset_name, args.dataset_config, split="train")
    raw_val = load_dataset(args.dataset_name, args.dataset_config, split="validation")

    # TODO: add support to set how many FS examples.
    # This currently directly uses the prompt which has all examples.
    # We need to extract N FS examples and prepend to question for training text.
    def tokenize(batch: dict) -> dict:
        """Concatenate prompt and answer, then tokenize."""
        texts = [p + a + tokenizer.eos_token for p, a in zip(batch["prompt"], batch["answer"])]
        return tokenizer(texts, truncation=True, max_length=args.max_position_embeddings)

    tokenized_train = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)

    # Training
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
        metric_for_best_model="eval_overloaded_accuracy",
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

    trainer = ArithmeticTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=raw_val,
        data_collator=collator,
        processing_class=tokenizer,
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
        wandb.run.summary["val/overall_accuracy"] = metrics["eval_overall_accuracy"]
        wandb.run.summary["val/standard_accuracy"] = metrics["eval_standard_accuracy"]
        wandb.run.summary["val/overloaded_accuracy"] = metrics["eval_overloaded_accuracy"]
        wandb.run.summary["val/n_standard"] = len([r for r in raw_val if r["base_operation"] == r["target_operation"]])
        wandb.run.summary["val/n_overloaded"] = len(
            [r for r in raw_val if r["base_operation"] != r["target_operation"]]
        )
        wandb.run.summary["val/n_neurons"] = args.num_hidden_layers * (
            args.intermediate_size + args.num_attention_heads
        )
