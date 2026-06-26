"""Prompt dataset for inference: builds the prompts that main.py runs the model on.

A "prompt" is just the input string you feed the model. The model reads it, generates a
continuation, and we capture its internal activations while it does so. So your prompts ARE
your experiment: they decide what behaviour you get to study. This file is where you define them.

You implement `PromptDataset.generate_prompts`. A small self-contained example (single-digit
addition) is shown in comments -- uncomment it to see the pipeline run, then replace it with
prompts for your own task.
"""


class PromptDataset:
    """A collection of prompts to run activation-extraction inference on.

    `inference.run` iterates over `self.prompts`, a list of {"prompt": str, "metadata": dict}
    entries; that is the only interface the rest of the pipeline relies on. "prompt" is the string
    fed to the model; "metadata" is an optional free-form dict that can carry that prompt's ground truth
    (e.g. the operands and correct answer) so you can score the model's output downstream. It's
    passed straight through to each result row and never inspected by the pipeline, so put whatever
    you need in it (or leave it empty).
    """

    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self.prompts: list[dict] = []

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    @classmethod
    def generate_prompts(cls, num_prompts: int) -> "PromptDataset":
        """Build the list of prompts to run inference on.

        Args:
            num_prompts: How many prompts to generate (comes from `--num-prompts`).

        Returns:
            A PromptDataset whose `.prompts` is a list of `num_prompts` {"prompt": str,
            "metadata": dict} entries.
        """
        # ----------------------------------------------------------------------------------- #
        # TODO: build your prompts and append each one to `instance.prompts` as a dict
        #     {"prompt": <string>, "metadata": <dict of ground truth>}.
        #
        # WHAT a prompt should be:
        #   The model continues whatever you give it, so end the prompt right where you want the
        #   answer to begin. For arithmetic you'd end with "7+5=" so the model produces "12".
        #   Whatever the model then generates is parsed by find_answer_span in src/inference.py.
        #
        # WHAT metadata is for:
        #   A free-form dict of that prompt's ground truth (e.g. {"a": 7, "b": 5, "answer": "12"}).
        #   It's copied verbatim onto the result row, so downstream code (scoring, lasso.py) can
        #   compare the model's output against the truth. The pipeline never reads it -- use {} if
        #   you don't need it.
        #
        # FEW-SHOT prompting (used in the example below):
        #   Small models follow patterns better when you first show a few solved examples
        #   ("shots") in the same prompt, separated by newlines, before the real question. The
        #   model infers the rule from the examples. This is optional but common for tiny models.
        #
        # TIPS:
        #   - Keep answers short (ideally one or two tokens) so the analysis is clean.
        #   - Use a fixed random seed (main.py already seeds RNGs) so runs are reproducible.
        #   - If your task needs extra knobs (operator, number of shots, digit count, ...), add
        #     them as CLI arguments in src/utils/parser.py and as parameters to this method,
        #     then pass them through from src/main.py.
        # ----------------------------------------------------------------------------------- #
        #
        # Example (single-digit addition, 4-shot) -- uncomment to run the pipeline, then replace:
        #
        #     import random
        #     instance = cls()
        #     for _ in range(num_prompts):
        #         shots = []                                    # the solved examples shown first
        #         for _ in range(4):
        #             a, b = random.randint(0, 9), random.randint(0, 9)
        #             shots.append(f"{a}+{b}={a + b}")          # e.g. "7+5=12"
        #         a, b = random.randint(0, 9), random.randint(0, 9)
        #         instance.prompts.append(
        #             {
        #                 "prompt": "\n".join(shots) + f"\n{a}+{b}=",  # ends at "=", answer to come
        #                 "metadata": {"a": a, "b": b, "answer": str(a + b)},
        #             }
        #         )
        #     return instance
        #
        # ----------------------------------------------------------------------------------- #
        raise NotImplementedError("Implement PromptDataset.generate_prompts() -- see the example in comments.")
