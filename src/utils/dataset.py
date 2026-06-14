"""Prompt dataset for inference: builds the prompts that main.py runs the model on.

Implement `PromptDataset.generate_prompts` to construct the prompts for your study.
Prompts are plain strings (what `inference.run` consumes). The operator-overloading
arithmetic study this template was derived from is shown in comments.
"""


class PromptDataset:
    """A collection of prompt strings to run activation-extraction inference on."""

    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self.prompts: list[str] = []

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    @classmethod
    def generate_prompts(cls, num_prompts: int) -> "PromptDataset":
        """Build a dataset of prompts to run inference on.

        Add any task-specific parameters (operator, few-shot count, ...) to this
        method's signature and pass them from main.py / your CLI args.

        Args:
            num_prompts: Number of prompts to generate.

        Returns:
            A PromptDataset whose `.prompts` is a list of prompt strings.
        """
        # ----------------------------------------------------------------------- #
        # TODO: build your prompts here -- append prompt strings to instance.prompts.
        # ----------------------------------------------------------------------- #
        #
        # Example (single-digit addition, few-shot):
        #
        #     import random
        #     instance = cls()
        #     for _ in range(num_prompts):
        #         shots = []
        #         for _ in range(4):
        #             a, b = random.randint(0, 9), random.randint(0, 9)
        #             shots.append(f"{a}+{b}={a + b}")
        #         a, b = random.randint(0, 9), random.randint(0, 9)
        #         instance.prompts.append("\n".join(shots) + f"\n{a}+{b}=")
        #     return instance
        #
        # Example (operator-overloading study): for each item generate a 'standard'
        # few-shot context and an 'overloaded' one (few-shot answers follow a
        # different operator's semantics), so the model's behaviour can be contrasted
        # between the two contexts on the same final question.
        #
        # ----------------------------------------------------------------------- #
        raise NotImplementedError("Implement PromptDataset.generate_prompts() -- see the example in comments.")
