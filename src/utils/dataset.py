"""Dataset utilities for generating math problem prompts with operator overloading."""

import random

from tqdm.auto import tqdm


class Prompt:
    """Class representing a math problem prompt with operator overloading."""

    def __init__(
        self,
        question: str,
        examples_original: list[str],
        examples_overloaded: list[str],
        answer_original: str,
        answer_overloaded: str,
    ) -> None:
        """Initialize the Prompt class with question, examples, and answers."""
        self.question = question
        self.examples_original = examples_original
        self.examples_overloaded = examples_overloaded
        self.answer_original = answer_original
        self.answer_overloaded = answer_overloaded


class PromptDataset:
    """Class for generating and storing a dataset of math problem prompts with operator overloading."""

    OPERATOR_SYMBOL_TO_NAME_MAP = {
        "+": "plus",
        "-": "minus",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "$": "concat",
    }
    OPERATOR_NAME_TO_SYMBOL_MAP = {v: k for k, v in OPERATOR_SYMBOL_TO_NAME_MAP.items()}

    def __init__(self) -> None:
        """Initialize the PromptDataset class."""
        self.prompts = list()

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    @staticmethod
    def generate_operand(min_op: int = 0, max_op: int = 9) -> int:
        """Generate a random operand for the math problem.

        We set max_op to 9 to keep the final answer a single token in the vocabulary of most language models.

        Args:
            min_op (int): The minimum value for the operand. Defaults to 0.
            max_op (int): The maximum value for the operand. Defaults to 9.
        """
        return random.randint(min_op, max_op)

    @staticmethod
    def _apply_op(a: int, b: int, op: str) -> str:
        """Apply operator `op` to operands a and b and return the string result.

        Handles all supported operators including non-Python-eval-able ones ($, /).
        `/` uses floor division (//). `$` concatenates digit strings.
        """
        if op == "$":
            return str(a) + str(b)
        if op == "/":
            return str(a // b)
        return str(eval(f"{a} {op} {b}"))

    @staticmethod
    def compute_result_with_overloaded_operator(
        a: int,
        b: int,
        overloading: bool,
        operator: str,
        overloading_operator: str | None,
    ) -> tuple[str, str]:
        """Compute the result of the operation with the overloaded operator.

        Args:
            a (int): The first operand.
            b (int): The second operand.
            overloading (bool): Whether operator overloading is applied.
            operator (str): The original operator used in the equation.
            overloading_operator (str): The overloading operator to use in the equation.

        Returns:
            tuple[str, str]: A tuple containing the original answer and the overloaded answer.
        """
        answer_original = PromptDataset._apply_op(a, b, operator)
        if overloading:
            if overloading_operator not in PromptDataset.OPERATOR_SYMBOL_TO_NAME_MAP:
                raise NotImplementedError(f"Overloading operator '{overloading_operator}' is not implemented.")
            answer_overloaded = PromptDataset._apply_op(a, b, overloading_operator)
        else:
            answer_overloaded = None

        return answer_original, answer_overloaded

    @staticmethod
    def generate_equation(
        overloading: bool,
        operator: str,
        overloading_operator: str | None,
        max_digits: int = 1,
        a: int | None = None,
        b: int | None = None,
        pad_zero: bool = False,
    ) -> tuple[str, str, str | None]:
        """Generate a random equation using the defined operators.

        Args:
            overloading (bool): Whether operator overloading is applied.
            operator (str): The original operator used in the equation.
            overloading_operator (Optional[str]): The overloading operator to use in the equation.
            max_digits (int): Maximum number of digits for each operand. Defaults to 1.
            a (int | None): If provided, use this value as the first operand instead of sampling.
            b (int | None): If provided, use this value as the second operand instead of sampling.
            pad_zero (bool): If True, pad operands in the printed expression with leading
                zeros up to `max_digits`. Answers are computed from integer values and
                are NOT padded. Defaults to False.

        Returns:
            tuple[str, str, str]: A tuple containing the generated expression, the original answer,
            and the overloaded answer.
        """
        max_op = 10**max_digits - 1
        if a is None:
            a = PromptDataset.generate_operand(max_op=max_op)
        needs_nonzero_b = {"/", "%"}
        min_op_b = 1 if (operator in needs_nonzero_b or overloading_operator in needs_nonzero_b) else 0

        if b is None:
            b = PromptDataset.generate_operand(min_op=min_op_b, max_op=max_op)
        # Keep answers computed from integer values, but optionally pad
        # operands in the printed expression to `max_digits` using leading zeros.
        if pad_zero and max_digits > 1:
            a_str = str(a).zfill(max_digits)
            b_str = str(b).zfill(max_digits)
            expression = f"{a_str}{operator}{b_str}"
        else:
            expression = f"{a}{operator}{b}"
        answer_original, answer_overloaded = PromptDataset.compute_result_with_overloaded_operator(
            a, b, overloading, operator, overloading_operator
        )
        return expression, answer_original, answer_overloaded

    @classmethod
    def generate_prompts(
        cls,
        num_prompts: int,
        few_shot_examples: int,
        overloading: bool,
        operator: str,
        overloading_operator: str | None,
        max_digits: int = 1,
        reverse: bool = False,
        pad_zero: bool = False,
    ) -> "PromptDataset":
        """Generate a list of math problem prompts and return a PromptDataset instance.

        Args:
            num_prompts (int): The number of prompts to generate.
            few_shot_examples (int): The number of few-shot examples to include in each prompt.
            overloading (bool): Whether operator overloading is applied.
            operator (str): The original operator used in the equation.
            overloading_operator (str | None): The overloading operator to use in the equation.
                Required if overloading is True.
            max_digits (int): Maximum number of digits for each operand. Defaults to 1.
            reverse (bool): If True, reverse digit order of all answers (LSD-first), following
                arxiv.org/abs/2307.03381. Negative sign is placed last (e.g. -46 -> "64-").
            pad_zero (bool): If True, pad operands in the printed expression with leading
                zeros up to `max_digits`. Answers are computed from integer values and
                are NOT padded. Defaults to False.
        """
        if overloading and overloading_operator is None:
            raise ValueError("overloading_operator must be specified when overloading is True")

        def _fmt(answer: str) -> str:
            return answer[::-1] if reverse else answer

        instance = cls()
        for _ in tqdm(range(num_prompts), desc="Generating prompts"):
            examples_original = list()
            examples_overloaded = list()
            for _ in range(few_shot_examples):
                expression, answer_original, answer_overloaded = cls.generate_equation(
                    overloading,
                    operator,
                    overloading_operator,
                    max_digits=max_digits,
                    pad_zero=pad_zero,
                )
                equation_original = f"{expression}={_fmt(answer_original)}"
                examples_original.append(equation_original)
                if answer_overloaded is not None:
                    equation_overloaded = f"{expression}={_fmt(answer_overloaded)}"
                    examples_overloaded.append(equation_overloaded)

            expression, answer_original, answer_overloaded = cls.generate_equation(
                overloading,
                operator,
                overloading_operator,
                max_digits=max_digits,
                pad_zero=pad_zero,
            )
            prompt = Prompt(
                question=expression,
                examples_original=examples_original,
                examples_overloaded=examples_overloaded,
                answer_original=_fmt(answer_original),
                answer_overloaded=_fmt(answer_overloaded) if answer_overloaded is not None else None,
            )
            instance.prompts.append(prompt)
        return instance
