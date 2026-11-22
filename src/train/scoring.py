"""How an answer is scored correct -- the one rule training, inference and analysis all use.

An answer is correct only if the model generates exactly the answer's tokens and then <eos>. Training
picks its best checkpoint by it (src/train/train_model.py), inference.py marks each generated answer
with it, and the ablation sweep's teacher forcing applies the same rule one step at a time
(inference.whole_answer_correct), so all three agree on what "correct" means.
"""


def is_correct(generated_ids: list[int], answer_ids: list[int], eos_token_id: int) -> bool:
    """Whether the generated tokens are exactly the answer's tokens followed by <eos>.

    Comparing token ids, not decoded text, makes this exact and task-independent: there are no
    choices about stripping whitespace or where the answer ends. Requiring <eos> means a model that
    writes the right answer but doesn't stop is scored wrong, as it should be.

    Args:
        generated_ids: The token ids the model generated after the prompt.
        answer_ids: The answer's token ids, tokenized without special tokens (as in training).
        eos_token_id: The tokenizer's <eos> id.
    """
    return eos_token_id in generated_ids and generated_ids[: generated_ids.index(eos_token_id)] == answer_ids
