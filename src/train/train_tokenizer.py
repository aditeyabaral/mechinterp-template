"""Build and push a character-level arithmetic tokenizer to HF Hub.

Uses a fixed WordLevel vocabulary covering all arithmetic characters:
digits (0-9), operators (+, -, *, /, %, $), equals (=), newline, and 4 special tokens.

The vocabulary is deterministic and requires no training data, making it
reusable across all model families and dataset configs.
"""

import argparse

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_CHARS = sorted("0123456789$+-*/%=\n")
_VOCAB = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _CHARS)}


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a fixed character-level tokenizer for arithmetic expressions."""
    tok_obj = Tokenizer(WordLevel(vocab=_VOCAB, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Split(pattern=Regex(r"[\s\S]"), behavior="isolated")
    tok_obj.decoder = Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push arithmetic tokenizer to HF Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default="arithmetic-circuit-overloading/tokenizer-full",
        help="HF Hub repo to push the tokenizer to",
    )
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    vocab = tokenizer.get_vocab()

    # Vocabulary structure
    print(f"Vocabulary ({len(tokenizer)} tokens):")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {repr(token)}")

    expected_vocab_size = len(_SPECIAL_TOKENS) + len(_CHARS)
    assert len(tokenizer) == expected_vocab_size, f"Expected {expected_vocab_size} tokens, got {len(tokenizer)}"

    # Special token IDs match _SPECIAL_TOKENS insertion order
    assert tokenizer.pad_token_id == 0, f"pad_token_id={tokenizer.pad_token_id}, expected 0"
    assert tokenizer.bos_token_id == 1, f"bos_token_id={tokenizer.bos_token_id}, expected 1"
    assert tokenizer.eos_token_id == 2, f"eos_token_id={tokenizer.eos_token_id}, expected 2"
    assert vocab["<unk>"] == 3, f"unk id={vocab['<unk>']}, expected 3"

    # pad and eos must be distinct (otherwise DataCollator masks out EOS during training)
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id must differ from eos_token_id"

    # All token IDs are within [0, vocab_size)
    assert max(vocab.values()) == len(tokenizer) - 1, "Token IDs are not contiguous"

    # Every arithmetic character has its own entry and no two chars share an ID
    char_ids = [vocab[c] for c in _CHARS]
    assert len(char_ids) == len(set(char_ids)), "Duplicate IDs among arithmetic characters"

    print("\nVocabulary assertions passed.")

    # Per-expression tokenization
    # (text, expected_tokens)
    tests = [
        # Standard arithmetic
        ("2+3=5\n", ["2", "+", "3", "=", "5", "\n"]),
        ("7*8=56\n", ["7", "*", "8", "=", "5", "6", "\n"]),
        ("5-9=-4\n", ["5", "-", "9", "=", "-", "4", "\n"]),
        ("0*0=0\n", ["0", "*", "0", "=", "0", "\n"]),
        # Division, modulo, and concatenation
        ("8/2=4\n", ["8", "/", "2", "=", "4", "\n"]),
        ("7%3=1\n", ["7", "%", "3", "=", "1", "\n"]),
        ("9/3=3\n", ["9", "/", "3", "=", "3", "\n"]),
        ("10%4=2\n", ["1", "0", "%", "4", "=", "2", "\n"]),
        ("3$4=34\n", ["3", "$", "4", "=", "3", "4", "\n"]),
        ("12$5=125\n", ["1", "2", "$", "5", "=", "1", "2", "5", "\n"]),
        ("0$9=09\n", ["0", "$", "9", "=", "0", "9", "\n"]),
        # Multi-digit operands and results
        ("99+1=100\n", ["9", "9", "+", "1", "=", "1", "0", "0", "\n"]),
        ("5-9=-4\n", ["5", "-", "9", "=", "-", "4", "\n"]),
        # Two-digit operands
        ("12+34=46\n", ["1", "2", "+", "3", "4", "=", "4", "6", "\n"]),
        ("50-27=23\n", ["5", "0", "-", "2", "7", "=", "2", "3", "\n"]),
        ("11*11=121\n", ["1", "1", "*", "1", "1", "=", "1", "2", "1", "\n"]),
        ("84/12=7\n", ["8", "4", "/", "1", "2", "=", "7", "\n"]),
        ("99%10=9\n", ["9", "9", "%", "1", "0", "=", "9", "\n"]),
        # Three-digit operands / results
        ("100+200=300\n", ["1", "0", "0", "+", "2", "0", "0", "=", "3", "0", "0", "\n"]),
        ("500-123=377\n", ["5", "0", "0", "-", "1", "2", "3", "=", "3", "7", "7", "\n"]),
        ("123*4=492\n", ["1", "2", "3", "*", "4", "=", "4", "9", "2", "\n"]),
        ("999/3=333\n", ["9", "9", "9", "/", "3", "=", "3", "3", "3", "\n"]),
        ("256%100=56\n", ["2", "5", "6", "%", "1", "0", "0", "=", "5", "6", "\n"]),
        # Negative results with multi-digit numbers
        ("10-99=-89\n", ["1", "0", "-", "9", "9", "=", "-", "8", "9", "\n"]),
        ("100-999=-899\n", ["1", "0", "0", "-", "9", "9", "9", "=", "-", "8", "9", "9", "\n"]),
    ]

    print("\nSanity checks:")
    for text, expected_tokens in tests:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        print(f"  {repr(text):16s} -> {tokens}")

        assert tokens == expected_tokens, (
            f"Tokenization mismatch for {repr(text)}:\n  expected: {expected_tokens}\n  got:      {tokens}"
        )
        # Round-trip correctness
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text, (
            f"Round-trip mismatch for {repr(text)}:\n  expected: {repr(text)}\n  got:      {repr(decoded)}"
        )
        # No unknown tokens
        assert tokenizer.unk_token_id not in ids, f"Unknown token in encoding of {repr(text)}: ids={ids}"
        # Every token ID is within bounds
        assert all(0 <= i < len(tokenizer) for i in ids), (
            f"Out-of-bounds token ID in encoding of {repr(text)}: ids={ids}"
        )

    # Full prompt test — mirrors an actual training example (8-shot, question only, no answer)
    full_prompt = "9*1=9\n4*1=4\n2*4=8\n6*4=24\n5*6=30\n1*8=8\n9*8=72\n9*8=72\n9*0="
    prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)

    print(f"\nFull prompt ({len(prompt_ids)} tokens):")
    print(f"  {prompt_tokens}")

    # Since every character maps 1:1 to a token, expected tokens == list(full_prompt)
    assert prompt_tokens == list(full_prompt), (
        f"Full prompt tokenization mismatch:\n  expected: {list(full_prompt)}\n  got:      {prompt_tokens}"
    )
    assert tokenizer.unk_token_id not in prompt_ids, "Unknown token in full prompt"
    assert all(0 <= i < len(tokenizer) for i in prompt_ids), "Out-of-bounds token ID in full prompt"
    # Prompt ends with "=" (no newline) — last token should be "="
    assert prompt_tokens[-1] == "=", f"Last token should be '=', got {repr(prompt_tokens[-1])}"

    print("\nAll tokenization assertions passed.")

    tokenizer.push_to_hub(args.hub_name)
    print(f"\nPushed to hub: {args.hub_name}")
