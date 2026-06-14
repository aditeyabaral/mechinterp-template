"""Build and push a character-level tokenizer to the HF Hub.

This builds a deterministic, training-free WordLevel tokenizer that maps each
character in `VOCAB_CHARS` to its own token. This pattern is useful for small
from-scratch models on synthetic tasks where the input alphabet is small and fixed.

TODO (you must implement this for your task):
  1. `VOCAB_CHARS`  -- every character that can appear in a prompt or answer.
  2. `SAMPLE_TEXTS` -- a few representative strings for the round-trip sanity
     checks at the bottom of this script.

The HF Hub repo to push to is passed via --hub-name on the command line.

See the commented-out example below, taken from the operator-overloading
arithmetic study this template was derived from.
"""

import argparse

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

# --------------------------------------------------------------------------- #
# TODO: fill these two lists in for your task.
#
#   VOCAB_CHARS:  EVERY character that can appear in any prompt or answer your model
#                 will ever see -- digits, operators, letters, spaces, newline, etc.
#                 The tokenizer gives each one its own token id. Anything you forget to
#                 list becomes the <unk> token, so be exhaustive. Use sorted(...) so the
#                 vocabulary order is deterministic across machines.
#
#   SAMPLE_TEXTS: a handful of representative strings (using only VOCAB_CHARS). The block
#                 at the bottom of this file encodes then decodes each one and checks it
#                 round-trips exactly -- a quick sanity check that your vocab is complete.
# --------------------------------------------------------------------------- #
#
# Example (arithmetic study):
# VOCAB_CHARS = sorted("0123456789$+-*/%=\n")
# SAMPLE_TEXTS = [
#     "2+3=5\n",
#     "12+34=46\n",
#     "9*1=9\n4*1=4\n2*4=8\n6*4=24\n5*6=30\n1*8=8\n9*8=72\n9*8=72\n9*0=",
# ]

VOCAB_CHARS: list[str] = []
SAMPLE_TEXTS: list[str] = []

# --------------------------------------------------------------------------- #

_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_VOCAB = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + VOCAB_CHARS)}


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a deterministic character-level tokenizer over VOCAB_CHARS.

    A tokenizer maps text <-> integer token IDs. Here we use the simplest possible scheme:
    one token per character (a WordLevel vocab plus a Split pre-tokenizer that isolates every
    single character). This is ideal for small synthetic tasks because it's transparent and has
    a tiny vocabulary -- no subword merges to reason about. The Fuse decoder just glues the
    characters back together when decoding.
    """
    tok_obj = Tokenizer(WordLevel(vocab=_VOCAB, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Split(pattern=Regex(r"[\s\S]"), behavior="isolated")
    tok_obj.decoder = Fuse()
    # Wrap in a PreTrainedTokenizerFast so it behaves like any HuggingFace tokenizer (padding,
    # special tokens, save/push_to_hub, etc.).
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",  # beginning-of-sequence
        eos_token="<eos>",  # end-of-sequence (the model learns to emit this to stop)
        pad_token="<pad>",  # fills short sequences so a batch is rectangular
        unk_token="<unk>",  # stands in for any character not in VOCAB_CHARS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push a character-level tokenizer to the HF Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        required=True,
        help="HF Hub repo to push the tokenizer to (e.g. your-username/your-project-tokenizer)",
    )
    args = parser.parse_args()

    if not VOCAB_CHARS or not SAMPLE_TEXTS:
        raise ValueError(
            "VOCAB_CHARS and SAMPLE_TEXTS are not set. "
            "Fill these in for your task -- see the module docstring for an example."
        )

    tokenizer = build_tokenizer()
    vocab = tokenizer.get_vocab()

    # Vocabulary structure
    print(f"Vocabulary ({len(tokenizer)} tokens):")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {repr(token)}")

    expected_vocab_size = len(_SPECIAL_TOKENS) + len(VOCAB_CHARS)
    assert len(tokenizer) == expected_vocab_size, f"Expected {expected_vocab_size} tokens, got {len(tokenizer)}"

    # Special token IDs match _SPECIAL_TOKENS insertion order
    assert tokenizer.pad_token_id == 0, f"pad_token_id={tokenizer.pad_token_id}, expected 0"
    assert tokenizer.bos_token_id == 1, f"bos_token_id={tokenizer.bos_token_id}, expected 1"
    assert tokenizer.eos_token_id == 2, f"eos_token_id={tokenizer.eos_token_id}, expected 2"
    assert vocab["<unk>"] == 3, f"unk id={vocab['<unk>']}, expected 3"

    # pad and eos must be distinct, otherwise DataCollatorForLanguageModeling masks out EOS during training
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id must differ from eos_token_id"

    # All token IDs are within [0, vocab_size)
    assert max(vocab.values()) == len(tokenizer) - 1, "Token IDs are not contiguous"

    # Every vocab character has its own entry and no two characters share an ID
    char_ids = [vocab[c] for c in VOCAB_CHARS]
    assert len(char_ids) == len(set(char_ids)), "Duplicate IDs among vocab characters"

    print("\nVocabulary assertions passed.")

    # Round-trip sanity checks on SAMPLE_TEXTS. For a character-level
    # tokenizer, each token should be exactly one input character.
    print("\nSanity checks:")
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {text!r:16s} -> {tokens}")

        assert tokens == list(text), (
            f"Tokenization mismatch for {text!r}:\n  expected: {list(text)}\n  got:      {tokens}"
        )
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip mismatch for {text!r}:\n  expected: {text!r}\n  got:      {decoded!r}"
        assert tokenizer.unk_token_id not in ids, f"Unknown token in encoding of {text!r}: ids={ids}"
        assert all(0 <= i < len(tokenizer) for i in ids), f"Out-of-bounds token ID in encoding of {text!r}: ids={ids}"

    print("\nAll tokenization assertions passed.")

    tokenizer.push_to_hub(args.hub_name)
    print(f"\nPushed to hub: {args.hub_name}")
